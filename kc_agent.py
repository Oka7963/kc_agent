#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kc_agent.py

Main decision/orchestration process for KC_AGENT.

Data flow:
    POI plugin / battle_receiver.py  -> event_q -> kc_agent.py
    scene_identify.py                -> event_q -> kc_agent.py
    kc_agent.py                      -> command_q -> mouse/keyboard executor
    mouse/keyboard executor          -> event_q -> kc_agent.py

This file is the workflow owner.  It receives normalized battle/map events,
scene events, action results, and user interrupts.  It decides when to wait
for scenes and when to emit semantic mouse/keyboard commands.
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Optional

from battle_receiver.battle_receiver import (
    DEFAULT_HOST as DEFAULT_RECEIVER_HOST,
    DEFAULT_NORMALIZED_LOG_PATH,
    DEFAULT_PORT as DEFAULT_RECEIVER_PORT,
    DEFAULT_RAW_LOG_PATH,
    ReceiverConfig,
    make_handler as make_battle_receiver_handler,
)
from utility.logger import setup_logger


logger = setup_logger(name="kc_agent")
ACTION_SCENE_CACHE_TTL_MS = 15000
DEFAULT_ACTION_POLICY_PATH = Path("utility/action_policy.json")


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def agent_event_from_packet(packet: dict[str, Any]) -> AgentEvent:
    return AgentEvent(
        type=packet["type"],
        payload=packet.get("payload") or {},
        event_id=packet.get("event_id", new_id("poi")),
        ts_ms=to_int(packet.get("ts_ms"), now_ms()) or now_ms(),
        source=packet.get("source", "poi"),
        correlation=packet.get("correlation") or {},
    )


@dataclass
class AgentEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_id("evt"))
    ts_ms: int = field(default_factory=now_ms)
    source: str = "unknown"
    correlation: dict[str, Any] = field(default_factory=dict)


@dataclass
class WaitSpec:
    wait_id: str
    expected_scene: str
    required_targets: list[str] = field(default_factory=list)
    trigger_event_id: Optional[str] = None
    timeout_ms: int = 8000
    stable_frames: int = 2
    reason: str = ""
    correlation: dict[str, Any] = field(default_factory=dict)


@dataclass
class Command:
    command_id: str
    command: str
    target: str
    reason: str
    trigger_event_id: Optional[str] = None
    wait_id: Optional[str] = None
    requires_scene: Optional[str] = None
    safety: dict[str, Any] = field(default_factory=dict)
    correlation: dict[str, Any] = field(default_factory=dict)
    created_at_ms: int = field(default_factory=now_ms)
    valid_ms: int = 5000


class AgentState(str, enum.Enum):
    IDLE = "IDLE"
    SORTIE_STARTING = "SORTIE_STARTING"
    MAP_MOVING = "MAP_MOVING"
    MAP_NODE_ARRIVED = "MAP_NODE_ARRIVED"
    WAIT_COMPASS_OR_PRODUCTION = "WAIT_COMPASS_OR_PRODUCTION"
    WAIT_FORMATION = "WAIT_FORMATION"
    IN_DAY_BATTLE = "IN_DAY_BATTLE"
    WAIT_NIGHT_CHOICE = "WAIT_NIGHT_CHOICE"
    IN_NIGHT_BATTLE = "IN_NIGHT_BATTLE"
    WAIT_BATTLE_RESULT = "WAIT_BATTLE_RESULT"
    WAIT_RESULT_CONFIRM = "WAIT_RESULT_CONFIRM"
    WAIT_ADVANCE_OR_RETREAT = "WAIT_ADVANCE_OR_RETREAT"
    RETREAT_REQUIRED = "RETREAT_REQUIRED"
    WAIT_ACTION_RESULT = "WAIT_ACTION_RESULT"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


@dataclass
class RuntimeContext:
    state: AgentState = AgentState.IDLE
    sortie_id: Optional[str] = None
    battle_id: Optional[str] = None
    map_node_id: Optional[str] = None
    current_world: Optional[str] = None
    current_route_no: Optional[int] = None
    current_node_type: str = "unknown"
    current_is_boss_node: bool = False
    current_deck_id: Optional[int] = None
    current_fleet: dict[str, Any] = field(default_factory=dict)
    battle_progress: str = "-"
    taiha_latch: bool = False
    taiha_ships: list[dict[str, Any]] = field(default_factory=list)
    pending_wait_id: Optional[str] = None
    pending_scene: Optional[str] = None
    pending_command_id: Optional[str] = None
    pending_command_target: Optional[str] = None
    pending_command_reason: Optional[str] = None
    last_event_id: Optional[str] = None
    last_event_type: str = "-"
    last_battle_result: Optional[dict[str, Any]] = None
    last_battle_result_event_id: Optional[str] = None
    battle_result_confirm_clicked: bool = False
    drop_check_clicked: bool = False
    scene_ready_cache: dict[str, AgentEvent] = field(default_factory=dict)
    latest_scene: Optional[dict[str, Any]] = None
    action_history: set[str] = field(default_factory=set)
    user_paused: bool = False


def damage_state(now_hp: Optional[int], max_hp: Optional[int]) -> str:
    if now_hp is None or max_hp is None or max_hp <= 0:
        return "unknown"
    if now_hp <= 0:
        return "zero_or_sunk"
    if now_hp * 4 <= max_hp:
        return "taiha"
    if now_hp * 2 <= max_hp:
        return "chuuha"
    if now_hp < max_hp:
        return "shouha_or_scratch"
    return "normal"


def analyze_damage_from_fleet(fleet: dict[str, Any]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for ship in fleet.get("ships") or []:
        now_hp = to_int(ship.get("battle_end_hp"), ship.get("now_hp"))
        max_hp = to_int(ship.get("max_hp"))
        state = damage_state(now_hp, max_hp)
        ratio_x1000 = int(now_hp * 1000 / max_hp) if now_hp is not None and max_hp else None
        reports.append({
            "pos": ship.get("pos") or ship.get("position"),
            "instance_id": ship.get("instance_id"),
            "ship_id": ship.get("ship_id"),
            "now_hp": now_hp,
            "max_hp": max_hp,
            "state": state,
            "ratio_x1000": ratio_x1000,
        })
    taiha = [s for s in reports if s["state"] == "taiha"]
    chuuha = [s for s in reports if s["state"] == "chuuha"]
    return {
        "has_taiha": bool(taiha),
        "has_chuuha": bool(chuuha),
        "ships": reports,
        "taiha_ships": taiha,
        "chuuha_ships": chuuha,
    }


def has_drop(drop: dict[str, Any]) -> bool:
    return any(drop.get(key) for key in ("ship_id", "item", "event_item"))


def load_action_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("action policy not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def target_visible(scene: dict[str, Any], target: str) -> bool:
    targets = scene.get("targets") or {}
    info = targets.get(target)
    return isinstance(info, dict) and bool(info.get("visible", True))


class MainDecisionStateMachine:
    """Combine battle state, scene state, and policy config into an action."""

    def __init__(self, action_policy: dict[str, Any]) -> None:
        self.action_policy = action_policy

    def action_to_target(self, action: str) -> Optional[str]:
        return (self.action_policy.get("button_targets") or {}).get(action)

    def decide(self, ctx: RuntimeContext) -> tuple[Optional[str], Optional[str]]:
        scene = ctx.latest_scene or {}
        scene_name = scene.get("scene")
        battle = (ctx.last_battle_result or {}).get("battle") or {}
        damage = (ctx.last_battle_result or {}).get("damage") or {}

        scene_actions = self.action_policy.get("scene_actions") or {}
        if scene_name in scene_actions:
            action = scene_actions[scene_name]
            return action, self.action_to_target(action)

        is_boss = bool(battle.get("boss")) or ctx.current_is_boss_node or ctx.current_node_type == "boss_battle"
        if scene_name == "night_battle_choice" and is_boss and self.action_policy.get("戰鬥類型:王點") == "進夜戰":
            action = "進夜戰"
            return action, self.action_to_target(action)

        if scene_name == "advance_or_retreat" and damage.get("has_taiha") and self.action_policy.get("大破") == "撤退":
            action = "撤退"
            return action, self.action_to_target(action)

        return None, None


def classify_node(body: dict[str, Any]) -> tuple[str, bool, bool]:
    route_no = to_int(body.get("api_no"))
    color_no = to_int(body.get("api_color_no"))
    event_id = to_int(body.get("api_event_id"))
    event_kind = to_int(body.get("api_event_kind"))
    boss_cell_no = to_int(body.get("api_bosscell_no"))
    is_boss = (
        route_no is not None and boss_cell_no is not None and route_no == boss_cell_no
    ) or color_no == 5 or event_id == 5
    is_battle = is_boss or color_no in (4, 5) or event_kind in (1, 2, 3, 4) or bool(body.get("api_e_deck_info"))
    if is_boss:
        return "boss_battle", True, True
    if is_battle:
        return "battle", True, False
    if color_no == 2:
        return "resource", False, False
    if color_no == 3:
        return "maelstrom", False, False
    if color_no == 6:
        return "nothing_happened", False, False
    if color_no == 8:
        return "port", False, False
    if color_no == 9:
        return "air_recon", False, False
    return "unknown", False, False


def active_fleet_from_snapshot(raw: dict[str, Any], deck_id: int = 1) -> dict[str, Any]:
    fleets = raw.get("fleet_snapshot") or []
    selected = next((f for f in fleets if to_int(f.get("deck_id")) == deck_id), None)
    if selected is None and fleets:
        selected = fleets[0]
    if not selected:
        return {"deck_id": deck_id, "ship_count": 0, "ships": []}

    ships = []
    for ship in selected.get("ships") or []:
        ships.append({
            "pos": ship.get("position") or ship.get("pos"),
            "instance_id": ship.get("instance_id"),
            "ship_id": ship.get("ship_id"),
            "level": ship.get("level"),
            "now_hp": ship.get("now_hp"),
            "max_hp": ship.get("max_hp"),
            "cond": ship.get("cond"),
            "fuel": ship.get("fuel"),
            "bullet": ship.get("bullet"),
            "slot": ship.get("slot") or [],
        })

    sortie = raw.get("sortie_snapshot") or {}
    return {
        "deck_id": selected.get("deck_id", deck_id),
        "combined_flag": sortie.get("combined_flag"),
        "sortie_status": sortie.get("sortie_status") or [],
        "escaped_pos": sortie.get("escaped_pos") or [],
        "ship_count": selected.get("ship_count", len(ships)),
        "ships": ships,
    }


def normalize_poi_raw_event(raw: dict[str, Any]) -> Optional[AgentEvent]:
    """Convert raw POI exporter event into AgentEvent."""
    if "event_type" in raw:
        return AgentEvent(
            type=raw["event_type"],
            payload=raw,
            event_id=raw.get("event_id", new_id("poi")),
            ts_ms=to_int(raw.get("ts_ms"), now_ms()) or now_ms(),
            source=raw.get("source", "poi"),
            correlation=raw.get("correlation") or {},
        )

    name = raw.get("event")
    path = raw.get("path")
    phase = raw.get("phase")
    body = raw.get("response_body") or {}
    req = raw.get("request_body") or {}
    seq = raw.get("seq")
    ts_ms = to_int(raw.get("exported_at"), now_ms()) or now_ms()

    if name == "map_start_request" or (path == "/kcsapi/api_req_map/start" and phase == "request"):
        area = to_int(req.get("api_maparea_id"))
        info_no = to_int(req.get("api_mapinfo_no"))
        deck_id = to_int(req.get("api_deck_id"), 1) or 1
        world = f"{area}-{info_no}" if area and info_no else None
        payload = {
            "schema": "kc.agent.battle_event.v1",
            "event_type": "sortie_start_requested",
            "phase": "request",
            "seq": seq,
            "source_event": raw.get("source"),
            "path": path,
            "map": {"area": area, "info_no": info_no, "world": world},
            "fleet": active_fleet_from_snapshot(raw, deck_id),
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": False, "ui_wait_reason": "waiting_map_start_response", "next_scenes": []},
        }
        return AgentEvent("sortie_start_requested", payload, new_id("poi"), ts_ms, "poi")

    if name in ("map_start_response", "map_next_response"):
        area = to_int(body.get("api_maparea_id"))
        info_no = to_int(body.get("api_mapinfo_no"))
        world = f"{area}-{info_no}" if area and info_no else None
        node_type, is_battle, is_boss = classify_node(body)
        deck_id = to_int(req.get("api_deck_id"), 1) or 1
        source_kind = "map_start" if path == "/kcsapi/api_req_map/start" else "map_next"
        next_scenes = []
        if to_int(body.get("api_rashin_flg")) == 1:
            next_scenes.append({"scene": "compass_or_map_production", "timeout_ms": 8000})
        if is_battle:
            next_scenes.append({"scene": "formation_select", "timeout_ms": 10000})
        payload = {
            "schema": "kc.agent.battle_event.v1",
            "event_type": "map_node_arrived",
            "phase": "map",
            "seq": seq,
            "source_event": raw.get("source"),
            "path": path,
            "source_kind": source_kind,
            "map": {
                "area": area,
                "info_no": info_no,
                "world": world,
                "route_no": to_int(body.get("api_no")),
                "from_route_no": to_int(body.get("api_from_no")),
                "is_first_node_after_sortie_start": path == "/kcsapi/api_req_map/start" and to_int(body.get("api_from_no")) == 0,
                "color_no": to_int(body.get("api_color_no")),
                "event_id": to_int(body.get("api_event_id")),
                "event_kind": to_int(body.get("api_event_kind")),
                "node_type": node_type,
                "node_type_string": node_type,
                "is_battle_node": is_battle,
                "is_boss_node": is_boss,
                "boss_cell_no": to_int(body.get("api_bosscell_no")),
                "next": to_int(body.get("api_next")),
                "rashin_flg": to_int(body.get("api_rashin_flg")),
                "rashin_id": to_int(body.get("api_rashin_id")),
            },
            "fleet": active_fleet_from_snapshot(raw, deck_id),
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": bool(next_scenes), "ui_wait_reason": "sortie_first_node_arrived" if source_kind == "map_start" else "advanced_to_next_node", "next_scenes": next_scenes},
        }
        return AgentEvent("map_node_arrived", payload, new_id("poi"), ts_ms, "poi")

    if name and name.endswith("_battle_request"):
        payload = {
            "schema": "kc.agent.battle_event.v1",
            "event_type": "formation_selected",
            "phase": "formation",
            "seq": seq,
            "source_event": raw.get("source"),
            "path": path,
            "battle": {"kind": "normal_day", "formation": {"requested": to_int(req.get("api_formation")), "friendly": None, "enemy": None, "engagement": None}},
            "fleet": active_fleet_from_snapshot(raw, 1),
            "expected": {"user_input_expected": False, "ui_wait_reason": "battle_started", "next_scenes": []},
        }
        return AgentEvent("formation_selected", payload, new_id("poi"), ts_ms, "poi")

    if name == "sortie_battle_response":
        formation = body.get("api_formation") or []
        can_night = to_int(body.get("api_midnight_flag")) == 1
        next_scenes = []
        if can_night:
            next_scenes.append({"scene": "night_battle_choice", "timeout_ms": 10000})
        payload = {
            "schema": "kc.agent.battle_event.v1",
            "event_type": "day_battle_received",
            "phase": "battle_day",
            "seq": seq,
            "source_event": raw.get("source"),
            "path": path,
            "battle": {
                "kind": "normal_day",
                "deck_id": to_int(body.get("api_deck_id")),
                "formation": {
                    "requested": to_int(req.get("api_formation")),
                    "friendly": to_int(formation[0]) if len(formation) > 0 else None,
                    "enemy": to_int(formation[1]) if len(formation) > 1 else None,
                    "engagement": to_int(formation[2]) if len(formation) > 2 else None,
                },
                "can_night_battle": can_night,
                "phases": {
                    "kouku": bool(body.get("api_kouku")),
                    "support": bool(body.get("api_support_info")),
                    "opening_taisen": bool(body.get("api_opening_taisen")),
                    "opening_torpedo": bool(body.get("api_opening_atack")),
                    "hougeki1": bool(body.get("api_hougeki1")),
                    "hougeki2": bool(body.get("api_hougeki2")),
                    "hougeki3": bool(body.get("api_hougeki3")),
                    "raigeki": bool(body.get("api_raigeki")),
                },
            },
            "fleet": active_fleet_from_snapshot(raw, 1),
            "enemy": {"ship_ids": body.get("api_ship_ke") or [], "start_hp": body.get("api_e_nowhps") or [], "max_hp": body.get("api_e_maxhps") or []},
            "expected": {"user_input_expected": can_night, "ui_wait_reason": "can_night_battle" if can_night else "waiting_battle_result", "next_scenes": next_scenes},
        }
        return AgentEvent("day_battle_received", payload, new_id("poi"), ts_ms, "poi")

    if name == "battle_result" or raw.get("phase") == "poi_battle_result":
        result = raw.get("battle_result") or raw.get("result") or {}
        fleet = active_fleet_from_snapshot(raw, 1)
        deck_ship_ids = result.get("deck_ship_id") or result.get("deckShipId") or []
        deck_hp = result.get("deck_hp") or result.get("deckHp") or []
        deck_init_hp = result.get("deck_init_hp") or result.get("deckInitHp") or []
        hp_by_iid: dict[int, dict[str, Optional[int]]] = {}
        for idx, iid in enumerate(deck_ship_ids):
            iid_int = to_int(iid)
            if iid_int is None:
                continue
            hp_by_iid[iid_int] = {"battle_end_hp": to_int(deck_hp[idx]) if idx < len(deck_hp) else None, "battle_start_hp": to_int(deck_init_hp[idx]) if idx < len(deck_init_hp) else None}
        for ship in fleet.get("ships", []):
            iid = to_int(ship.get("instance_id"))
            if iid in hp_by_iid:
                ship.update(hp_by_iid[iid])
        damage = analyze_damage_from_fleet(fleet)
        drop = {"ship_id": result.get("drop_ship_id") or result.get("dropShipId"), "item": result.get("drop_item") or result.get("dropItem"), "event_item": result.get("event_item") or result.get("eventItem")}
        next_scenes = [{"scene": "battle_result_confirm", "timeout_ms": 8000}]
        if has_drop(drop):
            next_scenes.append({"scene": "drop_check", "timeout_ms": 10000})
        next_scenes.append({"scene": "advance_or_retreat", "timeout_ms": 10000})
        payload = {
            "schema": "kc.agent.battle_event.v1",
            "event_type": "battle_result",
            "phase": "result",
            "seq": seq,
            "source_event": raw.get("source"),
            "path": path,
            "battle": {"rank": result.get("rank"), "boss": result.get("boss"), "map": result.get("map"), "map_cell": result.get("map_cell") or result.get("mapCell"), "enemy_name": result.get("enemy"), "mvp": result.get("mvp") or [], "drop": drop},
            "fleet": fleet,
            "damage": damage,
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": True, "ui_wait_reason": "battle_result_confirm_then_drop_check_then_advance_or_retreat" if has_drop(drop) else "battle_result_confirm_then_advance_or_retreat", "next_scenes": next_scenes},
        }
        return AgentEvent("battle_result", payload, new_id("poi"), ts_ms, "poi")

    return None


class KcAgent:
    def __init__(
        self,
        event_q: asyncio.Queue[AgentEvent],
        scene_wait_q: asyncio.Queue[WaitSpec],
        command_q: asyncio.Queue[Command],
        status_sink: Optional[Callable[[dict[str, Any]], None]] = None,
        action_policy: Optional[dict[str, Any]] = None,
    ) -> None:
        self.event_q = event_q
        self.scene_wait_q = scene_wait_q
        self.command_q = command_q
        self.ctx = RuntimeContext()
        self.status_sink = status_sink
        self.action_policy = action_policy or {}
        self.decision_sm = MainDecisionStateMachine(self.action_policy)

    async def run(self) -> None:
        logger.info("main loop started")
        self.publish_status()
        while True:
            event = await self.event_q.get()
            try:
                await self.handle_event(event)
            except Exception as exc:
                self.ctx.state = AgentState.ERROR
                logger.exception("exception while handling %s: %r", event.type, exc)
            finally:
                self.publish_status()
                self.event_q.task_done()

    async def handle_event(self, event: AgentEvent) -> None:
        self.ctx.last_event_id = event.event_id
        self.ctx.last_event_type = event.type
        fleet = (event.payload or {}).get("fleet") or {}
        if fleet:
            self.ctx.current_fleet = fleet
        if self.ctx.user_paused and event.type != "user_interrupt":
            logger.info("paused; ignoring event=%s", event.type)
            return
        logger.info("state=%s event=%s", self.ctx.state, event.type)
        handlers = {
            "sortie_start_requested": self.on_sortie_start_requested,
            "map_node_arrived": self.on_map_node_arrived,
            "formation_selected": self.on_formation_selected,
            "day_battle_received": self.on_day_battle_received,
            "night_battle_received": self.on_night_battle_received,
            "fleet_state_updated": self.on_fleet_state_updated,
            "battle_result": self.on_battle_result,
            "scene_ready": self.on_scene_ready,
            "scene_analyzed": self.on_scene_analyzed,
            "scene_timeout": self.on_scene_timeout,
            "action_result": self.on_action_result,
            "user_interrupt": self.on_user_interrupt,
        }
        handler = handlers.get(event.type)
        if handler is None:
            logger.warning("unhandled event type=%s", event.type)
            return
        await handler(event)

    def build_status_snapshot(self) -> dict[str, Any]:
        return {
            "state": self.ctx.state.value,
            "world": self.ctx.current_world or "-",
            "route_no": self.ctx.current_route_no,
            "node_type": self.ctx.current_node_type,
            "is_boss_node": self.ctx.current_is_boss_node,
            "deck_id": self.ctx.current_deck_id,
            "fleet": self.ctx.current_fleet,
            "last_event_type": self.ctx.last_event_type,
            "last_event_id": self.ctx.last_event_id,
            "battle_progress": self.ctx.battle_progress,
            "pending_scene": self.ctx.pending_scene or "-",
            "pending_wait_id": self.ctx.pending_wait_id,
            "pending_command_target": self.ctx.pending_command_target or "-",
            "pending_command_reason": self.ctx.pending_command_reason or "-",
            "taiha_latch": self.ctx.taiha_latch,
            "taiha_ships": self.ctx.taiha_ships,
            "latest_scene": self.ctx.latest_scene or {},
            "updated_at_ms": now_ms(),
        }

    def publish_status(self) -> None:
        if self.status_sink is None:
            return
        try:
            self.status_sink(self.build_status_snapshot())
        except Exception as exc:
            logger.warning("failed to publish GUI status: %r", exc)

    async def on_sortie_start_requested(self, event: AgentEvent) -> None:
        map_info = event.payload.get("map") or {}
        fleet = event.payload.get("fleet") or {}
        self.ctx.sortie_id = new_id("sortie")
        self.ctx.battle_id = None
        self.ctx.map_node_id = None
        self.ctx.current_world = map_info.get("world")
        self.ctx.current_is_boss_node = False
        self.ctx.current_deck_id = to_int(fleet.get("deck_id"), 1)
        self.ctx.current_fleet = fleet
        self.ctx.battle_progress = "-"
        self.ctx.taiha_latch = False
        self.ctx.taiha_ships.clear()
        self.ctx.pending_wait_id = None
        self.ctx.pending_scene = None
        self.ctx.pending_command_id = None
        self.ctx.pending_command_target = None
        self.ctx.pending_command_reason = None
        self.ctx.last_battle_result = None
        self.ctx.last_battle_result_event_id = None
        self.ctx.battle_result_confirm_clicked = False
        self.ctx.drop_check_clicked = False
        self.ctx.scene_ready_cache.clear()
        self.ctx.action_history.clear()
        self.ctx.latest_scene = None
        self.ctx.state = AgentState.SORTIE_STARTING
        logger.info(
            "sortie requested: sortie_id=%s world=%s deck=%s",
            self.ctx.sortie_id,
            self.ctx.current_world,
            self.ctx.current_deck_id,
        )

    async def on_map_node_arrived(self, event: AgentEvent) -> None:
        map_info = event.payload.get("map") or {}
        expected = event.payload.get("expected") or {}
        next_scenes = expected.get("next_scenes") or []
        self.ctx.current_world = map_info.get("world") or self.ctx.current_world
        self.ctx.current_route_no = to_int(map_info.get("route_no"))
        self.ctx.current_node_type = map_info.get("node_type", "unknown")
        self.ctx.current_is_boss_node = bool(map_info.get("is_boss_node"))
        self.ctx.current_fleet = event.payload.get("fleet") or self.ctx.current_fleet
        self.ctx.battle_progress = "-"
        self.ctx.map_node_id = f"{self.ctx.current_world}-route-{self.ctx.current_route_no}"
        self.ctx.state = AgentState.MAP_NODE_ARRIVED
        logger.info(
            "node arrived: %s type=%s rashin=%s/%s",
            self.ctx.map_node_id,
            self.ctx.current_node_type,
            map_info.get("rashin_flg"),
            map_info.get("rashin_id"),
        )
        if next_scenes:
            first_scene = (next_scenes[0] or {}).get("scene")
            self.ctx.pending_scene = first_scene
            self.ctx.state = AgentState.WAIT_FORMATION if first_scene == "formation_select" else AgentState.WAIT_COMPASS_OR_PRODUCTION
        else:
            logger.info("no expected scene; waiting for further POI event")

    async def on_formation_selected(self, event: AgentEvent) -> None:
        self.ctx.battle_id = new_id("battle")
        self.clear_action_scene_cache()
        self.ctx.battle_progress = "day_battle_starting"
        self.ctx.state = AgentState.IN_DAY_BATTLE
        formation = ((event.payload.get("battle") or {}).get("formation") or {}).get("requested")
        logger.info("formation selected: formation=%s battle_id=%s", formation, self.ctx.battle_id)

    async def on_day_battle_received(self, event: AgentEvent) -> None:
        battle = event.payload.get("battle") or {}
        expected = event.payload.get("expected") or {}
        can_night = battle.get("can_night_battle")
        self.clear_action_scene_cache()
        self.ctx.battle_progress = "day"
        logger.info("day battle received: formation=%s can_night=%s", battle.get("formation"), can_night)
        if can_night:
            self.ctx.state = AgentState.WAIT_NIGHT_CHOICE
            self.ctx.battle_progress = "wait_night_choice"
            self.ctx.pending_scene = "night_battle_choice"
        else:
            self.ctx.state = AgentState.WAIT_BATTLE_RESULT
            logger.info("no night choice expected; wait for battle_result")

    async def on_night_battle_received(self, event: AgentEvent) -> None:
        self.ctx.current_fleet = event.payload.get("fleet") or self.ctx.current_fleet
        self.ctx.battle_progress = "night"
        self.ctx.state = AgentState.IN_NIGHT_BATTLE
        logger.info("night battle received")

    async def on_fleet_state_updated(self, event: AgentEvent) -> None:
        self.ctx.current_fleet = event.payload.get("fleet") or self.ctx.current_fleet
        logger.info("fleet state updated: ships=%s", self.ctx.current_fleet.get("ship_count"))

    async def on_battle_result(self, event: AgentEvent) -> None:
        payload = event.payload
        damage = payload.get("damage") or {}
        battle = payload.get("battle") or {}
        expected = payload.get("expected") or {}
        self.ctx.last_battle_result = payload
        self.ctx.last_battle_result_event_id = event.event_id
        self.ctx.battle_result_confirm_clicked = False
        self.ctx.drop_check_clicked = False
        self.ctx.current_fleet = payload.get("fleet") or self.ctx.current_fleet
        self.ctx.battle_progress = "battle_result"
        self.ctx.taiha_latch = bool(damage.get("has_taiha"))
        self.ctx.taiha_ships = damage.get("taiha_ships") or []
        self.ctx.state = AgentState.RETREAT_REQUIRED if self.ctx.taiha_latch else AgentState.WAIT_RESULT_CONFIRM
        if self.ctx.taiha_latch:
            logger.warning("taiha detected: %s", self.ctx.taiha_ships)
        logger.info(
            "battle result: rank=%s boss=%s taiha=%s",
            battle.get("rank"),
            battle.get("boss"),
            self.ctx.taiha_latch,
        )
        self.ctx.pending_scene = "battle_result_confirm"
        await self.evaluate_policy()

    async def on_scene_ready(self, event: AgentEvent) -> None:
        p = event.payload
        scene = p.get("scene")
        wait_id = p.get("wait_id")
        logger.info("scene_ready: scene=%s wait_id=%s", scene, wait_id)
        if not scene:
            logger.warning("scene_ready ignored: missing scene payload=%s", p)
            return
        self.ctx.scene_ready_cache[scene] = event
        if wait_id and self.ctx.pending_wait_id and wait_id != self.ctx.pending_wait_id:
            logger.warning("stale scene_ready ignored: %s != %s", wait_id, self.ctx.pending_wait_id)
            return
        if wait_id and wait_id == self.ctx.pending_wait_id:
            self.ctx.pending_wait_id = None
            self.ctx.pending_scene = None
        self.ctx.latest_scene = p
        if scene == "formation_select":
            self.ctx.state = AgentState.WAIT_FORMATION
            logger.info("user input required: select formation")
        elif scene == "night_battle_choice":
            self.ctx.state = AgentState.WAIT_NIGHT_CHOICE
            self.ctx.battle_progress = "wait_night_choice"
            logger.info("user input required: choose night battle or no night battle")
        await self.evaluate_policy()

    async def on_scene_analyzed(self, event: AgentEvent) -> None:
        scene = event.payload.get("scene")
        if not scene:
            logger.warning("scene_analyzed ignored: missing scene payload=%s", event.payload)
            return
        self.ctx.latest_scene = event.payload
        self.ctx.scene_ready_cache[scene] = AgentEvent("scene_ready", event.payload, event.event_id, event.ts_ms, event.source, event.correlation)
        logger.info("scene analyzed: scene=%s targets=%s", scene, list((event.payload.get("targets") or {}).keys()))
        await self.evaluate_policy()

    def action_to_target(self, action: str) -> Optional[str]:
        return self.decision_sm.action_to_target(action)

    async def evaluate_policy(self) -> None:
        scene = self.ctx.latest_scene or {}
        scene_name = scene.get("scene")
        if not scene_name:
            return
        if self.ctx.pending_command_id:
            return
        action, target = self.decision_sm.decide(self.ctx)
        if not action:
            return
        if not target:
            logger.warning("policy action has no target mapping: %s", action)
            return
        action_key = f"{self.ctx.last_battle_result_event_id or self.ctx.last_event_id}:{scene_name}:{target}"
        if action_key in self.ctx.action_history:
            return
        if not target_visible(scene, target):
            logger.info("policy matched action=%s, but target=%s is not visible in scene=%s", action, target, scene_name)
            return
        self.ctx.action_history.add(action_key)
        await self.command_click(
            target,
            f"policy:{action}",
            self.ctx.last_battle_result_event_id or self.ctx.last_event_id,
            scene.get("wait_id"),
            scene_name,
            {"policy_action": action, "scene": scene_name},
        )

    def is_fresh_scene(self, event: AgentEvent, reference_ts_ms: Optional[int] = None) -> bool:
        reference = reference_ts_ms or now_ms()
        return abs(reference - event.ts_ms) <= ACTION_SCENE_CACHE_TTL_MS

    def clear_action_scene_cache(self) -> None:
        for scene in ("battle_result_confirm", "drop_check", "advance_or_retreat"):
            self.ctx.scene_ready_cache.pop(scene, None)

    async def on_scene_timeout(self, event: AgentEvent) -> None:
        logger.error("scene timeout: %s", event.payload)
        self.ctx.state = AgentState.ERROR

    async def on_action_result(self, event: AgentEvent) -> None:
        p = event.payload
        cmd_id = p.get("command_id")
        logger.info(
            "action_result: command_id=%s status=%s reason=%s",
            cmd_id,
            p.get("status"),
            p.get("reason"),
        )
        if cmd_id != self.ctx.pending_command_id:
            logger.warning("stale action_result ignored: %s != %s", cmd_id, self.ctx.pending_command_id)
            return
        command_target = self.ctx.pending_command_target
        self.ctx.pending_command_id = None
        self.ctx.pending_command_target = None
        self.ctx.pending_command_reason = None
        if p.get("status") == "executed":
            if command_target == "result_confirm_button" and self.ctx.last_battle_result:
                drop = ((self.ctx.last_battle_result.get("battle") or {}).get("drop") or {})
                self.ctx.pending_scene = "drop_check" if has_drop(drop) else "advance_or_retreat"
                logger.info("result confirmed; waiting for scene=%s from continuous scene checker", self.ctx.pending_scene)
                return
            if command_target == "drop_confirm_button" and self.ctx.last_battle_result:
                self.ctx.pending_scene = "advance_or_retreat"
                logger.info("drop confirmed; waiting for scene=%s from continuous scene checker", self.ctx.pending_scene)
                return
            logger.info("action executed; waiting for POI confirmation")
        elif p.get("status") in ("rejected", "failed"):
            self.ctx.state = AgentState.ERROR
            logger.error("action rejected/failed")

    async def on_user_interrupt(self, event: AgentEvent) -> None:
        action = event.payload.get("action", "pause")
        if action == "pause":
            self.ctx.user_paused = True
            self.ctx.state = AgentState.PAUSED
            logger.info("paused by user")
        elif action == "resume":
            self.ctx.user_paused = False
            self.ctx.state = AgentState.IDLE
            logger.info("resumed by user")

    async def request_scene_wait(self, scene: str, required_targets: list[str], timeout_ms: int, reason: str, trigger_event_id: Optional[str], correlation: dict[str, Any]) -> None:
        wait_id = new_id("wait")
        self.ctx.pending_wait_id = wait_id
        self.ctx.pending_scene = scene
        spec = WaitSpec(wait_id, scene, required_targets, trigger_event_id, timeout_ms, 2, reason, correlation)
        logger.info("wait scene: %s, targets=%s, wait_id=%s", scene, required_targets, wait_id)
        await self.scene_wait_q.put(spec)
        self.publish_status()

    async def command_click(self, target: str, reason: str, trigger_event_id: Optional[str], wait_id: Optional[str], requires_scene: Optional[str], safety: Optional[dict[str, Any]] = None) -> None:
        cmd_id = new_id("cmd")
        self.ctx.pending_command_id = cmd_id
        self.ctx.pending_command_target = target
        self.ctx.pending_command_reason = reason
        cmd = Command(cmd_id, "click_target", target, reason, trigger_event_id, wait_id, requires_scene, safety or {}, {"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id})
        logger.info("command: click %s, reason=%s, cmd_id=%s", target, reason, cmd_id)
        await self.command_q.put(cmd)
        self.publish_status()


async def stdin_json_event_producer(event_q: asyncio.Queue[AgentEvent]) -> None:
    loop = asyncio.get_running_loop()
    logger.info("stdin_producer: paste one JSON event per line; Ctrl+C to stop")
    while True:
        line = await loop.run_in_executor(None, input)
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            event = normalize_poi_raw_event(raw)
            if event is None:
                logger.warning("stdin_producer: ignored unsupported event")
                continue
            await event_q.put(event)
        except Exception as exc:
            logger.exception("stdin_producer error: %r", exc)


async def scene_waiter_stub(scene_wait_q: asyncio.Queue[WaitSpec], event_q: asyncio.Queue[AgentEvent], auto_ready: bool = False) -> None:
    logger.info("scene_waiter_stub started")
    while True:
        spec = await scene_wait_q.get()
        try:
            logger.info(
                "scene_waiter_stub: wait_id=%s scene=%s targets=%s timeout=%sms reason=%s",
                spec.wait_id,
                spec.expected_scene,
                spec.required_targets,
                spec.timeout_ms,
                spec.reason,
            )
            if auto_ready:
                await asyncio.sleep(0.5)
                await event_q.put(AgentEvent("scene_ready", {"wait_id": spec.wait_id, "scene": spec.expected_scene, "targets": {name: {"visible": True, "bbox": [0, 0, 100, 40], "confidence": 0.99} for name in spec.required_targets}}, source="scene_waiter_stub", correlation=spec.correlation))
        finally:
            scene_wait_q.task_done()


async def mouse_executor_stub(command_q: asyncio.Queue[Command], event_q: asyncio.Queue[AgentEvent], auto_execute: bool = False) -> None:
    logger.info("mouse_executor_stub started")
    while True:
        cmd = await command_q.get()
        try:
            logger.info(
                "mouse_executor_stub: command_id=%s command=%s target=%s reason=%s safety=%s",
                cmd.command_id,
                cmd.command,
                cmd.target,
                cmd.reason,
                cmd.safety,
            )
            if auto_execute:
                await asyncio.sleep(0.2)
                await event_q.put(AgentEvent("action_result", {"command_id": cmd.command_id, "status": "executed", "target": cmd.target}, source="mouse_executor_stub", correlation=cmd.correlation))
        finally:
            command_q.task_done()


async def external_event_bridge(external_event_q: Queue[AgentEvent], event_q: asyncio.Queue[AgentEvent]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        event = await loop.run_in_executor(None, external_event_q.get)
        await event_q.put(event)


def run_gui(args: argparse.Namespace) -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGridLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    from key_identify.idc_find_feature import capture_window_by_title

    status_q: Queue[dict[str, Any]] = Queue()
    external_event_q: Queue[AgentEvent] = Queue()

    def status_sink(snapshot: dict[str, Any]) -> None:
        status_q.put(snapshot)

    def run_agent_thread() -> None:
        try:
            asyncio.run(async_main(args, status_sink=status_sink, external_event_q=external_event_q))
        except Exception as exc:
            status_q.put({
                "state": "ERROR",
                "last_event_type": "gui_agent_thread_error",
                "pending_scene": "-",
                "battle_progress": "-",
                "world": "-",
                "node_type": "-",
                "fleet": {},
                "error": repr(exc),
                "updated_at_ms": now_ms(),
            })

    class AgentDashboard(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("KC Agent Dashboard")
            self.resize(1180, 820)
            self.latest_snapshot: dict[str, Any] = {}
            self.scene_classifier = None
            self.scene_metadata = self.load_scene_metadata()
            self.scene_labels = [scene.get("name") for scene in self.scene_metadata.values() if scene.get("name")]

            root = QWidget()
            self.setCentralWidget(root)
            root.setStyleSheet("background: #030712; color: #f9fafb;")
            layout = QGridLayout(root)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(12)

            self.status_cards: dict[str, QLabel] = {}
            cards = QGridLayout()
            for idx, key in enumerate(("State", "Sea Area", "Event", "Battle", "Waiting Scene", "Command")):
                frame = QFrame()
                frame.setFrameShape(QFrame.StyledPanel)
                frame.setStyleSheet("QFrame { border: 1px solid #374151; border-radius: 6px; background: #111827; } QLabel { border: 0; }")
                box = QVBoxLayout(frame)
                title = QLabel(key)
                title.setStyleSheet("font-size: 12px; color: #d1d5db;")
                value = QLabel("-")
                value.setStyleSheet("font-size: 19px; font-weight: 700; color: #f9fafb;")
                value.setWordWrap(True)
                box.addWidget(title)
                box.addWidget(value)
                self.status_cards[key] = value
                cards.addWidget(frame, idx // 3, idx % 3)
            layout.addLayout(cards, 0, 0)

            fleet_panel = QFrame()
            fleet_panel.setFrameShape(QFrame.StyledPanel)
            fleet_panel.setStyleSheet("QFrame { border: 1px solid #374151; border-radius: 6px; background: #111827; } QLabel, QTableWidget { border: 0; color: #f9fafb; }")
            fleet_layout = QVBoxLayout(fleet_panel)
            fleet_title = QLabel("Fleet HP")
            fleet_title.setStyleSheet("font-size: 14px; font-weight: 700; color: #f9fafb;")
            self.fleet_table = QTableWidget(0, 6)
            self.fleet_table.setStyleSheet(
                "QTableWidget { background: #030712; color: #f9fafb; gridline-color: #374151; alternate-background-color: #111827; }"
                "QHeaderView::section { background: #1f2937; color: #f9fafb; border: 1px solid #374151; padding: 4px; }"
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
            self.fleet_table.setHorizontalHeaderLabels(["Pos", "Ship ID", "HP", "State", "Cond", "Fuel/Bull"])
            self.fleet_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.fleet_table.verticalHeader().setVisible(False)
            self.fleet_table.setAlternatingRowColors(True)
            fleet_layout.addWidget(fleet_title)
            fleet_layout.addWidget(self.fleet_table)
            layout.addWidget(fleet_panel, 1, 0)

            screen_panel = QFrame()
            screen_panel.setFrameShape(QFrame.StyledPanel)
            screen_panel.setStyleSheet("QFrame { border: 1px solid #374151; border-radius: 6px; background: #111827; } QLabel { border: 0; color: #f9fafb; }")
            screen_layout = QVBoxLayout(screen_panel)
            self.screen_title = QLabel(f"POI Screen: {args.poi_window_title}")
            self.screen_title.setStyleSheet("font-size: 14px; font-weight: 700;")
            self.screen_label = QLabel("Waiting for screenshot...")
            self.screen_label.setAlignment(Qt.AlignCenter)
            self.screen_label.setMinimumSize(520, 360)
            self.screen_label.setStyleSheet("background: #030712; color: #d1d5db;")
            self.screen_status = QLabel("")
            self.screen_status.setStyleSheet("font-size: 12px; color: #d1d5db;")
            screen_layout.addWidget(self.screen_title)
            screen_layout.addWidget(self.screen_label, 1)
            screen_layout.addWidget(self.screen_status)
            layout.addWidget(screen_panel, 0, 1, 2, 1)

            self.detail = QLabel("")
            self.detail.setStyleSheet("color: #f9fafb; background: #030712;")
            layout.addWidget(self.detail, 2, 0, 1, 2)
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 1)
            layout.setRowStretch(1, 1)

            self.status_timer = QTimer(self)
            self.status_timer.timeout.connect(self.drain_status)
            self.status_timer.start(args.gui_refresh_ms)

            self.screenshot_timer = QTimer(self)
            self.screenshot_timer.timeout.connect(self.refresh_screenshot)
            self.screenshot_timer.start(args.screenshot_refresh_ms)

        def drain_status(self) -> None:
            changed = False
            while True:
                try:
                    self.latest_snapshot = status_q.get_nowait()
                    changed = True
                except Empty:
                    break
            if changed:
                self.render_status(self.latest_snapshot)

        def render_status(self, snapshot: dict[str, Any]) -> None:
            route_no = snapshot.get("route_no")
            sea = snapshot.get("world") or "-"
            if route_no is not None:
                sea = f"{sea} / node {route_no}"
            node_type = snapshot.get("node_type") or "-"
            if node_type != "-":
                sea = f"{sea} ({node_type})"

            self.status_cards["State"].setText(str(snapshot.get("state") or "-"))
            self.status_cards["Sea Area"].setText(sea)
            self.status_cards["Event"].setText(str(snapshot.get("last_event_type") or "-"))
            self.status_cards["Battle"].setText(str(snapshot.get("battle_progress") or "-"))
            self.status_cards["Waiting Scene"].setText(str(snapshot.get("pending_scene") or "-"))
            self.status_cards["Command"].setText(str(snapshot.get("pending_command_target") or "-"))

            fleet = snapshot.get("fleet") or {}
            ships = list(fleet.get("ships") or [])
            self.fleet_table.setRowCount(len(ships))
            for row, ship in enumerate(ships):
                now_hp = to_int(ship.get("battle_end_hp"), to_int(ship.get("now_hp"), 0)) or 0
                max_hp = to_int(ship.get("max_hp"), 0) or 0
                values = [
                    ship.get("pos") or ship.get("position") or row + 1,
                    ship.get("ship_id", "-"),
                    f"{now_hp}/{max_hp}",
                    damage_state(now_hp, max_hp),
                    ship.get("cond", "-"),
                    f"{ship.get('fuel', '-')}/{ship.get('bullet', '-')}",
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.fleet_table.setItem(row, col, item)

            taiha = snapshot.get("taiha_ships") or []
            error = snapshot.get("error")
            self.detail.setText(
                f"wait_id={snapshot.get('pending_wait_id') or '-'}    "
                f"event_id={snapshot.get('last_event_id') or '-'}    "
                f"taiha={snapshot.get('taiha_latch')} {taiha}"
                + (f"    error={error}" if error else "")
            )

        def refresh_screenshot(self) -> None:
            try:
                bgr, window = capture_window_by_title(args.poi_window_title)
                rgb = bgr[:, :, ::-1].copy()
                height, width, channels = rgb.shape
                qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
                pixmap = QPixmap.fromImage(qimage).scaled(
                    self.screen_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.screen_label.setPixmap(pixmap)
                self.screen_status.setText(f"{window.title}  {width}x{height}")
                self.analyze_scene(bgr)
            except Exception as exc:
                self.screen_label.setText("POI window not found or screenshot failed")
                self.screen_status.setText(str(exc))

        def load_scene_metadata(self) -> dict[str, Any]:
            path = Path(args.scene_metadata)
            if not path.exists():
                return {}
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)

        def get_scene_classifier(self):
            if self.scene_classifier is not None:
                return self.scene_classifier
            from scene_identify.si_progress_classify import LocalCLIPClassifier
            self.scene_classifier = LocalCLIPClassifier(
                model_ref=args.scene_model_ref,
                cache_dir=args.scene_model_cache,
                device=args.scene_device,
            )
            return self.scene_classifier

        def canonical_scene_name(self, label: str) -> str:
            mapping = {
                "night_battle": "night_battle_choice",
                "battle_result": "battle_result_confirm",
                "fleet_condition": "drop_check",
                "next_node": "advance_or_retreat",
            }
            return mapping.get(label, label)

        def analyze_scene(self, bgr: Any) -> None:
            if not self.scene_labels:
                return
            try:
                from PIL import Image
                classifier = self.get_scene_classifier()
                tmp_path = Path(args.scene_screenshot_path)
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                rgb = bgr[:, :, ::-1]
                Image.fromarray(rgb).save(tmp_path)
                result = classifier.classify(
                    str(tmp_path),
                    self.scene_labels,
                    text_template="a game screenshot of {}",
                    top_k=3,
                )
                raw_scene = result["best_label"]
                scene_name = self.canonical_scene_name(raw_scene)
                targets = self.find_scene_targets(raw_scene, bgr)
                external_event_q.put(AgentEvent(
                    "scene_analyzed",
                    {
                        "scene": scene_name,
                        "raw_scene": raw_scene,
                        "scores": result.get("scores") or [],
                        "targets": targets,
                    },
                    source="gui_scene_checker",
                ))
            except Exception as exc:
                self.screen_status.setText(f"scene analyze failed: {exc}")

        def find_scene_targets(self, raw_scene: str, bgr: Any) -> dict[str, Any]:
            from key_identify.idc_find_feature import FeatureMatcher
            scene_meta = next((scene for scene in self.scene_metadata.values() if scene.get("name") == raw_scene), {})
            buttons = scene_meta.get("buttons") or {}
            targets: dict[str, Any] = {}
            target_names = {
                "enter_night_battle": "night_battle_button",
                "skip_night_battle": "no_night_battle_button",
                "next": "result_confirm_button" if raw_scene == "battle_result" else "drop_confirm_button",
                "advance": "advance_button",
                "retreat": "retreat_button",
            }
            for button_name, button_meta in buttons.items():
                key_file = button_meta.get("key_file")
                if not key_file:
                    continue
                key_path = Path(key_file)
                if not key_path.is_absolute():
                    key_path = Path.cwd() / key_path
                if not key_path.exists():
                    continue
                target_name = target_names.get(button_name, button_name)
                try:
                    matcher = FeatureMatcher(args.poi_window_title, key_path)
                    match = matcher.match_template_tm(scene_bgr=bgr, threshold=args.key_match_threshold)
                    if match.found:
                        targets[target_name] = {
                            "visible": True,
                            "confidence": match.score,
                            "bbox": list(match.bbox_screen_xywh),
                            "button": button_name,
                            "key_file": str(key_path),
                        }
                except Exception as exc:
                    targets[target_name] = {"visible": False, "error": repr(exc), "button": button_name}
            return targets

    thread = threading.Thread(target=run_agent_thread, name="kc_agent_gui_loop", daemon=True)
    thread.start()

    app = QApplication([])
    dashboard = AgentDashboard()
    dashboard.show()
    return app.exec()


def start_battle_receiver(
    args: argparse.Namespace,
    loop: asyncio.AbstractEventLoop,
    event_q: asyncio.Queue[AgentEvent],
) -> tuple[HTTPServer, threading.Thread]:
    def packet_sink(packet: dict[str, Any]) -> None:
        def enqueue_packet() -> None:
            try:
                event_q.put_nowait(agent_event_from_packet(packet))
            except asyncio.QueueFull:
                logger.error("event queue full; dropped receiver packet type=%s", packet.get("type"))
            except Exception as exc:
                logger.exception("failed to enqueue receiver packet: %r", exc)

        loop.call_soon_threadsafe(enqueue_packet)

    config = ReceiverConfig(
        raw_log_path=args.receiver_raw_log,
        normalized_log_path=args.receiver_normalized_log,
        print_raw=args.receiver_print_raw,
        print_normalized=not args.receiver_quiet_normalized,
        packet_sink=packet_sink,
    )
    server = HTTPServer(
        (args.receiver_host, args.receiver_port),
        make_battle_receiver_handler(config),
    )
    thread = threading.Thread(
        target=server.serve_forever,
        name="battle_receiver_http",
        daemon=True,
    )
    thread.start()
    logger.info(
        "battle_receiver started: http://%s:%s/poi-event",
        args.receiver_host,
        server.server_port,
    )
    return server, thread


async def async_main(
    args: argparse.Namespace,
    status_sink: Optional[Callable[[dict[str, Any]], None]] = None,
    external_event_q: Optional[Queue[AgentEvent]] = None,
) -> None:
    event_q: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=args.event_queue_size)
    scene_wait_q: asyncio.Queue[WaitSpec] = asyncio.Queue(maxsize=args.scene_queue_size)
    command_q: asyncio.Queue[Command] = asyncio.Queue(maxsize=args.command_queue_size)
    agent = KcAgent(
        event_q,
        scene_wait_q,
        command_q,
        status_sink=status_sink,
        action_policy=load_action_policy(args.action_policy),
    )
    loop = asyncio.get_running_loop()
    receiver_server: Optional[HTTPServer] = None
    receiver_thread: Optional[threading.Thread] = None
    if not args.no_battle_receiver:
        # if we did not said we don't want battle receiver, start it
        receiver_server, receiver_thread = start_battle_receiver(args, loop, event_q)
    tasks = [
        asyncio.create_task(agent.run(), name="kc_agent_main_loop"),
        asyncio.create_task(stdin_json_event_producer(event_q), name="stdin_json_event_producer"),
        asyncio.create_task(scene_waiter_stub(scene_wait_q, event_q, auto_ready=args.auto_scene_ready), name="scene_waiter_stub"),
        asyncio.create_task(mouse_executor_stub(command_q, event_q, auto_execute=args.auto_mouse_execute), name="mouse_executor_stub"),
    ]
    if external_event_q is not None:
        tasks.append(asyncio.create_task(external_event_bridge(external_event_q, event_q), name="external_event_bridge"))
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        if receiver_server is not None:
            receiver_server.shutdown()
            receiver_server.server_close()
        if receiver_thread is not None:
            receiver_thread.join(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KC_AGENT main orchestrator")
    parser.add_argument("--gui", action="store_true", help="Run KC Agent with a PySide6 dashboard.")
    parser.add_argument("--poi-window-title", default="poi", help="Window title substring for POI screenshot capture.")
    parser.add_argument("--gui-refresh-ms", type=int, default=250, help="Dashboard state refresh interval.")
    parser.add_argument("--screenshot-refresh-ms", type=int, default=1000, help="POI screenshot refresh interval.")
    parser.add_argument("--action-policy", type=Path, default=DEFAULT_ACTION_POLICY_PATH, help="JSON policy file for main action decisions.")
    parser.add_argument("--scene-metadata", type=Path, default=Path("utility/scene_metadata.json"))
    parser.add_argument("--scene-model-ref", default="./scene_identify/models/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    parser.add_argument("--scene-model-cache", default="./scene_identify/models")
    parser.add_argument("--scene-device", default="cpu")
    parser.add_argument("--scene-screenshot-path", type=Path, default=Path("C:/tmp/kc_agent_scene.png"))
    parser.add_argument("--key-match-threshold", type=float, default=0.8)
    parser.add_argument("--event-queue-size", type=int, default=100)
    parser.add_argument("--scene-queue-size", type=int, default=20)
    parser.add_argument("--command-queue-size", type=int, default=20)
    parser.add_argument("--auto-scene-ready", action="store_true", help="Stub mode: scene waiter emits scene_ready automatically.")
    parser.add_argument("--auto-mouse-execute", action="store_true", help="Stub mode: mouse executor emits action_result automatically.")
    parser.add_argument("--no-battle-receiver", action="store_true", help="Do not start the embedded POI battle receiver.")
    parser.add_argument("--receiver-host", default=DEFAULT_RECEIVER_HOST)
    parser.add_argument("--receiver-port", type=int, default=DEFAULT_RECEIVER_PORT)
    parser.add_argument("--receiver-raw-log", type=Path, default=DEFAULT_RAW_LOG_PATH)
    parser.add_argument("--receiver-normalized-log", type=Path, default=DEFAULT_NORMALIZED_LOG_PATH)
    parser.add_argument("--receiver-print-raw", action="store_true")
    parser.add_argument("--receiver-quiet-normalized", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gui:
        return run_gui(args)
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
