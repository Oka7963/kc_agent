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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


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
    current_deck_id: Optional[int] = None
    taiha_latch: bool = False
    taiha_ships: list[dict[str, Any]] = field(default_factory=list)
    pending_wait_id: Optional[str] = None
    pending_scene: Optional[str] = None
    pending_command_id: Optional[str] = None
    last_event_id: Optional[str] = None
    last_battle_result: Optional[dict[str, Any]] = None
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
            next_scenes.append({"scene": "compass_or_map_production", "required_targets": [], "timeout_ms": 8000})
        if is_battle:
            next_scenes.append({"scene": "formation_select", "required_targets": ["formation_buttons"], "timeout_ms": 10000})
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
            next_scenes.append({"scene": "night_battle_choice", "required_targets": ["night_battle_button", "no_night_battle_button"], "timeout_ms": 10000})
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
        required_targets = ["retreat_button"] if damage["has_taiha"] else ["advance_button", "retreat_button"]
        payload = {
            "schema": "kc.agent.battle_event.v1",
            "event_type": "battle_result",
            "phase": "result",
            "seq": seq,
            "source_event": raw.get("source"),
            "path": path,
            "battle": {"rank": result.get("rank"), "boss": result.get("boss"), "map": result.get("map"), "map_cell": result.get("map_cell") or result.get("mapCell"), "enemy_name": result.get("enemy"), "mvp": result.get("mvp") or [], "drop": {"ship_id": result.get("drop_ship_id") or result.get("dropShipId"), "item": result.get("drop_item") or result.get("dropItem"), "event_item": result.get("event_item") or result.get("eventItem")}},
            "fleet": fleet,
            "damage": damage,
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": True, "ui_wait_reason": "battle_result_confirm_then_advance_or_retreat", "next_scenes": [{"scene": "battle_result_confirm", "required_targets": ["result_confirm_button"], "timeout_ms": 8000}, {"scene": "advance_or_retreat", "required_targets": required_targets, "timeout_ms": 10000}]},
        }
        return AgentEvent("battle_result", payload, new_id("poi"), ts_ms, "poi")

    return None


class KcAgent:
    def __init__(self, event_q: asyncio.Queue[AgentEvent], scene_wait_q: asyncio.Queue[WaitSpec], command_q: asyncio.Queue[Command]) -> None:
        self.event_q = event_q
        self.scene_wait_q = scene_wait_q
        self.command_q = command_q
        self.ctx = RuntimeContext()

    async def run(self) -> None:
        print("[kc_agent] main loop started")
        while True:
            event = await self.event_q.get()
            try:
                await self.handle_event(event)
            except Exception as exc:
                self.ctx.state = AgentState.ERROR
                print(f"[kc_agent][ERROR] exception while handling {event.type}: {exc!r}")
            finally:
                self.event_q.task_done()

    async def handle_event(self, event: AgentEvent) -> None:
        self.ctx.last_event_id = event.event_id
        if self.ctx.user_paused and event.type != "user_interrupt":
            print(f"[kc_agent] paused; ignoring event={event.type}")
            return
        print(f"[kc_agent] state={self.ctx.state} event={event.type}")
        handlers = {
            "sortie_start_requested": self.on_sortie_start_requested,
            "map_node_arrived": self.on_map_node_arrived,
            "formation_selected": self.on_formation_selected,
            "day_battle_received": self.on_day_battle_received,
            "battle_result": self.on_battle_result,
            "scene_ready": self.on_scene_ready,
            "scene_timeout": self.on_scene_timeout,
            "action_result": self.on_action_result,
            "user_interrupt": self.on_user_interrupt,
        }
        handler = handlers.get(event.type)
        if handler is None:
            print(f"[kc_agent] unhandled event type={event.type}")
            return
        await handler(event)

    async def on_sortie_start_requested(self, event: AgentEvent) -> None:
        map_info = event.payload.get("map") or {}
        fleet = event.payload.get("fleet") or {}
        self.ctx.sortie_id = new_id("sortie")
        self.ctx.battle_id = None
        self.ctx.map_node_id = None
        self.ctx.current_world = map_info.get("world")
        self.ctx.current_deck_id = to_int(fleet.get("deck_id"), 1)
        self.ctx.taiha_latch = False
        self.ctx.taiha_ships.clear()
        self.ctx.state = AgentState.SORTIE_STARTING
        print(f"[kc_agent] sortie requested: sortie_id={self.ctx.sortie_id} world={self.ctx.current_world} deck={self.ctx.current_deck_id}")

    async def on_map_node_arrived(self, event: AgentEvent) -> None:
        map_info = event.payload.get("map") or {}
        expected = event.payload.get("expected") or {}
        next_scenes = expected.get("next_scenes") or []
        self.ctx.current_world = map_info.get("world") or self.ctx.current_world
        self.ctx.current_route_no = to_int(map_info.get("route_no"))
        self.ctx.current_node_type = map_info.get("node_type", "unknown")
        self.ctx.map_node_id = f"{self.ctx.current_world}-route-{self.ctx.current_route_no}"
        self.ctx.state = AgentState.MAP_NODE_ARRIVED
        print(f"[kc_agent] node arrived: {self.ctx.map_node_id} type={self.ctx.current_node_type} rashin={map_info.get('rashin_flg')}/{map_info.get('rashin_id')}")
        if next_scenes:
            first = next_scenes[0]
            await self.request_scene_wait(first["scene"], first.get("required_targets", []), first.get("timeout_ms", 8000), expected.get("ui_wait_reason", "map_node_arrived"), event.event_id, {"sortie_id": self.ctx.sortie_id, "map_node_id": self.ctx.map_node_id, "next_scenes": next_scenes, "next_scene_index": 0})
            self.ctx.state = AgentState.WAIT_FORMATION if first["scene"] == "formation_select" else AgentState.WAIT_COMPASS_OR_PRODUCTION
        else:
            print("[kc_agent] no expected scene; waiting for further POI event")

    async def on_formation_selected(self, event: AgentEvent) -> None:
        self.ctx.battle_id = new_id("battle")
        self.ctx.state = AgentState.IN_DAY_BATTLE
        formation = ((event.payload.get("battle") or {}).get("formation") or {}).get("requested")
        print(f"[kc_agent] formation selected: formation={formation}, battle_id={self.ctx.battle_id}")

    async def on_day_battle_received(self, event: AgentEvent) -> None:
        battle = event.payload.get("battle") or {}
        expected = event.payload.get("expected") or {}
        can_night = battle.get("can_night_battle")
        print(f"[kc_agent] day battle received: formation={battle.get('formation')} can_night={can_night}")
        if can_night:
            self.ctx.state = AgentState.WAIT_NIGHT_CHOICE
            scene = (expected.get("next_scenes") or [{}])[0]
            await self.request_scene_wait(scene.get("scene", "night_battle_choice"), scene.get("required_targets", ["night_battle_button", "no_night_battle_button"]), scene.get("timeout_ms", 10000), "can_night_battle", event.event_id, {"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id})
        else:
            self.ctx.state = AgentState.WAIT_BATTLE_RESULT
            print("[kc_agent] no night choice expected; wait for battle_result")

    async def on_battle_result(self, event: AgentEvent) -> None:
        payload = event.payload
        damage = payload.get("damage") or {}
        battle = payload.get("battle") or {}
        expected = payload.get("expected") or {}
        self.ctx.last_battle_result = payload
        self.ctx.taiha_latch = bool(damage.get("has_taiha"))
        self.ctx.taiha_ships = damage.get("taiha_ships") or []
        self.ctx.state = AgentState.RETREAT_REQUIRED if self.ctx.taiha_latch else AgentState.WAIT_RESULT_CONFIRM
        if self.ctx.taiha_latch:
            print(f"[kc_agent][SAFETY] taiha detected: {self.ctx.taiha_ships}")
        print(f"[kc_agent] battle result: rank={battle.get('rank')} boss={battle.get('boss')} taiha={self.ctx.taiha_latch}")
        scenes = expected.get("next_scenes") or []
        if scenes:
            first = scenes[0]
            await self.request_scene_wait(first.get("scene", "battle_result_confirm"), first.get("required_targets", ["result_confirm_button"]), first.get("timeout_ms", 8000), "battle_result_received", event.event_id, {"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id, "next_scenes": scenes, "next_scene_index": 0})

    async def on_scene_ready(self, event: AgentEvent) -> None:
        p = event.payload
        scene = p.get("scene")
        wait_id = p.get("wait_id")
        corr = event.correlation or p.get("correlation") or {}
        print(f"[kc_agent] scene_ready: scene={scene} wait_id={wait_id}")
        if wait_id != self.ctx.pending_wait_id:
            print(f"[kc_agent] stale scene_ready ignored: {wait_id} != {self.ctx.pending_wait_id}")
            return
        self.ctx.pending_wait_id = None
        self.ctx.pending_scene = None
        scenes = corr.get("next_scenes") or []
        idx = to_int(corr.get("next_scene_index"), 0) or 0
        if idx + 1 < len(scenes):
            nxt = scenes[idx + 1]
            await self.request_scene_wait(nxt["scene"], nxt.get("required_targets", []), nxt.get("timeout_ms", 8000), f"next_scene_after_{scene}", event.event_id, {**corr, "next_scene_index": idx + 1})
            self.ctx.state = AgentState.WAIT_FORMATION if nxt["scene"] == "formation_select" else AgentState.WAIT_ADVANCE_OR_RETREAT if nxt["scene"] == "advance_or_retreat" else self.ctx.state
            return
        if scene == "battle_result_confirm":
            await self.command_click("result_confirm_button", "confirm_battle_result", event.event_id, wait_id, scene)
            self.ctx.state = AgentState.WAIT_ACTION_RESULT
        elif scene == "advance_or_retreat":
            if self.ctx.taiha_latch:
                await self.command_click("retreat_button", "taiha_detected", event.event_id, wait_id, scene, {"taiha_latch": True, "forbid_target": "advance_button"})
                self.ctx.state = AgentState.WAIT_ACTION_RESULT
            else:
                self.ctx.state = AgentState.WAIT_ADVANCE_OR_RETREAT
                print("[kc_agent] user input required: choose advance or retreat")
        elif scene == "formation_select":
            self.ctx.state = AgentState.WAIT_FORMATION
            print("[kc_agent] user input required: select formation")
        elif scene == "night_battle_choice":
            self.ctx.state = AgentState.WAIT_NIGHT_CHOICE
            print("[kc_agent] user input required: choose night battle or no night battle")
        else:
            print(f"[kc_agent] scene ready but no action defined: {scene}")

    async def on_scene_timeout(self, event: AgentEvent) -> None:
        print(f"[kc_agent][TIMEOUT] scene timeout: {event.payload}")
        self.ctx.state = AgentState.ERROR

    async def on_action_result(self, event: AgentEvent) -> None:
        p = event.payload
        cmd_id = p.get("command_id")
        print(f"[kc_agent] action_result: command_id={cmd_id} status={p.get('status')} reason={p.get('reason')}")
        if cmd_id != self.ctx.pending_command_id:
            print(f"[kc_agent] stale action_result ignored: {cmd_id} != {self.ctx.pending_command_id}")
            return
        self.ctx.pending_command_id = None
        if p.get("status") == "executed":
            if self.ctx.last_battle_result:
                scenes = (self.ctx.last_battle_result.get("expected") or {}).get("next_scenes") or []
                advance_scene = next((s for s in scenes if s.get("scene") == "advance_or_retreat"), None)
                if advance_scene:
                    await self.request_scene_wait("advance_or_retreat", advance_scene.get("required_targets", ["advance_button", "retreat_button"]), advance_scene.get("timeout_ms", 10000), "after_result_confirm", event.event_id, {"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id})
                    self.ctx.state = AgentState.WAIT_ADVANCE_OR_RETREAT
                    return
            print("[kc_agent] action executed; waiting for POI confirmation")
        elif p.get("status") in ("rejected", "failed"):
            self.ctx.state = AgentState.ERROR
            print("[kc_agent][ERROR] action rejected/failed")

    async def on_user_interrupt(self, event: AgentEvent) -> None:
        action = event.payload.get("action", "pause")
        if action == "pause":
            self.ctx.user_paused = True
            self.ctx.state = AgentState.PAUSED
            print("[kc_agent] paused by user")
        elif action == "resume":
            self.ctx.user_paused = False
            self.ctx.state = AgentState.IDLE
            print("[kc_agent] resumed by user")

    async def request_scene_wait(self, scene: str, required_targets: list[str], timeout_ms: int, reason: str, trigger_event_id: Optional[str], correlation: dict[str, Any]) -> None:
        wait_id = new_id("wait")
        self.ctx.pending_wait_id = wait_id
        self.ctx.pending_scene = scene
        spec = WaitSpec(wait_id, scene, required_targets, trigger_event_id, timeout_ms, 2, reason, correlation)
        print(f"[kc_agent] wait scene: {scene}, targets={required_targets}, wait_id={wait_id}")
        await self.scene_wait_q.put(spec)

    async def command_click(self, target: str, reason: str, trigger_event_id: Optional[str], wait_id: Optional[str], requires_scene: Optional[str], safety: Optional[dict[str, Any]] = None) -> None:
        cmd_id = new_id("cmd")
        self.ctx.pending_command_id = cmd_id
        cmd = Command(cmd_id, "click_target", target, reason, trigger_event_id, wait_id, requires_scene, safety or {}, {"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id})
        print(f"[kc_agent] command: click {target}, reason={reason}, cmd_id={cmd_id}")
        await self.command_q.put(cmd)


async def stdin_json_event_producer(event_q: asyncio.Queue[AgentEvent]) -> None:
    loop = asyncio.get_running_loop()
    print("[stdin_producer] paste one JSON event per line; Ctrl+C to stop")
    while True:
        line = await loop.run_in_executor(None, input)
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
            event = normalize_poi_raw_event(raw)
            if event is None:
                print("[stdin_producer] ignored unsupported event")
                continue
            await event_q.put(event)
        except Exception as exc:
            print(f"[stdin_producer][ERROR] {exc!r}")


async def scene_waiter_stub(scene_wait_q: asyncio.Queue[WaitSpec], event_q: asyncio.Queue[AgentEvent], auto_ready: bool = False) -> None:
    print("[scene_waiter_stub] started")
    while True:
        spec = await scene_wait_q.get()
        try:
            print(f"[scene_waiter_stub] wait_id={spec.wait_id} scene={spec.expected_scene} targets={spec.required_targets} timeout={spec.timeout_ms}ms reason={spec.reason}")
            if auto_ready:
                await asyncio.sleep(0.5)
                await event_q.put(AgentEvent("scene_ready", {"wait_id": spec.wait_id, "scene": spec.expected_scene, "targets": {name: {"visible": True, "bbox": [0, 0, 100, 40], "confidence": 0.99} for name in spec.required_targets}}, source="scene_waiter_stub", correlation=spec.correlation))
        finally:
            scene_wait_q.task_done()


async def mouse_executor_stub(command_q: asyncio.Queue[Command], event_q: asyncio.Queue[AgentEvent], auto_execute: bool = False) -> None:
    print("[mouse_executor_stub] started")
    while True:
        cmd = await command_q.get()
        try:
            print(f"[mouse_executor_stub] command_id={cmd.command_id} command={cmd.command} target={cmd.target} reason={cmd.reason} safety={cmd.safety}")
            if auto_execute:
                await asyncio.sleep(0.2)
                await event_q.put(AgentEvent("action_result", {"command_id": cmd.command_id, "status": "executed", "target": cmd.target}, source="mouse_executor_stub", correlation=cmd.correlation))
        finally:
            command_q.task_done()


async def async_main(args: argparse.Namespace) -> None:
    event_q: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=args.event_queue_size)
    scene_wait_q: asyncio.Queue[WaitSpec] = asyncio.Queue(maxsize=args.scene_queue_size)
    command_q: asyncio.Queue[Command] = asyncio.Queue(maxsize=args.command_queue_size)
    agent = KcAgent(event_q, scene_wait_q, command_q)
    tasks = [
        asyncio.create_task(agent.run(), name="kc_agent_main_loop"),
        asyncio.create_task(stdin_json_event_producer(event_q), name="stdin_json_event_producer"),
        asyncio.create_task(scene_waiter_stub(scene_wait_q, event_q, auto_ready=args.auto_scene_ready), name="scene_waiter_stub"),
        asyncio.create_task(mouse_executor_stub(command_q, event_q, auto_execute=args.auto_mouse_execute), name="mouse_executor_stub"),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KC_AGENT main orchestrator")
    parser.add_argument("--event-queue-size", type=int, default=100)
    parser.add_argument("--scene-queue-size", type=int, default=20)
    parser.add_argument("--command-queue-size", type=int, default=20)
    parser.add_argument("--auto-scene-ready", action="store_true", help="Stub mode: scene waiter emits scene_ready automatically.")
    parser.add_argument("--auto-mouse-execute", action="store_true", help="Stub mode: mouse executor emits action_result automatically.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n[kc_agent] stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
