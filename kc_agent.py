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

from kc_core.decoder import normalize_poi_raw_event
from kc_core.event_models import AgentEvent, Command, JsonDict, SceneObservation, WaitSpec, new_id, to_int


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
    latest_scene: Optional[SceneObservation] = None
    fired_action_keys: set[str] = field(default_factory=set)
    last_event_id: Optional[str] = None
    last_battle_result_event_id: Optional[str] = None
    last_battle_result: Optional[dict[str, Any]] = None
    last_battle_result_event_id: Optional[str] = None
    battle_result_confirm_clicked: bool = False
    drop_check_clicked: bool = False
    scene_ready_cache: dict[str, AgentEvent] = field(default_factory=dict)
    latest_scene: Optional[dict[str, Any]] = None
    action_history: set[str] = field(default_factory=set)
    user_paused: bool = False


@dataclass(frozen=True)
class ActionRule:
    """Condition that allows scene observation to become a click command."""

    scene: str
    target: str
    reason: str
    requires_event_condition: bool = True
    requires_taiha: bool = False
    forbid_target: Optional[str] = None


ACTION_RULES = [
    # Scenario 2: the current screen itself is enough. If result confirm is
    # visible/clickable, clicking it is safe even if the POI event arrived late
    # or was missed.
    ActionRule(
        scene="battle_result_confirm",
        target="result_confirm_button",
        reason="confirm_battle_result",
        requires_event_condition=False,
    ),
    # Scenario 1: retreat is allowed only when both the event-derived safety
    # condition and the screen observation match.
    ActionRule(
        scene="advance_or_retreat",
        target="retreat_button",
        reason="taiha_detected",
        requires_event_condition=True,
        requires_taiha=True,
        forbid_target="advance_button",
    ),
]


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
        self.ctx.last_battle_result_event_id = None
        self.ctx.last_battle_result = None
        self.ctx.fired_action_keys.clear()
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
        await self.try_fire_action_from_latest_scene(event.event_id)

    async def on_scene_ready(self, event: AgentEvent) -> None:
        p = event.payload
        scene = p.get("scene")
        wait_id = p.get("wait_id")
        corr = event.correlation or p.get("correlation") or {}
        print(f"[kc_agent] scene_ready: scene={scene} wait_id={wait_id}")
        if not scene:
            print("[kc_agent] scene_ready missing scene; ignored")
            return

        observation = self.remember_scene_observation(event, corr)
        if wait_id is not None and wait_id != self.ctx.pending_wait_id:
            print(f"[kc_agent] stale scene_ready ignored: {wait_id} != {self.ctx.pending_wait_id}")
            return

        if wait_id is not None:
            self.ctx.pending_wait_id = None
            self.ctx.pending_scene = None

        if await self.try_fire_action_from_scene(observation, event.event_id):
            return

        if wait_id is None:
            print(f"[kc_agent] latest scene recorded; no action matched: {scene}")
            return

        scenes = corr.get("next_scenes") or []
        idx = to_int(corr.get("next_scene_index"), 0) or 0
        if idx + 1 < len(scenes):
            nxt = scenes[idx + 1]
            await self.request_scene_wait(nxt["scene"], nxt.get("required_targets", []), nxt.get("timeout_ms", 8000), f"next_scene_after_{scene}", event.event_id, {**corr, "next_scene_index": idx + 1})
            self.ctx.state = AgentState.WAIT_FORMATION if nxt["scene"] == "formation_select" else AgentState.WAIT_ADVANCE_OR_RETREAT if nxt["scene"] == "advance_or_retreat" else self.ctx.state
            return
        if scene == "advance_or_retreat":
            self.ctx.state = AgentState.WAIT_ADVANCE_OR_RETREAT
            print("[kc_agent] user input required: choose advance or retreat")
        elif scene == "formation_select":
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
            if self.ctx.last_battle_result:
                scenes = (self.ctx.last_battle_result.get("expected") or {}).get("next_scenes") or []
                advance_scene = next((s for s in scenes if s.get("scene") == "advance_or_retreat"), None)
                if advance_scene:
                    await self.request_scene_wait("advance_or_retreat", advance_scene.get("required_targets", ["advance_button", "retreat_button"]), advance_scene.get("timeout_ms", 10000), "after_result_confirm", event.event_id, {"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id})
                    self.ctx.state = AgentState.WAIT_ADVANCE_OR_RETREAT
                    await self.try_fire_action_from_latest_scene(event.event_id)
                    return
            print("[kc_agent] action executed; waiting for POI confirmation")
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

    def remember_scene_observation(self, event: AgentEvent, correlation: JsonDict) -> SceneObservation:
        payload = event.payload
        previous_scene = self.ctx.latest_scene.scene if self.ctx.latest_scene else None
        observation = SceneObservation(
            scene=payload.get("scene"),
            targets=payload.get("targets") or {},
            wait_id=payload.get("wait_id"),
            source_event_id=event.event_id,
            correlation=correlation,
        )
        if previous_scene and previous_scene != observation.scene:
            self.ctx.fired_action_keys = {
                key for key in self.ctx.fired_action_keys if not key.endswith(":unbound")
            }
        self.ctx.latest_scene = observation
        return observation

    async def try_fire_action_from_latest_scene(self, trigger_event_id: Optional[str]) -> bool:
        if self.ctx.latest_scene is None:
            return False
        return await self.try_fire_action_from_scene(self.ctx.latest_scene, trigger_event_id)

    async def try_fire_action_from_scene(self, observation: SceneObservation, trigger_event_id: Optional[str]) -> bool:
        if self.ctx.pending_command_id:
            print(f"[kc_agent] command pending; action gate deferred for scene={observation.scene}")
            return False

        for rule in ACTION_RULES:
            if rule.scene != observation.scene:
                continue
            target_observation = observation.target(rule.target)
            if not self.target_is_clickable(target_observation):
                continue
            if not self.rule_event_condition_met(rule):
                print(f"[kc_agent] action gate blocked: scene={rule.scene} target={rule.target} reason={rule.reason}")
                continue

            action_key = self.action_key(rule, observation)
            if action_key in self.ctx.fired_action_keys:
                print(f"[kc_agent] duplicate action suppressed: {action_key}")
                return False

            self.ctx.fired_action_keys.add(action_key)
            safety = {"trigger_mode": "scene_only" if not rule.requires_event_condition else "event_and_scene"}
            if rule.requires_taiha:
                safety["taiha_latch"] = True
            if rule.forbid_target:
                safety["forbid_target"] = rule.forbid_target
            await self.command_click(rule.target, rule.reason, trigger_event_id, observation.wait_id, rule.scene, safety, target_observation)
            self.ctx.state = AgentState.WAIT_ACTION_RESULT
            return True
        return False

    def rule_event_condition_met(self, rule: ActionRule) -> bool:
        if not rule.requires_event_condition:
            return True
        if rule.requires_taiha and not self.ctx.taiha_latch:
            return False
        return True

    @staticmethod
    def target_is_clickable(target: Optional[JsonDict]) -> bool:
        if not target:
            return False
        if target.get("visible") is False or target.get("clickable") is False:
            return False
        bbox = target.get("bbox_screen_xywh") or target.get("bbox")
        return bool(bbox)

    def action_key(self, rule: ActionRule, observation: SceneObservation) -> str:
        if rule.requires_event_condition:
            source_id = observation.wait_id or self.ctx.last_battle_result_event_id or "event"
        else:
            source_id = observation.wait_id or "unbound"
        return f"{rule.scene}:{rule.target}:{source_id}"

    async def command_click(self, target: str, reason: str, trigger_event_id: Optional[str], wait_id: Optional[str], requires_scene: Optional[str], safety: Optional[dict[str, Any]] = None, target_observation: Optional[JsonDict] = None) -> None:
        cmd_id = new_id("cmd")
        self.ctx.pending_command_id = cmd_id
        cmd = Command(
            command_id=cmd_id,
            command="click_target",
            target=target,
            reason=reason,
            trigger_event_id=trigger_event_id,
            wait_id=wait_id,
            requires_scene=requires_scene,
            safety=safety or {},
            correlation={"sortie_id": self.ctx.sortie_id, "battle_id": self.ctx.battle_id},
            target_observation=target_observation or {},
        )
        print(f"[kc_agent] command: click {target}, reason={reason}, cmd_id={cmd_id}, bbox={(target_observation or {}).get('bbox_screen_xywh') or (target_observation or {}).get('bbox')}")
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
            bbox = cmd.target_observation.get("bbox_screen_xywh") or cmd.target_observation.get("bbox")
            print(f"[mouse_executor_stub] command_id={cmd.command_id} command={cmd.command} target={cmd.target} bbox={bbox} reason={cmd.reason} safety={cmd.safety}")
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
