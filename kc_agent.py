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
from typing import Any, Optional

from kc_core.decoder import normalize_poi_raw_event
from kc_core.event_models import AgentEvent, Command, WaitSpec, new_id, to_int


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
