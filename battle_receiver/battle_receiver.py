#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battle_receiver.py

Receive raw POI exporter events and produce normalized KC agent event packets.

Receive target:
    http://127.0.0.1:8765/poi-event

Plugin load confirmation target:
    http://127.0.0.1:8765/poi-plugin-loaded

This is a standalone normalizer. It intentionally copies the event normalization
logic from kc_agent.py so raw POI events can be converted before they enter the
main agent state machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utility.logger import setup_logger


JsonDict = dict[str, Any]

logger = setup_logger(name="battle_receiver")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_RAW_LOG_PATH = Path("poi_raw_events.jsonl")
DEFAULT_NORMALIZED_LOG_PATH = Path("normalized_battle_events.jsonl")


def now_ms() -> int:
    return int(time.time() * 1000)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def append_jsonl(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


@dataclass
class AgentEvent:
    type: str
    payload: JsonDict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_id("evt"))
    ts_ms: int = field(default_factory=now_ms)
    source: str = "unknown"
    correlation: JsonDict = field(default_factory=dict)

    def to_packet(self) -> JsonDict:
        return asdict(self)


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


def analyze_damage_from_fleet(fleet: JsonDict) -> JsonDict:
    reports: list[JsonDict] = []
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

    taiha = [ship for ship in reports if ship["state"] == "taiha"]
    chuuha = [ship for ship in reports if ship["state"] == "chuuha"]
    return {
        "has_taiha": bool(taiha),
        "has_chuuha": bool(chuuha),
        "ships": reports,
        "taiha_ships": taiha,
        "chuuha_ships": chuuha,
    }


def classify_node(body: JsonDict) -> tuple[str, bool, bool]:
    route_no = to_int(body.get("api_no"))
    color_no = to_int(body.get("api_color_no"))
    event_id = to_int(body.get("api_event_id"))
    event_kind = to_int(body.get("api_event_kind"))
    boss_cell_no = to_int(body.get("api_bosscell_no"))

    is_boss = (
        route_no is not None and boss_cell_no is not None and route_no == boss_cell_no
    ) or color_no == 5 or event_id == 5
    is_battle = (
        is_boss
        or color_no in (4, 5)
        or event_kind in (1, 2, 3, 4)
        or bool(body.get("api_e_deck_info"))
    )

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


def active_fleet_from_snapshot(raw: JsonDict, deck_id: int = 1) -> JsonDict:
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


def normalize_poi_raw_event(raw: JsonDict) -> Optional[AgentEvent]:
    """Convert a raw POI exporter event into an AgentEvent packet."""
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
            "expected": {
                "user_input_expected": False,
                "ui_wait_reason": "waiting_map_start_response",
                "next_scenes": [],
            },
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
            next_scenes.append({
                "scene": "compass_or_map_production",
                "required_targets": [],
                "timeout_ms": 8000,
            })
        if is_battle:
            next_scenes.append({
                "scene": "formation_select",
                "required_targets": ["formation_buttons"],
                "timeout_ms": 10000,
            })

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
                "is_first_node_after_sortie_start": (
                    path == "/kcsapi/api_req_map/start"
                    and to_int(body.get("api_from_no")) == 0
                ),
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
            "expected": {
                "user_input_expected": bool(next_scenes),
                "ui_wait_reason": (
                    "sortie_first_node_arrived"
                    if source_kind == "map_start"
                    else "advanced_to_next_node"
                ),
                "next_scenes": next_scenes,
            },
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
            "battle": {
                "kind": "normal_day",
                "formation": {
                    "requested": to_int(req.get("api_formation")),
                    "friendly": None,
                    "enemy": None,
                    "engagement": None,
                },
            },
            "fleet": active_fleet_from_snapshot(raw, 1),
            "expected": {
                "user_input_expected": False,
                "ui_wait_reason": "battle_started",
                "next_scenes": [],
            },
        }
        return AgentEvent("formation_selected", payload, new_id("poi"), ts_ms, "poi")

    if name == "sortie_battle_response":
        formation = body.get("api_formation") or []
        can_night = to_int(body.get("api_midnight_flag")) == 1
        next_scenes = []
        if can_night:
            next_scenes.append({
                "scene": "night_battle_choice",
                "required_targets": ["night_battle_button", "no_night_battle_button"],
                "timeout_ms": 10000,
            })

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
            "enemy": {
                "ship_ids": body.get("api_ship_ke") or [],
                "start_hp": body.get("api_e_nowhps") or [],
                "max_hp": body.get("api_e_maxhps") or [],
            },
            "expected": {
                "user_input_expected": can_night,
                "ui_wait_reason": "can_night_battle" if can_night else "waiting_battle_result",
                "next_scenes": next_scenes,
            },
        }
        return AgentEvent("day_battle_received", payload, new_id("poi"), ts_ms, "poi")

    if name == "battle_result" or raw.get("phase") == "poi_battle_result":
        result = raw.get("battle_result") or raw.get("result") or {}
        fleet = active_fleet_from_snapshot(raw, 1)
        deck_ship_ids = result.get("deck_ship_id") or result.get("deckShipId") or []
        deck_hp = result.get("deck_hp") or result.get("deckHp") or []
        deck_init_hp = result.get("deck_init_hp") or result.get("deckInitHp") or []

        hp_by_iid: dict[int, JsonDict] = {}
        for idx, iid in enumerate(deck_ship_ids):
            iid_int = to_int(iid)
            if iid_int is None:
                continue
            hp_by_iid[iid_int] = {
                "battle_end_hp": to_int(deck_hp[idx]) if idx < len(deck_hp) else None,
                "battle_start_hp": to_int(deck_init_hp[idx]) if idx < len(deck_init_hp) else None,
            }

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
            "battle": {
                "rank": result.get("rank"),
                "boss": result.get("boss"),
                "map": result.get("map"),
                "map_cell": result.get("map_cell") or result.get("mapCell"),
                "enemy_name": result.get("enemy"),
                "mvp": result.get("mvp") or [],
                "drop": {
                    "ship_id": result.get("drop_ship_id") or result.get("dropShipId"),
                    "item": result.get("drop_item") or result.get("dropItem"),
                    "event_item": result.get("event_item") or result.get("eventItem"),
                },
            },
            "fleet": fleet,
            "damage": damage,
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {
                "user_input_expected": True,
                "ui_wait_reason": "battle_result_confirm_then_advance_or_retreat",
                "next_scenes": [
                    {
                        "scene": "battle_result_confirm",
                        "required_targets": ["result_confirm_button"],
                        "timeout_ms": 8000,
                    },
                    {
                        "scene": "advance_or_retreat",
                        "required_targets": required_targets,
                        "timeout_ms": 10000,
                    },
                ],
            },
        }
        return AgentEvent("battle_result", payload, new_id("poi"), ts_ms, "poi")

    return None


@dataclass
class ReceiverConfig:
    raw_log_path: Path
    normalized_log_path: Path
    print_raw: bool = False
    print_normalized: bool = True


def make_handler(config: ReceiverConfig):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path not in {"/poi-event", "/poi-plugin-loaded"}:
                self.send_plain_response(404, b"not found")
                return

            raw_event = self.read_json_body()
            if raw_event is None:
                return

            if self.path == "/poi-plugin-loaded":
                print_plugin_loaded(raw_event)
                self.send_plain_response(200, b"ok")
                return

            append_jsonl(config.raw_log_path, raw_event)
            normalized = normalize_poi_raw_event(raw_event)
            if normalized is None:
                print_unsupported(raw_event)
                self.send_plain_response(202, b"ignored unsupported event")
                return

            packet = normalized.to_packet()
            append_jsonl(config.normalized_log_path, packet)
            print_received(raw_event, packet, config)
            self.send_json_response(200, packet)

        def do_OPTIONS(self) -> None:
            if self.path not in {"/health", "/poi-event", "/poi-plugin-loaded"}:
                self.send_plain_response(404, b"not found")
                return

            self.send_response(204)
            self.send_cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_plain_response(200, b"ok")
                return

            self.send_plain_response(404, b"not found")

        def read_json_body(self) -> Optional[JsonDict]:
            length_str = self.headers.get("Content-Length", "0")
            try:
                length = int(length_str)
            except ValueError:
                self.send_plain_response(411, b"invalid content-length")
                return None

            raw_body = self.rfile.read(length)
            try:
                parsed = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                self.send_plain_response(400, f"bad json: {exc}".encode("utf-8"))
                return None

            if not isinstance(parsed, dict):
                self.send_plain_response(400, b"json body must be an object")
                return None

            return parsed

        def send_json_response(self, status: int, payload: JsonDict) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_plain_response(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")

        def log_message(self, fmt, *args) -> None:
            return

    return Handler


def print_plugin_loaded(payload: JsonDict) -> None:
    logger.info(
        "\n%s\n[%s] POI PLUGIN CONNECTION CONFIRMED\nplugin=%s\nmessage=%s\n%s",
        "=" * 100,
        now_str(),
        payload.get("plugin"),
        payload.get("message"),
        "=" * 100,
    )


def print_unsupported(raw_event: JsonDict) -> None:
    logger.warning(
        "\n%s\n[%s] unsupported raw event ignored\nseq=%s event=%s phase=%s\npath=%s\n%s",
        "-" * 100,
        now_str(),
        raw_event.get("seq"),
        raw_event.get("event"),
        raw_event.get("phase"),
        raw_event.get("path"),
        "-" * 100,
    )


def print_received(raw_event: JsonDict, packet: JsonDict, config: ReceiverConfig) -> None:
    lines = [
        "",
        "=" * 100,
        f"[{now_str()}] normalized POI event",
        (
            f"raw: seq={raw_event.get('seq')} event={raw_event.get('event')} "
            f"phase={raw_event.get('phase')} path={raw_event.get('path')}"
        ),
        (
            f"packet: type={packet.get('type')} event_id={packet.get('event_id')} "
            f"source={packet.get('source')} ts_ms={packet.get('ts_ms')}"
        ),
        f"raw_log={config.raw_log_path.resolve()}",
        f"normalized_log={config.normalized_log_path.resolve()}",
    ]

    if config.print_raw:
        lines.extend([
            "-" * 100,
            json.dumps(raw_event, ensure_ascii=False, indent=2),
        ])
    if config.print_normalized:
        lines.extend([
            "-" * 100,
            json.dumps(packet, ensure_ascii=False, indent=2),
        ])

    lines.append("=" * 100)
    logger.info("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive POI raw events and emit normalized KC agent event packets."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--raw-log", type=Path, default=DEFAULT_RAW_LOG_PATH)
    parser.add_argument("--normalized-log", type=Path, default=DEFAULT_NORMALIZED_LOG_PATH)
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument(
        "--quiet-normalized",
        action="store_true",
        help="Do not print full normalized packets to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ReceiverConfig(
        raw_log_path=args.raw_log,
        normalized_log_path=args.normalized_log,
        print_raw=args.print_raw,
        print_normalized=not args.quiet_normalized,
    )

    logger.info("Listening on http://%s:%s/poi-event", args.host, args.port)
    logger.info("Plugin load confirmation: http://%s:%s/poi-plugin-loaded", args.host, args.port)
    logger.info("Health check: http://%s:%s/health", args.host, args.port)
    logger.info("Raw log: %s", config.raw_log_path.resolve())
    logger.info("Normalized log: %s", config.normalized_log_path.resolve())
    logger.info("Start poi and enable the event exporter plugin. Press Ctrl+C to stop.")

    server = HTTPServer((args.host, args.port), make_handler(config))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping.")
        server.server_close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
