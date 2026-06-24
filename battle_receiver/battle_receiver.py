#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
battle_receiver.py

Receive raw POI exporter events and produce normalized KC agent event packets.

Receive target:
    http://127.0.0.1:8765/poi-event

Plugin load confirmation target:
    http://127.0.0.1:8765/poi-plugin-loaded

This is a standalone HTTP receiver. It delegates normalization to
`kc_core.decoder.PoiEventDecoder` so decoding rules live in one testable module
before events enter the main agent state machine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from kc_core.decoder import normalize_poi_raw_event
from kc_core.event_models import JsonDict


logger = setup_logger(name="battle_receiver")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_RAW_LOG_PATH = Path("poi_raw_events.jsonl")
DEFAULT_NORMALIZED_LOG_PATH = Path("normalized_battle_events.jsonl")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_jsonl(path: Path, obj: JsonDict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    except OSError as exc:
        logger.warning("failed to append jsonl path=%s error=%r", path, exc)


@dataclass
class ReceiverConfig:
    raw_log_path: Path
    normalized_log_path: Path
    print_raw: bool = False
    print_normalized: bool = True
    packet_sink: Optional[Callable[[JsonDict], None]] = None


def make_handler(config: ReceiverConfig):
    fleet_state = FleetStateTracker()

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
            #　main decode part
            normalized = normalize_poi_raw_event(raw_event, fleet_state=fleet_state)
            if normalized is None:
                print_unsupported(raw_event)
                self.send_plain_response(202, b"ignored unsupported event")
                return

            packet = normalized.to_packet()
            append_jsonl(config.normalized_log_path, packet)
            print_received(raw_event, packet, config)
            if config.packet_sink is not None:
                config.packet_sink(packet)
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
