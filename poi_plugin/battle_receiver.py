#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
poi_event_viewer.py

Purpose:
    Receive battle-related events exported from poi plugin and print them
    in a readable way.

Expected plugin POST target:
    http://127.0.0.1:8765/poi-event

Plugin load confirmation target:
    http://127.0.0.1:8765/poi-plugin-loaded

Expected event shape from updated poi plugin:
    {
        "event": "...",
        "phase": "request" | "response" | "poi_battle_result",
        "source": "game.request" | "game.response" | "@@BattleResult",
        "method": "GET" | "POST" | null,
        "path": "/kcsapi/...",
        "request_body": {...} | null,
        "response_summary": {...} | null,
        "response_body": {...} | null,
        "battle_result": {...} | null,
        "fleet_snapshot": [...],
        "sortie_snapshot": {...},
        "session_id": "...",
        "event_id": "<session_id>:<seq>",
        "seq": 1,
        "exported_at": 1710000000000
    }

Notes:
    This script is for mechanism verification only.
    It does not make decisions and does not control mouse/keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import argparse
import json
import math
import sys
import textwrap


JsonDict = Dict[str, Any]


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_FORMATION_HISTORY_PATH = Path("poi_formation_history.json")
IGNORED_LOG_EVENTS = {
    "game_start_response",
    "require_info_response",
    "port_response",
    "ship3_update_response",
    "sortie_battle_result_response",
}

MAP_RESPONSE_PATHS = {
    "/kcsapi/api_req_map/start",
    "/kcsapi/api_req_map/next",
}

NODE_COLOR_NAMES = {
    0: "起點／無事件",
    2: "資源點",
    3: "漩渦點",
    4: "戰鬥點",
    5: "Boss點",
    6: "無事件點",
    7: "航空戰點",
    8: "港口／返航",
    9: "航空偵察",
}

PRODUCTION_KIND_NAMES = {
    0: "無地圖演出",
    1: "偵察機演出",
}


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"poi_event_{timestamp}.jsonl")


FORMATION_NAMES = {
    1: "單縱陣",
    2: "複縱陣",
    3: "輪形陣",
    4: "梯形陣",
    5: "單橫陣",
    6: "警戒陣",
    11: "第一警戒航行序列（對潛警戒）",
    12: "第二警戒航行序列（前方警戒）",
    13: "第三警戒航行序列（防空警戒）",
    14: "第四警戒航行序列（戰鬥隊形）",
}

ENGAGEMENT_NAMES = {
    1: "同航戰",
    2: "反航戰",
    3: "T字有利",
    4: "T字不利",
}


# -----------------------------
# Generic helpers
# -----------------------------

def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ms_to_datetime_string(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    try:
        return datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        return str(ms)


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    """
    Safe nested dict/list getter.

    Example:
        get_path(event, "response_summary.battle.api_formation")
    """
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return default
        else:
            return default
    return cur


def short_json(obj: Any, max_len: int = 800) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = repr(obj)

    if len(s) > max_len:
        return s[:max_len] + "\n... <truncated>"
    return s


def compact_json(obj: Any) -> str:
    if obj is None:
        return "-"
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return repr(obj)


def value_with_name(value: Any, names: Dict[int, str]) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value} ({names.get(value, '未知值')})"
    return str(value)


def formation_with_name(value: Any) -> str:
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return value_with_name(value, FORMATION_NAMES)


def node_event_name(event_id: Any, event_kind: Any) -> str:
    """Describe api_event_id using the context supplied by api_event_kind."""
    pair_names = {
        (4, 1): "通常晝戰",
        (4, 2): "夜戰",
        (4, 3): "夜轉晝戰",
        (4, 4): "航空戰",
        (4, 5): "敵聯合艦隊戰",
        (4, 6): "長距離空襲戰",
        (6, 1): "迴避戰鬥／未見敵影",
        (6, 2): "能動分岐",
        (7, 0): "航空偵察",
        (10, 0): "泊地修理",
    }
    base_names = {
        0: "起點／無事件",
        1: "迴避戰鬥",
        2: "獲得資源",
        3: "損失資源／漩渦",
        4: "戰鬥",
        5: "Boss戰",
        6: "無事件／特殊點",
        7: "航空事件",
        8: "護衛成功",
        9: "輸送物資",
        10: "長距離空襲／特殊點",
    }
    return pair_names.get((event_id, event_kind), base_names.get(event_id, "未知值"))


def event_kind_name(event_id: Any, event_kind: Any) -> str:
    """api_event_kind is contextual, so reuse the decoded event pair."""
    if event_kind is None:
        return "-"
    return f"{event_kind} ({node_event_name(event_id, event_kind)})"


def node_type(event_id: Any, event_kind: Any, color_no: Any = None) -> JsonDict:
    """Return a stable broad node category plus the detailed API meaning."""
    categories = {
        0: ("start", "起點／無事件"),
        1: ("no_event", "無事件"),
        2: ("resource", "資源獲得"),
        3: ("whirlpool", "資源損失／漩渦"),
        4: ("battle", "戰鬥"),
        5: ("boss_battle", "Boss戰鬥"),
        6: (
            "route_selection" if event_kind == 2 else "no_event",
            "能動分岐" if event_kind == 2 else "無事件／特殊點",
        ),
        7: ("aviation", "航空事件"),
        8: ("escort", "護衛／特殊任務"),
        9: ("transport", "輸送物資"),
        10: ("repair_or_air_raid", "泊地修理／空襲"),
    }
    color_fallback = {
        0: ("start", "起點／無事件"),
        2: ("resource", "資源獲得"),
        3: ("whirlpool", "資源損失／漩渦"),
        4: ("battle", "戰鬥"),
        5: ("boss_battle", "Boss戰鬥"),
        6: ("no_event", "無事件"),
        7: ("aviation", "航空戰"),
        8: ("port", "港口／返航"),
        9: ("recon", "航空偵察"),
    }
    if (event_id, event_kind) == (6, 2):
        code, name = ("route_selection", "能動分岐")
    elif color_no in color_fallback and color_no != 0:
        # api_color_no is the most direct broad visual category used by the map.
        code, name = color_fallback[color_no]
    else:
        code, name = categories.get(
            event_id,
            color_fallback.get(color_no, ("unknown", "未知")),
        )
    return {
        "code": code,
        "name": name,
        "detail": node_event_name(event_id, event_kind),
    }


def explain_boss_cell(value: Any) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0 (未提供／無Boss節點)"
    return f"{value} (Boss節點編號)"


def explain_uncertain_flag(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value} (公開定義未明，保留原值)"


def explain_midnight_flag(value: Any) -> str:
    meanings = {
        0: "不可／不顯示夜戰選擇",
        1: "可選擇進入夜戰",
    }
    return value_with_name(value, meanings)


def explain_airsearch(value: Any) -> str:
    if not isinstance(value, dict):
        return "-" if value is None else compact_json(value)

    plane_type = value.get("api_plane_type")
    result = value.get("api_result")
    plane_meaning = "無／未指定" if plane_type == 0 else "機種值定義未明"
    result_meaning = "無結果／未成功" if result == 0 else "結果值定義未明"
    return (
        f"{compact_json(value)} "
        f"(機種={plane_type}:{plane_meaning}；結果={result}:{result_meaning})"
    )


def explain_eventmap(value: Any) -> str:
    if not isinstance(value, dict):
        return "-" if value is None else compact_json(value)

    return (
        f"{compact_json(value)} "
        f"(最大={value.get('api_max_maphp')}；"
        f"目前={value.get('api_now_maphp')}；"
        f"損傷值={value.get('api_dmg')}, 定義依活動機制)"
    )


def explain_select_route(value: Any) -> str:
    if not isinstance(value, dict):
        return "-" if value is None else compact_json(value)

    cells = value.get("api_select_cells")
    return f"{compact_json(value)} (可選目的節點={compact_json(cells)})"


def extract_response_data(event: JsonDict) -> JsonDict:
    body = event.get("response_body")
    if not isinstance(body, dict):
        return {}
    if isinstance(body.get("api_data"), dict):
        return body["api_data"]
    return body


def extract_map_response(event: JsonDict) -> JsonDict:
    if event.get("path") not in MAP_RESPONSE_PATHS:
        return {}

    body = extract_response_data(event)

    summary = event.get("response_summary")
    summary_map = summary.get("map") if isinstance(summary, dict) else None
    if not isinstance(summary_map, dict):
        summary_map = {}

    # Raw response data contains optional fields omitted by older exporter summaries.
    return {**summary_map, **body}


def is_battle_result_response(event: JsonDict) -> bool:
    path = event.get("path") or ""
    name = event.get("event") or ""
    return (
        "battleresult" in path
        or "battle_result" in path
        or name.endswith("_battle_result_response")
    )


def format_drop_ship(drop_ship: Any) -> str:
    if not isinstance(drop_ship, dict):
        return "無"

    ship_name = drop_ship.get("api_ship_name") or drop_ship.get("ship_name") or "名稱未知"
    ship_type = drop_ship.get("api_ship_type") or drop_ship.get("ship_type")
    ship_id = drop_ship.get("api_ship_id") or drop_ship.get("ship_id")
    details = [part for part in (ship_type, f"sid={ship_id}" if ship_id is not None else None) if part]
    return f"{ship_name} ({', '.join(details)})" if details else str(ship_name)


def print_kv(label: str, value: Any, indent: int = 2) -> None:
    prefix = " " * indent
    print(f"{prefix}{label:<20}: {value}")


def append_jsonl(path: Path, event: JsonDict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


def load_last_packet_no(path: Path) -> int:
    if not path.exists():
        return 0

    last_packet_no = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                packet_no = record.get("packet_no") if isinstance(record, dict) else None
                if isinstance(packet_no, int):
                    last_packet_no = max(last_packet_no, packet_no)
    except OSError:
        return 0

    return last_packet_no


def load_receiver_context(path: Path) -> tuple[Dict[Any, str], Optional[JsonDict]]:
    map_names: Dict[Any, str] = {}
    last_selected_formation: Optional[JsonDict] = None
    if not path.exists():
        return map_names, last_selected_formation

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                updates = record.get("updates")
                if not isinstance(updates, dict):
                    continue
                map_update = updates.get("map")
                if isinstance(map_update, dict):
                    map_name = map_update.get("map_name")
                    map_id = map_update.get("map_id")
                    maparea_id = map_update.get("api_maparea_id")
                    mapinfo_no = map_update.get("api_mapinfo_no")
                    if isinstance(map_name, str):
                        if isinstance(map_id, int):
                            map_names[map_id] = map_name
                        if isinstance(maparea_id, int) and isinstance(mapinfo_no, int):
                            map_names[(maparea_id, mapinfo_no)] = map_name
                formation = updates.get("formation")
                if isinstance(formation, dict) and isinstance(
                    formation.get("last_selected"),
                    dict,
                ):
                    last_selected_formation = formation["last_selected"]
    except OSError:
        pass

    return map_names, last_selected_formation


def load_formation_history(path: Path) -> Dict[str, JsonDict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, dict)
    }


def save_formation_history(path: Path, history: Dict[str, JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def prune_empty(value: Any) -> Any:
    """Remove unavailable fields while preserving valid 0/False values."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            cleaned = prune_empty(child)
            if cleaned is not None and cleaned != {} and cleaned != [] and cleaned != "":
                result[key] = cleaned
        return result
    if isinstance(value, list):
        result = [prune_empty(item) for item in value]
        return [item for item in result if item is not None and item != {} and item != []]
    return value


def named_value(value: Any, names: Dict[int, str]) -> Optional[JsonDict]:
    if value is None:
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return {
        "value": value,
        "name": names.get(value, "unknown") if isinstance(value, int) else "unknown",
    }


def normalize_fleet_rows(
    ships: Any,
    ship_names: Dict[int, str],
    enemy: bool = False,
) -> List[JsonDict]:
    if not isinstance(ships, list):
        return []

    result = []
    for index, ship in enumerate(ships):
        if not isinstance(ship, dict):
            continue
        now_hp = ship.get("now_hp")
        max_hp = ship.get("max_hp")
        ship_id = ship.get("ship_id")
        hp_text = None
        if now_hp is not None or max_hp is not None:
            hp_text = f"{now_hp if now_hp is not None else '?'}/{max_hp if max_hp is not None else '?'}"
        row = {
            "position": ship.get("position", index + 1),
            "ship_id": ship_id,
            "hp": hp_text,
            "now_hp": now_hp,
            "max_hp": max_hp,
            "damage": damage_state(now_hp, max_hp),
        }
        if enemy:
            row["name"] = ship_names.get(ship_id, "名稱未載入")
            for key in (
                "initial_hp",
                "raw_now_hp",
                "lost_hp",
                "used_damage_control",
            ):
                row[key] = ship.get(key)
        else:
            row.update(
                {
                    "instance_id": ship.get("instance_id"),
                    "morale": ship.get("cond"),
                    "ammo": ship.get("bullet"),
                    "fuel": ship.get("fuel"),
                }
            )
            for key in (
                "initial_hp",
                "raw_now_hp",
                "lost_hp",
                "used_damage_control",
            ):
                row[key] = ship.get(key)
        result.append(prune_empty(row))
    return result


def normalize_enemy_api_rows(
    ship_ids: Any,
    ship_names: Dict[int, str],
    levels: Any = None,
    now_hps: Any = None,
    max_hps: Any = None,
) -> List[JsonDict]:
    if not isinstance(ship_ids, list):
        return []

    rows = []
    for index, ship_id in enumerate(ship_ids):
        if not isinstance(ship_id, int) or ship_id <= 0:
            continue
        now_hp = now_hps[index] if isinstance(now_hps, list) and index < len(now_hps) else None
        max_hp = max_hps[index] if isinstance(max_hps, list) and index < len(max_hps) else None
        level = levels[index] if isinstance(levels, list) and index < len(levels) else None
        hp_text = None
        if now_hp is not None or max_hp is not None:
            hp_text = f"{now_hp if now_hp is not None else '?'}/{max_hp if max_hp is not None else '?'}"
        rows.append(
            prune_empty(
                {
                    "position": index + 1,
                    "ship_id": ship_id,
                    "name": ship_names.get(ship_id, "名稱未載入"),
                    "level": level,
                    "hp": hp_text,
                    "now_hp": now_hp,
                    "max_hp": max_hp,
                    "damage": damage_state(now_hp, max_hp),
                }
            )
        )
    return rows


def build_unified_updates(
    event: JsonDict,
    ship_names: Dict[int, str],
    map_names: Dict[Any, str],
    last_selected_formation: Optional[JsonDict],
    formation_history_hint: Optional[JsonDict],
) -> JsonDict:
    """Convert an unchanged plugin payload into receiver-owned partial updates."""
    updates: JsonDict = {}
    request_body = event.get("request_body")
    if isinstance(request_body, dict) and request_body:
        updates["request"] = prune_empty(request_body)

    summary = event.get("response_summary")
    response_body = event.get("response_body")
    response_data = extract_response_data(event)
    api_result = summary.get("api_result") if isinstance(summary, dict) else None
    api_result_msg = summary.get("api_result_msg") if isinstance(summary, dict) else None
    if api_result is None and isinstance(response_body, dict):
        api_result = response_body.get("api_result")
        api_result_msg = response_body.get("api_result_msg")
    api_update = prune_empty(
        {
            "api_result": api_result,
            "api_result_msg": api_result_msg,
        }
    )
    if api_update:
        updates["api"] = api_update

    map_data = extract_map_response(event)
    if map_data:
        maparea_id = map_data.get("api_maparea_id")
        mapinfo_no = map_data.get("api_mapinfo_no")
        map_id = (
            maparea_id * 10 + mapinfo_no
            if isinstance(maparea_id, int) and isinstance(mapinfo_no, int)
            else None
        )
        map_update = {
            key: map_data.get(key)
            for key in (
                "api_maparea_id",
                "api_mapinfo_no",
                "api_no",
                "api_color_no",
                "api_event_id",
                "api_event_kind",
                "api_bosscell_no",
                "api_bosscomp",
                "api_production_kind",
                "api_airsearch",
                "api_eventmap",
                "api_select_route",
                "api_limit_state",
            )
        }
        map_update["map_id"] = map_id
        map_update["map_name"] = (
            map_names.get((maparea_id, mapinfo_no))
            or map_names.get(map_id)
        )
        map_update["event_name"] = node_event_name(
            map_data.get("api_event_id"),
            map_data.get("api_event_kind"),
        )
        current_node_type = node_type(
            map_data.get("api_event_id"),
            map_data.get("api_event_kind"),
            map_data.get("api_color_no"),
        )
        map_update["node_type"] = current_node_type
        updates["map"] = prune_empty(map_update)

        if event.get("event") in {"map_start_response", "map_next_response"}:
            prompt_timing = event.get("event")
            history_key = (
                formation_history_hint.get("key")
                if formation_history_hint
                else (
                    f"{map_id}-{map_data.get('api_no')}"
                    if map_id is not None and map_data.get("api_no") is not None
                    else None
                )
            )
            updates["formation"] = {
                "prompt_timing": prompt_timing,
                "history_key": history_key,
                "last_selected": formation_history_hint
                or {
                    "value": None,
                    "name": "Unknown",
                    "key": history_key,
                    "known": False,
                },
            }

        enemy_preview = []
        for deck in map_data.get("api_e_deck_info") or []:
            if not isinstance(deck, dict):
                continue
            enemy_preview.append(
                prune_empty(
                    {
                        "kind": deck.get("api_kind"),
                        "ships": normalize_enemy_api_rows(
                            deck.get("api_ship_ids"),
                            ship_names,
                        ),
                    }
                )
            )
        if enemy_preview:
            updates["enemy_preview"] = enemy_preview

    if event.get("event") not in {
        "map_start_response",
        "map_next_response",
        "day_battle_hp_update",
        "battle_result",
    }:
        formation = response_data.get("api_formation")
        selected_formation = (
            request_body.get("api_formation")
            if isinstance(request_body, dict)
            else None
        )
        if selected_formation is None and isinstance(formation, list) and formation:
            selected_formation = formation[0]
        formation_update = {
            "friendly": named_value(selected_formation, FORMATION_NAMES),
            "enemy": named_value(
                formation[1]
                if isinstance(formation, list) and len(formation) > 1
                else None,
                FORMATION_NAMES,
            ),
            "engagement": named_value(
                formation[2]
                if isinstance(formation, list) and len(formation) > 2
                else None,
                ENGAGEMENT_NAMES,
            ),
        }
        if last_selected_formation and any(formation_update.values()):
            formation_update["last_selected"] = last_selected_formation
        formation_update = prune_empty(formation_update)
        if formation_update:
            updates["formation"] = formation_update

    if "api_midnight_flag" in response_data:
        midnight_flag = response_data.get("api_midnight_flag")
        updates["night_battle"] = {
            "api_midnight_flag": midnight_flag,
            "available": midnight_flag == 1,
        }

    fleets = event.get("fleet_snapshot")
    if isinstance(fleets, list) and fleets:
        main = next((fleet for fleet in fleets if fleet.get("deck_id") == 1), fleets[0])
        player_fleet = {
            "main": normalize_fleet_rows(main.get("ships"), ship_names),
        }
        combined_flag = (event.get("sortie_snapshot") or {}).get("combined_flag")
        if combined_flag:
            escort = next((fleet for fleet in fleets if fleet.get("deck_id") == 2), None)
            if escort:
                player_fleet["escort"] = normalize_fleet_rows(
                    escort.get("ships"),
                    ship_names,
                )
        updates["player_fleet"] = prune_empty(player_fleet)

    if isinstance(response_data.get("api_ship_ke"), list):
        enemy_fleet = {
            "main": normalize_enemy_api_rows(
                response_data.get("api_ship_ke"),
                ship_names,
                response_data.get("api_ship_lv"),
                response_data.get("api_e_nowhps"),
                response_data.get("api_e_maxhps"),
            ),
            "escort": normalize_enemy_api_rows(
                response_data.get("api_ship_ke_combined"),
                ship_names,
                response_data.get("api_ship_lv_combined"),
                response_data.get("api_e_nowhps_combined"),
                response_data.get("api_e_maxhps_combined"),
            ),
        }
        updates["enemy_fleet"] = prune_empty(enemy_fleet)
        updates["battle"] = prune_empty(
            {
                "stage": "battle_response",
                "deck_id": response_data.get("api_deck_id"),
            }
        )

    day_hp = event.get("day_battle_hp")
    if isinstance(day_hp, dict):
        if "api_midnight_flag" in day_hp:
            midnight_flag = day_hp.get("api_midnight_flag")
            updates["night_battle"] = {
                "api_midnight_flag": midnight_flag,
                "available": midnight_flag == 1,
            }
        updates["battle"] = prune_empty(
            {
                "stage": "day_battle_end",
                "valid": day_hp.get("valid"),
                "simulator": day_hp.get("simulator"),
                "derived_from_event": day_hp.get("derived_from_event"),
                "derived_from_event_id": day_hp.get("derived_from_event_id"),
                "deck_id": day_hp.get("deck_id"),
                "combined_flag": day_hp.get("combined_flag"),
            }
        )
        updates["player_fleet"] = prune_empty(
            {
                "main": normalize_fleet_rows(
                    day_hp.get("player_main"),
                    ship_names,
                ),
                "escort": normalize_fleet_rows(
                    day_hp.get("player_escort"),
                    ship_names,
                ),
            }
        )
        updates["enemy_fleet"] = prune_empty(
            {
                "main": normalize_fleet_rows(
                    day_hp.get("enemy_main"),
                    ship_names,
                    enemy=True,
                ),
                "escort": normalize_fleet_rows(
                    day_hp.get("enemy_escort"),
                    ship_names,
                    enemy=True,
                ),
            }
        )

    battle_result = event.get("battle_result")
    if isinstance(battle_result, dict):
        updates["battle_result"] = prune_empty(battle_result)
        updates["battle"] = prune_empty(
            {
                "stage": "battle_result",
                "rank": battle_result.get("rank"),
                "boss": battle_result.get("boss"),
                "map": battle_result.get("map"),
                "map_cell": battle_result.get("map_cell"),
                "enemy": battle_result.get("enemy"),
                "drop_ship": {
                    "ship_id": battle_result.get("drop_ship_id"),
                    "name": battle_result.get("drop_ship_name"),
                    "type": battle_result.get("drop_ship_type"),
                },
            }
        )
        result_map_id = battle_result.get("map")
        result_map_name = map_names.get(result_map_id)
        if result_map_id is not None or result_map_name:
            updates["map"] = prune_empty(
                {
                    "map_id": result_map_id,
                    "map_name": result_map_name,
                    "api_no": battle_result.get("map_cell"),
                }
            )
        snapshot_by_iid = {}
        for fleet in fleets or []:
            for ship in fleet.get("ships") or []:
                instance_id = ship.get("instance_id")
                if isinstance(instance_id, int):
                    snapshot_by_iid[instance_id] = ship

        player_rows = []
        deck_ship_ids = battle_result.get("deck_ship_id") or []
        deck_hps = battle_result.get("deck_hp") or []
        deck_init_hps = battle_result.get("deck_init_hp") or []
        for index, instance_id in enumerate(deck_ship_ids):
            if not isinstance(instance_id, int) or instance_id <= 0:
                continue
            snapshot_ship = snapshot_by_iid.get(instance_id, {})
            player_rows.append(
                {
                    **snapshot_ship,
                    "position": index + 1,
                    "instance_id": instance_id,
                    "now_hp": (
                        max(0, deck_hps[index])
                        if index < len(deck_hps) and isinstance(deck_hps[index], int)
                        else None
                    ),
                    "raw_now_hp": (
                        deck_hps[index] if index < len(deck_hps) else None
                    ),
                    "initial_hp": (
                        deck_init_hps[index]
                        if index < len(deck_init_hps)
                        else None
                    ),
                }
            )
        if player_rows:
            updates["player_fleet"] = {
                "main": normalize_fleet_rows(player_rows, ship_names)
            }

        enemy_rows = []
        enemy_ids = battle_result.get("enemy_ship_id") or []
        enemy_hps = battle_result.get("enemy_hp") or []
        for index, ship_id in enumerate(enemy_ids):
            if not isinstance(ship_id, int) or ship_id <= 0:
                continue
            raw_now_hp = enemy_hps[index] if index < len(enemy_hps) else None
            enemy_rows.append(
                {
                    "position": index + 1,
                    "ship_id": ship_id,
                    "now_hp": (
                        max(0, raw_now_hp)
                        if isinstance(raw_now_hp, int)
                        else None
                    ),
                    "raw_now_hp": raw_now_hp,
                }
            )
        if enemy_rows:
            updates["enemy_fleet"] = {
                "main": normalize_fleet_rows(
                    enemy_rows,
                    ship_names,
                    enemy=True,
                )
            }

    return prune_empty(updates)


def packet_error_for_event(event: JsonDict) -> Optional[JsonDict]:
    if not isinstance(event.get("event"), str):
        return {
            "code": "missing_event",
            "message": "plugin payload does not contain a string event field",
        }

    summary = event.get("response_summary")
    response_body = event.get("response_body")
    api_result = summary.get("api_result") if isinstance(summary, dict) else None
    api_result_msg = summary.get("api_result_msg") if isinstance(summary, dict) else None
    if api_result is None and isinstance(response_body, dict):
        api_result = response_body.get("api_result")
        api_result_msg = response_body.get("api_result_msg")
    if api_result not in (None, 1):
        return {
            "code": "api_error",
            "message": api_result_msg or f"api_result={api_result}",
        }

    day_hp = event.get("day_battle_hp")
    if isinstance(day_hp, dict) and day_hp.get("valid") is False:
        return {
            "code": "simulation_invalid",
            "message": "day battle HP event is marked invalid",
        }

    return None


def should_ignore_log_event(event: JsonDict) -> bool:
    return event.get("event") in IGNORED_LOG_EVENTS


def separator(width: int = 118) -> None:
    print("=" * width)


def sub_separator(width: int = 118) -> None:
    print("-" * width)


# -----------------------------
# Damage helper for display only
# -----------------------------

def numeric_value(value: Any) -> Optional[float]:
    """Return a finite number for comparisons; placeholders such as N/A stay unknown."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def damage_state(now_hp: Any, max_hp: Any) -> str:
    """
    Display helper only.

    We do not make decisions here.
    Integer comparison avoids float rounding:
        heavy damage   : now_hp * 4 <= max_hp
        moderate damage: now_hp * 2 <= max_hp
    """
    comparable_now_hp = numeric_value(now_hp)
    comparable_max_hp = numeric_value(max_hp)
    if comparable_now_hp is None or comparable_max_hp is None:
        return "unknown"
    if comparable_max_hp <= 0:
        return "unknown"
    if comparable_now_hp <= 0:
        return "zero_or_sunk"
    if comparable_now_hp * 4 <= comparable_max_hp:
        return "heavy_damage"
    if comparable_now_hp * 2 <= comparable_max_hp:
        return "moderate_damage"
    if comparable_now_hp < comparable_max_hp:
        return "minor_damage_or_scratch"
    return "normal"


# -----------------------------
# Event display
# -----------------------------

@dataclass
class ViewerConfig:
    log_path: Path
    show_full_body: bool = False
    show_fleet: bool = True
    show_only_battle_summary: bool = False
    formation_history_path: Path = DEFAULT_FORMATION_HISTORY_PATH


@dataclass
class DeliveryState:
    seen_event_ids: Set[str] = field(default_factory=set)
    last_seq_by_session: Dict[str, int] = field(default_factory=dict)

    def is_duplicate(self, event: JsonDict) -> bool:
        event_id = event.get("event_id")
        return isinstance(event_id, str) and event_id in self.seen_event_ids

    def ordering_error(self, event: JsonDict) -> Optional[str]:
        session_id = event.get("session_id")
        seq = event.get("seq")

        if not isinstance(session_id, str) or not isinstance(seq, int):
            return None

        previous = self.last_seq_by_session.get(session_id)
        if previous is not None and seq != previous + 1:
            return f"out of order: expected seq {previous + 1}, got {seq}"
        return None

    def record(self, event: JsonDict) -> None:
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            self.seen_event_ids.add(event_id)

        session_id = event.get("session_id")
        seq = event.get("seq")
        if isinstance(session_id, str) and isinstance(seq, int):
            self.last_seq_by_session[session_id] = seq


def load_delivery_state(path: Path) -> DeliveryState:
    state = DeliveryState()
    if not path.exists():
        return state

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict) and event.get("accepted") is not False:
                    state.record(event)
    except OSError as exc:
        print(f"Warning: could not load delivery state from {path}: {exc}")

    return state


class PoiEventPrinter:
    def __init__(self, config: ViewerConfig):
        self.config = config
        self.ship_names: Dict[int, str] = {}
        self.map_names, self.last_selected_formation = load_receiver_context(
            config.log_path
        )
        self.formation_history = load_formation_history(
            config.formation_history_path
        )
        self.current_spot_key: Optional[str] = None
        self.current_map_id: Optional[int] = None
        self.current_api_no: Optional[int] = None
        self.current_formation_hint: Optional[JsonDict] = None
        self.packet_no = load_last_packet_no(config.log_path)

    def observe_event(self, event: JsonDict) -> Optional[JsonDict]:
        """Update display caches even when an event is intentionally not logged."""
        map_data = extract_map_response(event)
        if map_data:
            maparea_id = map_data.get("api_maparea_id")
            mapinfo_no = map_data.get("api_mapinfo_no")
            api_no = map_data.get("api_no")
            map_id = (
                maparea_id * 10 + mapinfo_no
                if isinstance(maparea_id, int) and isinstance(mapinfo_no, int)
                else None
            )
            if isinstance(map_id, int) and isinstance(api_no, int):
                self.current_map_id = map_id
                self.current_api_no = api_no
                self.current_spot_key = f"{map_id}-{api_no}"
                self.current_formation_hint = self.formation_history.get(
                    self.current_spot_key
                )
            else:
                self.current_spot_key = None
                self.current_map_id = None
                self.current_api_no = None
                self.current_formation_hint = None

        request_body = event.get("request_body")
        selected_formation = (
            request_body.get("api_formation")
            if isinstance(request_body, dict)
            else None
        )
        selection_source = "request_body.api_formation"
        if selected_formation is None and self.last_selected_formation is None:
            response_formation = extract_response_data(event).get("api_formation")
            if isinstance(response_formation, list) and response_formation:
                selected_formation = response_formation[0]
                selection_source = "response_body.api_formation[0]"
        if isinstance(selected_formation, str) and selected_formation.isdigit():
            selected_formation = int(selected_formation)
        if selected_formation is not None:
            named = named_value(selected_formation, FORMATION_NAMES) or {}
            self.last_selected_formation = prune_empty(
                {
                    **named,
                    "key": self.current_spot_key,
                    "map_id": self.current_map_id,
                    "api_no": self.current_api_no,
                    "source": selection_source,
                    "event": event.get("event"),
                    "event_id": event.get("event_id"),
                    "seq": event.get("seq"),
                    "updated_at": datetime.now().astimezone().isoformat(
                        timespec="milliseconds"
                    ),
                }
            )
            if self.current_spot_key:
                self.formation_history[
                    self.current_spot_key
                ] = self.last_selected_formation
                self.current_formation_hint = self.last_selected_formation
                try:
                    save_formation_history(
                        self.config.formation_history_path,
                        self.formation_history,
                    )
                except OSError as exc:
                    return {
                        "code": "formation_history_write_failed",
                        "message": str(exc),
                    }

        if event.get("event") == "game_start_response":
            master_data = extract_response_data(event)
            ships = master_data.get("api_mst_ship")
            if isinstance(ships, list):
                for ship in ships:
                    if not isinstance(ship, dict):
                        continue
                    ship_id = ship.get("api_id")
                    ship_name = ship.get("api_name")
                    if isinstance(ship_id, int) and isinstance(ship_name, str):
                        self.ship_names[ship_id] = ship_name

            maps = master_data.get("api_mst_mapinfo")
            if isinstance(maps, list):
                for map_info in maps:
                    if not isinstance(map_info, dict):
                        continue
                    map_id = map_info.get("api_id")
                    maparea_id = map_info.get("api_maparea_id")
                    mapinfo_no = map_info.get("api_no")
                    map_name = map_info.get("api_name")
                    if not isinstance(map_name, str):
                        continue
                    if isinstance(map_id, int):
                        self.map_names[map_id] = map_name
                    if isinstance(maparea_id, int) and isinstance(mapinfo_no, int):
                        self.map_names[(maparea_id, mapinfo_no)] = map_name

        return None

    def next_packet_no(self) -> int:
        self.packet_no += 1
        return self.packet_no

    def make_record(
        self,
        event: Optional[JsonDict],
        *,
        accepted: bool,
        error: Optional[JsonDict] = None,
        include_updates: bool = True,
    ) -> JsonDict:
        event = event if isinstance(event, dict) else {}
        updates = (
            build_unified_updates(
                event,
                self.ship_names,
                self.map_names,
                self.last_selected_formation,
                self.current_formation_hint,
            )
            if include_updates
            else {}
        )
        if not self.config.show_fleet:
            updates.pop("player_fleet", None)

        packet_no = self.next_packet_no()
        record: JsonDict = {
            "schema": "battle_receiver.v1",
            "packet_no": packet_no,
            "error": error,
            "accepted": accepted,
            "received_at": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "event_id": event.get("event_id"),
            "session_id": event.get("session_id"),
            "seq": event.get("seq"),
            "event": event.get("event"),
            "phase": event.get("phase"),
            "source": event.get("source"),
            "method": event.get("method"),
            "path": event.get("path"),
            "exported_at": event.get("exported_at"),
            "updates": updates,
        }
        if self.config.show_full_body and event:
            record["raw_event"] = event
        return record

    def print_record(self, record: JsonDict) -> None:
        separator()
        print(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"\nUnified record saved to: {self.config.log_path.resolve()}")
        separator()

    def print_event(self, event: JsonDict) -> None:
        observe_error = self.observe_event(event)
        record = self.make_record(
            event,
            accepted=True,
            error=observe_error or packet_error_for_event(event),
        )
        self.print_record(record)

    def print_plugin_loaded(self, payload: JsonDict) -> None:
        event = {
            "event": "plugin_loaded",
            "phase": "receiver",
            "source": "poi_plugin",
            "path": "/poi-plugin-loaded",
        }
        record = self.make_record(event, accepted=True)
        record["updates"] = {"plugin": prune_empty(payload)}
        self.print_record(record)

    def print_header(self, event: JsonDict) -> None:
        print(f"[{now_string()}] POI EVENT")
        print_kv("seq", event.get("seq"), 2)
        print_kv("event", event.get("event"), 2)
        print_kv("phase", event.get("phase"), 2)
        print_kv("source", event.get("source"), 2)
        print_kv("method", event.get("method"), 2)
        print_kv("path", event.get("path"), 2)
        print_kv("exported_at", ms_to_datetime_string(event.get("exported_at")), 2)

    def print_request_event(self, event: JsonDict) -> None:
        path = event.get("path")
        body = event.get("request_body") or {}

        print("KCSAPI REQUEST")
        print_kv("path", path)

        # Formation selection is usually visible here for battle request.
        formation = body.get("api_formation")
        if formation is not None:
            print_kv("api_formation(選擇陣型)", formation_with_name(formation))

        deck_id = body.get("api_deck_id")
        if deck_id is not None:
            print_kv("api_deck_id", deck_id)

        maparea_id = body.get("api_maparea_id")
        mapinfo_no = body.get("api_mapinfo_no")
        if maparea_id is not None or mapinfo_no is not None:
            print_kv("map", f"{maparea_id}-{mapinfo_no}")

        if not self.config.show_only_battle_summary:
            print("\nREQUEST BODY")
            print(short_json(body, max_len=2000))

    def print_response_event(self, event: JsonDict) -> None:
        path = event.get("path")
        summary = event.get("response_summary") or {}

        print("KCSAPI RESPONSE")
        print_kv("path", path)
        print_kv("api_result", summary.get("api_result"))
        print_kv("api_result_msg", summary.get("api_result_msg"))
        print_kv("has_api_data", summary.get("has_api_data"))

        map_response = extract_map_response(event)
        if map_response:
            self.print_map_summary(map_response)
            self.print_enemy_deck_preview(map_response)

        response_data = extract_response_data(event)
        request_body = event.get("request_body")
        selected_formation = (
            request_body.get("api_formation") if isinstance(request_body, dict) else None
        )
        if selected_formation is None:
            battle_formation = response_data.get("api_formation")
            if isinstance(battle_formation, list) and battle_formation:
                selected_formation = battle_formation[0]
        if selected_formation is not None:
            print("\nFORMATION SELECTION / 我方陣型選擇")
            print_kv(
                "api_formation(選擇陣型)",
                formation_with_name(selected_formation),
            )

        if "api_midnight_flag" in response_data:
            print("\nNIGHT BATTLE AVAILABILITY")
            print_kv(
                "api_midnight_flag(夜戰選擇)",
                explain_midnight_flag(response_data.get("api_midnight_flag")),
            )

        if isinstance(response_data.get("api_ship_ke"), list):
            self.print_enemy_composition(response_data)

        if summary.get("battle"):
            self.print_battle_api_summary(summary["battle"])

        raw_battle_result = summary.get("raw_battle_result")
        if is_battle_result_response(event):
            # Older exporter versions did not create response_summary when poi
            # supplied detail.body as api_data directly. The raw body is authoritative.
            raw_battle_result = extract_response_data(event) or raw_battle_result
        if isinstance(raw_battle_result, dict):
            self.print_raw_battle_result(raw_battle_result)

        if self.config.show_full_body:
            print("\nFULL RESPONSE BODY")
            print(short_json(event.get("response_body"), max_len=6000))
        elif not self.config.show_only_battle_summary:
            keys = summary.get("api_data_keys") or []
            if keys:
                print("\napi_data keys:")
                print(textwrap.fill(", ".join(keys), width=110, subsequent_indent="  "))

    def print_poi_battle_result(self, event: JsonDict) -> None:
        result = event.get("battle_result") or {}

        print("POI NORMALIZED BATTLE RESULT")
        print_kv("valid", result.get("valid"))
        print_kv("rank", result.get("rank"))
        print_kv("boss", result.get("boss"))
        print_kv("map / route no.", f"{result.get('map')} / {result.get('map_cell')}")
        print_kv("quest", result.get("quest"))
        print_kv("enemy", result.get("enemy"))
        print_kv("combined", result.get("combined"))
        print_kv("mvp", result.get("mvp"))
        print_kv(
            "drop_ship",
            format_drop_ship(
                {
                    "ship_id": result.get("drop_ship_id"),
                    "ship_name": result.get("drop_ship_name"),
                    "ship_type": result.get("drop_ship_type"),
                }
            )
            if result.get("drop_ship_id") or result.get("drop_ship_name")
            else "無",
        )
        print_kv("enemy_formation", result.get("enemy_formation"))
        print_kv("battle_time", result.get("time"))

        print("\nPlayer fleet HP from @@BattleResult:")
        deck_ship_id = result.get("deck_ship_id") or []
        deck_hp = result.get("deck_hp") or []
        deck_init_hp = result.get("deck_init_hp") or []

        max_hp_map = self.build_max_hp_map(event.get("fleet_snapshot") or [])

        if not deck_ship_id:
            print("  <empty>")
        else:
            for idx, instance_id in enumerate(deck_ship_id):
                now_hp = deck_hp[idx] if idx < len(deck_hp) else None
                init_hp = deck_init_hp[idx] if idx < len(deck_init_hp) else None
                max_hp = max_hp_map.get(instance_id)
                state = damage_state(now_hp, max_hp)

                print(
                    f"  pos={idx + 1:<2} "
                    f"iid={instance_id:<8} "
                    f"hp={now_hp}/{max_hp} "
                    f"init={init_hp:<4} "
                    f"display_state={state}"
                )

        print("\nEnemy HP:")
        print_kv("enemy_ship_id", result.get("enemy_ship_id"), 2)
        print_kv("enemy_hp", result.get("enemy_hp"), 2)

    def print_day_battle_hp_update(self, event: JsonDict) -> None:
        hp = event.get("day_battle_hp") or {}
        formation = hp.get("api_formation")
        friendly_formation = formation[0] if isinstance(formation, list) and formation else None
        enemy_formation = formation[1] if isinstance(formation, list) and len(formation) > 1 else None
        engagement = formation[2] if isinstance(formation, list) and len(formation) > 2 else None

        print("DAY BATTLE HP UPDATE / 日戰結束血量")
        print_kv("valid", hp.get("valid"))
        print_kv("derived_from(來源事件)", hp.get("derived_from_event"))
        print_kv("derived_from_event_id", hp.get("derived_from_event_id"))
        print_kv("simulator(結算器)", hp.get("simulator"))
        print_kv("api_formation(我方陣形)", formation_with_name(friendly_formation))
        print_kv("api_formation(敵方陣形)", formation_with_name(enemy_formation))
        print_kv("engagement(交戰形態)", value_with_name(engagement, ENGAGEMENT_NAMES))
        print_kv("api_midnight_flag(夜戰選擇)", explain_midnight_flag(hp.get("api_midnight_flag")))

        self.print_calculated_player_hp("PLAYER MAIN / 我方主力", hp.get("player_main"))
        self.print_calculated_player_hp("PLAYER ESCORT / 我方隨伴", hp.get("player_escort"))
        self.print_calculated_enemy_hp("ENEMY MAIN / 敵方主力", hp.get("enemy_main"))
        self.print_calculated_enemy_hp("ENEMY ESCORT / 敵方隨伴", hp.get("enemy_escort"))

    def print_calculated_player_hp(self, title: str, ships: Any) -> None:
        if not isinstance(ships, list) or not ships:
            return
        print(f"\n  {title}:")
        for ship in ships:
            now_hp = ship.get("now_hp")
            max_hp = ship.get("max_hp")
            state = damage_state(now_hp, max_hp)
            used_item = ship.get("used_damage_control")
            item_text = f" damage_control={used_item}" if used_item else ""
            print(
                "    "
                f"pos={ship.get('position'):<2} "
                f"iid={ship.get('instance_id'):<8} "
                f"sid={ship.get('ship_id'):<5} "
                f"hp={now_hp}/{max_hp:<4} "
                f"lost={ship.get('lost_hp'):<4} "
                f"damage={state}{item_text}"
            )

    def print_calculated_enemy_hp(self, title: str, ships: Any) -> None:
        if not isinstance(ships, list) or not ships:
            return
        print(f"\n  {title}:")
        for ship in ships:
            ship_id = ship.get("ship_id")
            now_hp = ship.get("now_hp")
            max_hp = ship.get("max_hp")
            state = damage_state(now_hp, max_hp)
            print(
                "    "
                f"pos={ship.get('position'):<2} "
                f"sid={ship_id:<5} "
                f"name={self.ship_names.get(ship_id, '名稱未載入')} "
                f"hp={now_hp}/{max_hp:<4} "
                f"lost={ship.get('lost_hp'):<4} "
                f"damage={state}"
            )

    def print_map_summary(self, map_summary: JsonDict) -> None:
        event_id = map_summary.get("api_event_id")
        event_kind = map_summary.get("api_event_kind")
        event_name = node_event_name(event_id, event_kind)

        print("\nCURRENT MAP / NODE")
        print_kv("api_maparea_id(海域區域)", f"{map_summary.get('api_maparea_id')} (區域ID)")
        print_kv("api_mapinfo_no(海域編號)", f"{map_summary.get('api_mapinfo_no')} (區域內海域編號)")
        print_kv("api_event_id(節點種類)", f"{event_id} ({event_name})" if event_id is not None else "-")
        print_kv("api_event_kind(事件子類)", event_kind_name(event_id, event_kind))
        print_kv("api_no(目前節點)", f"{map_summary.get('api_no')} (路線節點編號)")
        print_kv("api_color_no(節點色類)", value_with_name(map_summary.get("api_color_no"), NODE_COLOR_NAMES))
        print_kv("api_bosscell_no(王點編號)", explain_boss_cell(map_summary.get("api_bosscell_no")))
        print_kv("api_bosscomp(王點狀態)", explain_uncertain_flag(map_summary.get("api_bosscomp")))
        print_kv(
            "api_production_kind(地圖演出)",
            value_with_name(map_summary.get("api_production_kind"), PRODUCTION_KIND_NAMES),
        )
        print_kv("api_airsearch(航空偵察)", explain_airsearch(map_summary.get("api_airsearch")))
        print_kv("api_eventmap(活動血條)", explain_eventmap(map_summary.get("api_eventmap")))
        print_kv("api_select_route(能動分岐)", explain_select_route(map_summary.get("api_select_route")))
        print_kv("api_limit_state(海域限制)", explain_uncertain_flag(map_summary.get("api_limit_state")))

    def print_enemy_deck_preview(self, data: JsonDict) -> None:
        enemy_decks = data.get("api_e_deck_info")
        if not isinstance(enemy_decks, list) or not enemy_decks:
            return

        print("\nENEMY FLEET PREVIEW / 敵方編成預告")
        for deck_no, deck in enumerate(enemy_decks, 1):
            if not isinstance(deck, dict):
                continue
            print(f"  deck={deck_no} api_kind(編成種類)={deck.get('api_kind')}")
            self.print_enemy_rows(deck.get("api_ship_ids"), indent=4)

    def print_enemy_composition(self, data: JsonDict) -> None:
        formation = data.get("api_formation")
        enemy_formation = formation[1] if isinstance(formation, list) and len(formation) > 1 else None
        engagement = formation[2] if isinstance(formation, list) and len(formation) > 2 else None

        print("\nENEMY FLEET COMPOSITION / 敵方編成")
        print_kv("api_formation(敵方陣形)", value_with_name(enemy_formation, FORMATION_NAMES))
        print_kv("engagement(交戰形態)", value_with_name(engagement, ENGAGEMENT_NAMES))
        print("  Main fleet / 主力艦隊:")
        self.print_enemy_rows(
            data.get("api_ship_ke"),
            data.get("api_ship_lv"),
            data.get("api_e_nowhps"),
            data.get("api_e_maxhps"),
            indent=4,
        )

        combined = data.get("api_ship_ke_combined")
        if isinstance(combined, list) and any(isinstance(ship_id, int) and ship_id > 0 for ship_id in combined):
            print("  Escort fleet / 隨伴艦隊:")
            self.print_enemy_rows(
                combined,
                data.get("api_ship_lv_combined"),
                data.get("api_e_nowhps_combined"),
                data.get("api_e_maxhps_combined"),
                indent=4,
            )

    def print_enemy_rows(
        self,
        ship_ids: Any,
        levels: Any = None,
        now_hps: Any = None,
        max_hps: Any = None,
        indent: int = 2,
    ) -> None:
        if not isinstance(ship_ids, list):
            print(" " * indent + "<empty>")
            return

        displayed = False
        for index, ship_id in enumerate(ship_ids):
            if not isinstance(ship_id, int) or ship_id <= 0:
                continue
            displayed = True
            name = self.ship_names.get(ship_id, "名稱未載入")
            level = levels[index] if isinstance(levels, list) and index < len(levels) else None
            now_hp = now_hps[index] if isinstance(now_hps, list) and index < len(now_hps) else None
            max_hp = max_hps[index] if isinstance(max_hps, list) and index < len(max_hps) else None
            hp = f"{now_hp}/{max_hp}" if now_hp is not None or max_hp is not None else "-"
            level_text = level if level is not None else "-"
            print(
                " " * indent
                + f"pos={index + 1:<2} sid={ship_id:<5} name={name} lv={level_text} hp={hp}"
            )

        if not displayed:
            print(" " * indent + "<empty>")

    def print_battle_api_summary(self, battle: JsonDict) -> None:
        print("\nBATTLE API SUMMARY")
        print_kv("api_deck_id", battle.get("api_deck_id"))
        print_kv("api_formation", battle.get("api_formation"))
        print_kv("api_escape_idx", battle.get("api_escape_idx"))
        print_kv("api_escape_idx_combined", battle.get("api_escape_idx_combined"))

        print("\n  Player HP arrays from raw battle API:")
        print_kv("api_f_nowhps", battle.get("api_f_nowhps"), 4)
        print_kv("api_f_maxhps", battle.get("api_f_maxhps"), 4)
        print_kv("api_f_nowhps_combined", battle.get("api_f_nowhps_combined"), 4)
        print_kv("api_f_maxhps_combined", battle.get("api_f_maxhps_combined"), 4)

        print("\n  Enemy HP arrays from raw battle API:")
        print_kv("api_e_nowhps", battle.get("api_e_nowhps"), 4)
        print_kv("api_e_maxhps", battle.get("api_e_maxhps"), 4)
        print_kv("api_e_nowhps_combined", battle.get("api_e_nowhps_combined"), 4)
        print_kv("api_e_maxhps_combined", battle.get("api_e_maxhps_combined"), 4)

        phases = [
            "has_api_air_base_attack",
            "has_api_kouku",
            "has_api_injection_kouku",
            "has_api_support_info",
            "has_api_opening_taisen",
            "has_api_opening_atack",
            "has_api_hougeki",
            "has_api_hougeki1",
            "has_api_hougeki2",
            "has_api_hougeki3",
            "has_api_raigeki",
        ]
        active = [name for name in phases if battle.get(name)]
        print("\n  Battle phase keys:")
        print("    " + (", ".join(active) if active else "<none>"))

    def print_raw_battle_result(self, raw_result: JsonDict) -> None:
        enemy_info = raw_result.get("api_enemy_info") or {}

        print("\nBATTLE RESULT")
        print_kv("api_win_rank(勝敗評價)", raw_result.get("api_win_rank", "-"))
        print_kv("api_quest_name(交戰海域)", raw_result.get("api_quest_name", "-"))
        print_kv("enemy_deck(敵方艦隊)", enemy_info.get("api_deck_name", "-"))
        print_kv("api_mvp(本隊MVP)", raw_result.get("api_mvp", "-"))
        print_kv("api_mvp_combined(隨伴MVP)", raw_result.get("api_mvp_combined", "-"))
        print_kv("api_dests(擊沉數)", raw_result.get("api_dests", "-"))
        print_kv("api_get_ship(掉落艦娘)", format_drop_ship(raw_result.get("api_get_ship")))

        if raw_result.get("api_get_eventitem") is not None:
            print_kv("api_get_eventitem(活動獎勵)", compact_json(raw_result.get("api_get_eventitem")))
        if raw_result.get("api_get_useitem") is not None:
            print_kv("api_get_useitem(道具獎勵)", compact_json(raw_result.get("api_get_useitem")))

    def print_sortie_snapshot(self, sortie: JsonDict) -> None:
        print("SORTIE SNAPSHOT")
        print_kv("combined_flag", sortie.get("combined_flag"), 2)
        print_kv("sortie_status", sortie.get("sortie_status"), 2)
        print_kv("escaped_pos", sortie.get("escaped_pos"), 2)

    def print_fleet_snapshot(self, fleets: List[JsonDict]) -> None:
        print("\nFIRST FLEET SNAPSHOT")
        if not fleets:
            print("  <empty>")
            return

        fleet = next((item for item in fleets if item.get("deck_id") == 1), fleets[0])
        for ship in fleet.get("ships", []):
            now_hp = ship.get("now_hp")
            max_hp = ship.get("max_hp")
            state = damage_state(now_hp, max_hp)

            print(
                "    "
                f"iid={ship.get('instance_id'):<8} "
                f"sid={ship.get('ship_id'):<5} "
                f"hp={now_hp}/{max_hp:<4} "
                f"damage={state:<23} "
                f"morale={ship.get('cond'):<4} "
                f"ammo={ship.get('bullet'):<4} "
                f"fuel={ship.get('fuel')}"
            )

    def print_unknown_event(self, event: JsonDict) -> None:
        print("UNKNOWN EVENT FORMAT")
        print(short_json(event, max_len=4000))

    @staticmethod
    def build_max_hp_map(fleets: List[JsonDict]) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for fleet in fleets:
            for ship in fleet.get("ships", []):
                iid = ship.get("instance_id")
                max_hp = ship.get("max_hp")
                if isinstance(iid, int) and isinstance(max_hp, int):
                    result[iid] = max_hp
        return result


# -----------------------------
# HTTP server
# -----------------------------

def make_handler(config: ViewerConfig):
    printer = PoiEventPrinter(config)
    delivery_state = load_delivery_state(config.log_path)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in {"/poi-event", "/poi-plugin-loaded"}:
                self.save_receiver_error(
                    "unsupported_path",
                    f"unsupported POST path: {self.path}",
                    self.path,
                )
                self.send_plain_response(404, b"not found")
                return

            event = self.read_json_body()
            if event is None:
                return

            if self.path == "/poi-plugin-loaded":
                plugin_event = {
                    "event": "plugin_loaded",
                    "phase": "receiver",
                    "source": "poi_plugin",
                    "path": self.path,
                }
                record = printer.make_record(plugin_event, accepted=True)
                record["updates"] = {"plugin": prune_empty(event)}
                if not self.save_record(record):
                    self.send_plain_response(500, b"log write failed")
                    return
                self.send_plain_response(200, b"ok")
                return

            if delivery_state.is_duplicate(event):
                record = printer.make_record(
                    event,
                    accepted=False,
                    error={
                        "code": "duplicate",
                        "message": "event_id was already accepted",
                    },
                    include_updates=False,
                )
                self.save_record(record)
                self.send_plain_response(200, b"duplicate already recorded")
                return

            ordering_error = delivery_state.ordering_error(event)
            if ordering_error:
                record = printer.make_record(
                    event,
                    accepted=False,
                    error={
                        "code": "out_of_order",
                        "message": ordering_error,
                    },
                    include_updates=False,
                )
                self.save_record(record)
                self.send_plain_response(409, ordering_error.encode("utf-8"))
                return

            # This event is excluded from JSONL, but its master data lets the
            # console display enemy ship names instead of IDs alone.
            observe_error = printer.observe_event(event)

            if should_ignore_log_event(event):
                delivery_state.record(event)
                self.send_plain_response(200, b"ok (not logged)")
                return

            try:
                record = printer.make_record(
                    event,
                    accepted=True,
                    error=observe_error or packet_error_for_event(event),
                )
            except Exception as exc:
                record = printer.make_record(
                    event,
                    accepted=True,
                    error={
                        "code": "processing_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                    include_updates=False,
                )
            if not self.save_record(record):
                self.send_plain_response(500, b"log write failed")
                return

            delivery_state.record(event)

            self.send_plain_response(200, b"ok")

        def save_record(self, record: JsonDict) -> bool:
            try:
                append_jsonl(config.log_path, record)
            except OSError as exc:
                failed_record = dict(record)
                failed_record["accepted"] = False
                failed_record["error"] = {
                    "code": "log_write_failed",
                    "message": str(exc),
                }
                printer.print_record(failed_record)
                return False

            printer.print_record(record)
            return True

        def save_receiver_error(self, code: str, message: str, path: str) -> None:
            event = {
                "event": "receiver_error",
                "phase": "receiver",
                "source": "battle_receiver",
                "path": path,
            }
            record = printer.make_record(
                event,
                accepted=False,
                error={"code": code, "message": message},
                include_updates=False,
            )
            self.save_record(record)

        def do_OPTIONS(self):
            if self.path not in {"/health", "/poi-event", "/poi-plugin-loaded"}:
                self.send_plain_response(404, b"not found")
                return

            self.send_response(204)
            self.send_cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def read_json_body(self) -> Optional[JsonDict]:
            length_str = self.headers.get("Content-Length", "0")

            try:
                length = int(length_str)
            except ValueError:
                self.save_receiver_error(
                    "invalid_content_length",
                    f"invalid Content-Length: {length_str}",
                    self.path,
                )
                self.send_plain_response(411, b"invalid content-length")
                return None

            raw = self.rfile.read(length)

            try:
                event = json.loads(raw.decode("utf-8"))
                if not isinstance(event, dict):
                    self.save_receiver_error(
                        "invalid_json_type",
                        "JSON payload must be an object",
                        self.path,
                    )
                    self.send_plain_response(400, b"JSON payload must be an object")
                    return None
                return event
            except Exception as exc:
                self.save_receiver_error("bad_json", str(exc), self.path)
                self.send_plain_response(400, f"bad json: {exc}".encode("utf-8"))
                return None

        def do_GET(self):
            if self.path == "/health":
                self.send_plain_response(200, b"ok")
                return

            self.send_plain_response(404, b"not found")

        def send_plain_response(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")

        def log_message(self, fmt, *args):
            # Silence default HTTP access log.
            return

    return Handler


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive and display poi battle event exporter JSON events."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log", type=Path, default=default_log_path())
    parser.add_argument(
        "--formation-history",
        type=Path,
        default=DEFAULT_FORMATION_HISTORY_PATH,
        help="Persistent per-map-node formation history JSON file.",
    )
    parser.add_argument(
        "--full-body",
        action="store_true",
        help="Include the unchanged raw plugin event in each unified log/console record.",
    )
    parser.add_argument(
        "--no-fleet",
        action="store_true",
        help="Do not include player_fleet in unified log/console updates.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Deprecated compatibility flag; unified output is always compact.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    config = ViewerConfig(
        log_path=args.log,
        show_full_body=args.full_body,
        show_fleet=not args.no_fleet,
        show_only_battle_summary=args.summary_only,
        formation_history_path=args.formation_history,
    )

    if config.log_path.exists():
        print(f"Appending to existing log: {config.log_path.resolve()}")
    else:
        print(f"Creating log: {config.log_path.resolve()}")

    handler = make_handler(config)
    server = HTTPServer((args.host, args.port), handler)

    print(f"Listening on http://{args.host}:{args.port}/poi-event")
    print(f"Formation history: {config.formation_history_path.resolve()}")
    print(f"Plugin load confirmation: http://{args.host}:{args.port}/poi-plugin-loaded")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print("Start poi, enable Battle Event Exporter, then run sortie/practice.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
