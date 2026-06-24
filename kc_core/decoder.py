"""Config-driven POI raw-event decoder.

Decoder responsibility is intentionally narrow:
1. read dynamic routing/scene arguments from JSON config;
2. decode a raw POI/exporter event into one normalized AgentEvent;
3. keep all outputs deterministic and unit-testable.

It does not run the receiver server, mutate agent state, click UI elements, or
wait for scenes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kc_core.damage import analyze_damage_from_fleet
from kc_core.event_models import AgentEvent, JsonDict, SCHEMA_VERSION, SceneRequirement, new_id, now_ms, to_int

DEFAULT_DECODER_CONFIG = Path(__file__).resolve().parents[1] / "config" / "decoder_rules.json"


@dataclass(frozen=True)
class NodeClassification:
    node_type: str
    is_battle: bool
    is_boss: bool


class PoiEventDecoder:
    """Decode raw POI exporter payloads into normalized agent events."""

    def __init__(self, config_path: str | Path = DEFAULT_DECODER_CONFIG) -> None:
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as f:
            self.config: JsonDict = json.load(f)
        self.event_routes: JsonDict = self.config.get("event_routes", {})
        self.scene_defaults: JsonDict = self.config.get("scene_defaults", {})
        self.node_rules: JsonDict = self.config.get("node_classification", {})

    def decode(self, raw: JsonDict) -> Optional[AgentEvent]:
        """Decode one raw event. Return None for unsupported events."""
        if "event_type" in raw:
            return self._already_normalized(raw)

        name = raw.get("event")
        phase = raw.get("phase")

        if self._matches_sortie_start(raw):
            return self._decode_sortie_start(raw)
        if name in self.event_routes["map_node_arrived"].get("match_events", []):
            return self._decode_map_node(raw)
        if name and name.endswith(self.event_routes["formation_selected"].get("event_suffix", "_battle_request")):
            return self._decode_formation_selected(raw)
        if name in self.event_routes["day_battle_received"].get("match_events", []):
            return self._decode_day_battle(raw)
        battle_route = self.event_routes["battle_result"]
        if name in battle_route.get("match_events", []) or phase == battle_route.get("match_phase"):
            return self._decode_battle_result(raw)
        return None

    def classify_node(self, body: JsonDict) -> NodeClassification:
        route_no = to_int(body.get("api_no"))
        color_no = to_int(body.get("api_color_no"))
        event_id = to_int(body.get("api_event_id"))
        event_kind = to_int(body.get("api_event_kind"))
        boss_cell_no = to_int(body.get("api_bosscell_no"))

        boss_rules = self.node_rules.get("boss", {})
        battle_rules = self.node_rules.get("battle", {})
        is_boss = (
            route_no is not None and boss_cell_no is not None and route_no == boss_cell_no
        ) or color_no in boss_rules.get("color_no", []) or event_id in boss_rules.get("event_id", [])
        is_battle = (
            is_boss
            or color_no in battle_rules.get("color_no", [])
            or event_kind in battle_rules.get("event_kind", [])
            or bool(body.get("api_e_deck_info"))
        )

        if is_boss:
            return NodeClassification("boss_battle", True, True)
        if is_battle:
            return NodeClassification("battle", True, False)

        by_color = self.node_rules.get("non_battle_by_color", {})
        return NodeClassification(by_color.get(str(color_no), "unknown"), False, False)

    def scene(self, key: str) -> JsonDict:
        value = dict(self.scene_defaults[key])
        scene_name = value.pop("scene", key)
        return SceneRequirement(scene=scene_name, **value).to_dict()

    def active_fleet_from_snapshot(self, raw: JsonDict, deck_id: int = 1, include_slot: bool = False) -> JsonDict:
        fleets = raw.get("fleet_snapshot") or []
        selected = next((f for f in fleets if to_int(f.get("deck_id")) == deck_id), None)
        if selected is None and fleets:
            selected = fleets[0]
        if not selected:
            return {"deck_id": deck_id, "ship_count": 0, "ships": []}

        ships = []
        for ship in selected.get("ships") or []:
            item = {
                "pos": ship.get("position") or ship.get("pos"),
                "instance_id": ship.get("instance_id"),
                "ship_id": ship.get("ship_id"),
                "level": ship.get("level"),
                "now_hp": ship.get("now_hp"),
                "max_hp": ship.get("max_hp"),
                "cond": ship.get("cond"),
                "fuel": ship.get("fuel"),
                "bullet": ship.get("bullet"),
            }
            if include_slot:
                item["slot"] = ship.get("slot") or []
            ships.append(item)

        sortie = raw.get("sortie_snapshot") or {}
        return {
            "deck_id": selected.get("deck_id", deck_id),
            "combined_flag": sortie.get("combined_flag"),
            "sortie_status": sortie.get("sortie_status") or [],
            "escaped_pos": sortie.get("escaped_pos") or [],
            "ship_count": selected.get("ship_count", len(ships)),
            "ships": ships,
        }

    def _matches_sortie_start(self, raw: JsonDict) -> bool:
        route = self.event_routes["sortie_start_requested"].get("match", {})
        return raw.get("event") == route.get("event") or (
            raw.get("path") == route.get("path") and raw.get("phase") == route.get("phase")
        )

    def _already_normalized(self, raw: JsonDict) -> AgentEvent:
        return AgentEvent(
            type=raw["event_type"],
            payload=raw,
            event_id=raw.get("event_id", new_id("poi")),
            ts_ms=to_int(raw.get("ts_ms"), now_ms()) or now_ms(),
            source=raw.get("source", "poi"),
            correlation=raw.get("correlation") or {},
        )

    def _base_payload(self, raw: JsonDict, event_type: str, phase: str) -> JsonDict:
        return {
            "schema": SCHEMA_VERSION,
            "event_type": event_type,
            "phase": phase,
            "seq": raw.get("seq"),
            "source_event": raw.get("source"),
            "path": raw.get("path"),
        }

    def _agent_event(self, event_type: str, payload: JsonDict, raw: JsonDict) -> AgentEvent:
        return AgentEvent(event_type, payload, new_id("poi"), to_int(raw.get("exported_at"), now_ms()) or now_ms(), "poi")

    def _decode_sortie_start(self, raw: JsonDict) -> AgentEvent:
        req = raw.get("request_body") or {}
        area = to_int(req.get("api_maparea_id"))
        info_no = to_int(req.get("api_mapinfo_no"))
        deck_id = to_int(req.get("api_deck_id"), 1) or 1
        payload = self._base_payload(raw, "sortie_start_requested", "request")
        payload.update({
            "map": {"area": area, "info_no": info_no, "world": f"{area}-{info_no}" if area and info_no else None},
            "fleet": self.active_fleet_from_snapshot(raw, deck_id, include_slot=True),
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": False, "ui_wait_reason": "waiting_map_start_response", "next_scenes": []},
        })
        return self._agent_event("sortie_start_requested", payload, raw)

    def _decode_map_node(self, raw: JsonDict) -> AgentEvent:
        body = raw.get("response_body") or {}
        req = raw.get("request_body") or {}
        area = to_int(body.get("api_maparea_id"))
        info_no = to_int(body.get("api_mapinfo_no"))
        node = self.classify_node(body)
        next_scenes = []
        if to_int(body.get("api_rashin_flg")) == 1:
            next_scenes.append(self.scene("compass_or_map_production"))
        if node.is_battle:
            next_scenes.append(self.scene("formation_select"))
        source_kind = self.event_routes["map_node_arrived"].get("source_kind_by_path", {}).get(raw.get("path"), "map_next")
        payload = self._base_payload(raw, "map_node_arrived", "map")
        payload.update({
            "source_kind": source_kind,
            "map": {
                "area": area,
                "info_no": info_no,
                "world": f"{area}-{info_no}" if area and info_no else None,
                "route_no": to_int(body.get("api_no")),
                "from_route_no": to_int(body.get("api_from_no")),
                "is_first_node_after_sortie_start": raw.get("path") == "/kcsapi/api_req_map/start" and to_int(body.get("api_from_no")) == 0,
                "color_no": to_int(body.get("api_color_no")),
                "event_id": to_int(body.get("api_event_id")),
                "event_kind": to_int(body.get("api_event_kind")),
                "node_type": node.node_type,
                "is_battle_node": node.is_battle,
                "is_boss_node": node.is_boss,
                "boss_cell_no": to_int(body.get("api_bosscell_no")),
                "next": to_int(body.get("api_next")),
                "rashin_flg": to_int(body.get("api_rashin_flg")),
                "rashin_id": to_int(body.get("api_rashin_id")),
            },
            "fleet": self.active_fleet_from_snapshot(raw, to_int(req.get("api_deck_id"), 1) or 1, include_slot=True),
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": bool(next_scenes), "ui_wait_reason": "sortie_first_node_arrived" if source_kind == "map_start" else "advanced_to_next_node", "next_scenes": next_scenes},
        })
        return self._agent_event("map_node_arrived", payload, raw)

    def _decode_formation_selected(self, raw: JsonDict) -> AgentEvent:
        req = raw.get("request_body") or {}
        payload = self._base_payload(raw, "formation_selected", "formation")
        payload.update({
            "battle": {"kind": "normal_day", "formation": {"requested": to_int(req.get("api_formation")), "friendly": None, "enemy": None, "engagement": None}},
            "fleet": self.active_fleet_from_snapshot(raw, 1, include_slot=True),
            "expected": {"user_input_expected": False, "ui_wait_reason": "battle_started", "next_scenes": []},
        })
        return self._agent_event("formation_selected", payload, raw)

    def _decode_day_battle(self, raw: JsonDict) -> AgentEvent:
        body = raw.get("response_body") or {}
        req = raw.get("request_body") or {}
        formation = body.get("api_formation") or []
        can_night = to_int(body.get("api_midnight_flag")) == 1
        next_scenes = [self.scene("night_battle_choice")] if can_night else []
        payload = self._base_payload(raw, "day_battle_received", "battle_day")
        payload.update({
            "battle": {
                "kind": "normal_day",
                "deck_id": to_int(body.get("api_deck_id")),
                "formation": {"requested": to_int(req.get("api_formation")), "friendly": to_int(formation[0]) if len(formation) > 0 else None, "enemy": to_int(formation[1]) if len(formation) > 1 else None, "engagement": to_int(formation[2]) if len(formation) > 2 else None},
                "can_night_battle": can_night,
                "phases": {"kouku": bool(body.get("api_kouku")), "support": bool(body.get("api_support_info")), "opening_taisen": bool(body.get("api_opening_taisen")), "opening_torpedo": bool(body.get("api_opening_atack")), "hougeki1": bool(body.get("api_hougeki1")), "hougeki2": bool(body.get("api_hougeki2")), "hougeki3": bool(body.get("api_hougeki3")), "raigeki": bool(body.get("api_raigeki"))},
            },
            "fleet": self.active_fleet_from_snapshot(raw, 1, include_slot=True),
            "enemy": {"ship_ids": body.get("api_ship_ke") or [], "start_hp": body.get("api_e_nowhps") or [], "max_hp": body.get("api_e_maxhps") or []},
            "expected": {"user_input_expected": can_night, "ui_wait_reason": "can_night_battle" if can_night else "waiting_battle_result", "next_scenes": next_scenes},
        })
        return self._agent_event("day_battle_received", payload, raw)

    def _decode_battle_result(self, raw: JsonDict) -> AgentEvent:
        result = raw.get("battle_result") or raw.get("result") or {}
        fleet = self.active_fleet_from_snapshot(raw, 1, include_slot=True)
        hp_by_iid: dict[int, JsonDict] = {}
        deck_ship_ids = result.get("deck_ship_id") or result.get("deckShipId") or []
        deck_hp = result.get("deck_hp") or result.get("deckHp") or []
        deck_init_hp = result.get("deck_init_hp") or result.get("deckInitHp") or []
        for idx, iid in enumerate(deck_ship_ids):
            iid_int = to_int(iid)
            if iid_int is not None:
                hp_by_iid[iid_int] = {"battle_end_hp": to_int(deck_hp[idx]) if idx < len(deck_hp) else None, "battle_start_hp": to_int(deck_init_hp[idx]) if idx < len(deck_init_hp) else None}
        for ship in fleet.get("ships", []):
            iid = to_int(ship.get("instance_id"))
            if iid in hp_by_iid:
                ship.update(hp_by_iid[iid])
        damage = analyze_damage_from_fleet(fleet)
        is_boss_result = bool(result.get("boss"))
        has_drop = bool(
            result.get("drop_ship_id")
            or result.get("dropShipId")
            or result.get("drop_item")
            or result.get("dropItem")
            or result.get("event_item")
            or result.get("eventItem")
        )
        next_scenes = [self.scene("battle_result_confirm"), self.scene("post_battle_next")]
        if has_drop:
            next_scenes.append(self.scene("drop_confirm"))
        if not is_boss_result:
            next_scenes.append(self.scene("retreat_only" if damage["has_taiha"] else "advance_or_retreat"))
        payload = self._base_payload(raw, "battle_result", "result")
        payload.update({
            "battle": {"rank": result.get("rank"), "boss": result.get("boss"), "map": result.get("map"), "map_cell": result.get("map_cell") or result.get("mapCell"), "enemy_name": result.get("enemy"), "mvp": result.get("mvp") or [], "drop": {"ship_id": result.get("drop_ship_id") or result.get("dropShipId"), "item": result.get("drop_item") or result.get("dropItem"), "event_item": result.get("event_item") or result.get("eventItem")}},
            "fleet": fleet,
            "damage": damage,
            "sortie": raw.get("sortie_snapshot") or {},
            "expected": {"user_input_expected": True, "ui_wait_reason": "boss_battle_result_end" if is_boss_result else "battle_result_confirm_then_advance_or_retreat", "next_scenes": next_scenes},
        })
        return self._agent_event("battle_result", payload, raw)


_default_decoder: PoiEventDecoder | None = None


def normalize_poi_raw_event(raw: JsonDict, config_path: str | Path = DEFAULT_DECODER_CONFIG) -> Optional[AgentEvent]:
    """Compatibility wrapper used by legacy entry points."""
    global _default_decoder
    path = Path(config_path)
    if _default_decoder is None or _default_decoder.config_path != path:
        _default_decoder = PoiEventDecoder(path)
    return _default_decoder.decode(raw)
