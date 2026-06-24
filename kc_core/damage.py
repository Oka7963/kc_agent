"""Pure damage and fleet helpers used by decoders and tests."""

from __future__ import annotations

from typing import Optional

from kc_core.event_models import JsonDict, to_int


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
