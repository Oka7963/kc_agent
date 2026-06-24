import unittest

from kc_core.damage import damage_state
from kc_core.decoder import PoiEventDecoder


class PoiEventDecoderTest(unittest.TestCase):
    def setUp(self):
        self.decoder = PoiEventDecoder()

    def test_map_node_arrived_adds_compass_and_formation_scenes(self):
        event = self.decoder.decode({
            "event": "map_start_response",
            "phase": "response",
            "path": "/kcsapi/api_req_map/start",
            "request_body": {"api_deck_id": "1"},
            "response_body": {
                "api_maparea_id": 1,
                "api_mapinfo_no": 1,
                "api_no": 2,
                "api_from_no": 0,
                "api_color_no": 4,
                "api_event_id": 4,
                "api_event_kind": 1,
                "api_rashin_flg": 1,
            },
        })

        self.assertIsNotNone(event)
        self.assertEqual(event.type, "map_node_arrived")
        self.assertTrue(event.payload["map"]["is_battle_node"])
        self.assertEqual(
            [scene["scene"] for scene in event.payload["expected"]["next_scenes"]],
            ["compass_or_map_production", "formation_select"],
        )

    def test_day_battle_response_adds_night_scene_only_when_flag_is_one(self):
        event = self.decoder.decode({
            "event": "sortie_battle_response",
            "phase": "response",
            "path": "/kcsapi/api_req_sortie/battle",
            "request_body": {"api_formation": 1},
            "response_body": {"api_midnight_flag": 1, "api_formation": [1, 2, 3]},
        })

        self.assertIsNotNone(event)
        self.assertEqual(event.type, "day_battle_received")
        self.assertTrue(event.payload["battle"]["can_night_battle"])
        self.assertEqual(event.payload["expected"]["next_scenes"][0]["scene"], "night_battle_choice")

    def test_battle_result_uses_retreat_only_targets_when_taiha(self):
        event = self.decoder.decode({
            "event": "battle_result",
            "phase": "poi_battle_result",
            "battle_result": {
                "deckShipId": [101],
                "deckHp": [5],
                "deckInitHp": [20],
                "rank": "A",
            },
            "fleet_snapshot": [{
                "deck_id": 1,
                "ships": [{"position": 1, "instance_id": 101, "ship_id": 1001, "now_hp": 20, "max_hp": 20}],
            }],
        })

        self.assertIsNotNone(event)
        self.assertEqual(event.type, "battle_result")
        self.assertTrue(event.payload["damage"]["has_taiha"])
        self.assertEqual(event.payload["expected"]["next_scenes"][1]["required_targets"], ["retreat_button"])

    def test_damage_thresholds_are_pure_and_testable(self):
        self.assertEqual(damage_state(5, 20), "taiha")
        self.assertEqual(damage_state(10, 20), "chuuha")
        self.assertEqual(damage_state(19, 20), "shouha_or_scratch")
        self.assertEqual(damage_state(20, 20), "normal")

    def test_decoder_config_has_target_button_placeholders(self):
        buttons = self.decoder.config["target_buttons"]
        for name in [
            "rashin_confirm_button",
            "formation_line_ahead_button",
            "no_night_battle_button",
            "result_confirm_button",
            "drop_confirm_button",
            "advance_button",
            "retreat_button",
        ]:
            self.assertIn(name, buttons)
            self.assertTrue(buttons[name]["template_file"].endswith(".png"))
            self.assertEqual(
                set(buttons[name]["coordinates"]),
                {"client_xywh", "screen_xywh", "roi_xywh"},
            )


if __name__ == "__main__":
    unittest.main()
