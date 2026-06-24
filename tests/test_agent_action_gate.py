import asyncio
import unittest

from kc_agent import AgentState, KcAgent
from kc_core.event_models import AgentEvent


def scene_ready(scene, target, wait_id=None):
    return AgentEvent(
        "scene_ready",
        {
            "scene": scene,
            "wait_id": wait_id,
            "targets": {
                target: {
                    "visible": True,
                    "clickable": True,
                    "bbox_screen_xywh": [10, 20, 30, 40],
                    "confidence": 0.99,
                }
            },
        },
        source="test",
    )


class KcAgentActionGateTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.event_q = asyncio.Queue()
        self.scene_wait_q = asyncio.Queue()
        self.command_q = asyncio.Queue()
        self.agent = KcAgent(self.event_q, self.scene_wait_q, self.command_q)

    async def test_scene_only_rule_clicks_battle_result_confirm_without_event(self):
        await self.agent.handle_event(scene_ready("battle_result_confirm", "result_confirm_button"))

        self.assertEqual(self.command_q.qsize(), 1)
        cmd = await self.command_q.get()
        self.assertEqual(cmd.target, "result_confirm_button")
        self.assertEqual(cmd.requires_scene, "battle_result_confirm")
        self.assertEqual(cmd.safety["trigger_mode"], "scene_only")
        self.assertEqual(cmd.target_observation["bbox_screen_xywh"], [10, 20, 30, 40])
        self.assertEqual(self.agent.ctx.state, AgentState.WAIT_ACTION_RESULT)

    async def test_scene_only_rules_cover_normal_battle_flow_buttons(self):
        cases = [
            ("compass_or_map_production", "rashin_confirm_button"),
            ("formation_select", "formation_line_ahead_button"),
            ("night_battle_choice", "no_night_battle_button"),
            ("drop_confirm", "drop_confirm_button"),
        ]

        for scene, target in cases:
            with self.subTest(scene=scene, target=target):
                self.agent.ctx.pending_command_id = None
                self.agent.ctx.fired_action_keys.clear()
                await self.agent.handle_event(scene_ready(scene, target))
                cmd = await self.command_q.get()
                self.assertEqual(cmd.target, target)
                self.assertEqual(cmd.requires_scene, scene)
                self.assertEqual(cmd.safety["trigger_mode"], "scene_only")

    async def test_event_and_scene_rule_does_not_click_retreat_without_taiha_event(self):
        await self.agent.handle_event(scene_ready("advance_or_retreat", "retreat_button"))

        self.assertEqual(self.command_q.qsize(), 0)
        self.assertEqual(self.agent.ctx.latest_scene.scene, "advance_or_retreat")

    async def test_event_and_scene_rule_clicks_retreat_when_latest_scene_matches_taiha_event(self):
        await self.agent.handle_event(scene_ready("advance_or_retreat", "retreat_button"))
        battle_result = AgentEvent(
            "battle_result",
            {
                "battle": {"rank": "A"},
                "damage": {"has_taiha": True, "taiha_ships": [{"pos": 1}]},
                "expected": {
                    "next_scenes": [
                        {"scene": "battle_result_confirm", "required_targets": ["result_confirm_button"], "timeout_ms": 8000},
                        {"scene": "advance_or_retreat", "required_targets": ["retreat_button"], "timeout_ms": 10000},
                    ]
                },
            },
            event_id="evt-battle-result",
            source="test",
        )

        await self.agent.handle_event(battle_result)

        self.assertEqual(self.command_q.qsize(), 1)
        cmd = await self.command_q.get()
        self.assertEqual(cmd.target, "retreat_button")
        self.assertEqual(cmd.trigger_event_id, "evt-battle-result")
        self.assertEqual(cmd.requires_scene, "advance_or_retreat")
        self.assertEqual(cmd.safety["trigger_mode"], "event_and_scene")
        self.assertTrue(cmd.safety["taiha_latch"])
        self.assertEqual(cmd.safety["forbid_target"], "advance_button")


if __name__ == "__main__":
    unittest.main()
