# KC Agent Event Contract

This document defines the responsibility split and the event fields used by the
agent state machine. Fields marked **Key condition** are used to choose a route
or decide whether the agent may proceed.

## Responsibility split

| Module | Responsibility | Must not do |
| --- | --- | --- |
| `poi_plugin/poi-plugin-kcagent/index.es` | Export raw Poi `game.request`, `game.response`, and `@@BattleResult` data. | Decide agent state, click UI, or evaluate safety policy. |
| `battle_receiver/battle_receiver.py` | HTTP receive/log/response wrapper. | Decode business rules directly; it delegates decoding to `kc_core.decoder`. |
| `kc_core/decoder.py` | Decode one raw event into one normalized `AgentEvent` using `config/decoder_rules.json`. | Keep runtime state, wait for scenes, or execute commands. |
| `kc_agent.py` | Consume normalized events and own the workflow state machine. | Parse raw Poi API structures directly. |
| `key_identify/` and `scene_identify/` | Detect scenes and targets. | Decide sortie/battle policy. |
| `apply_action/` | Execute mouse movement/click actions. | Decide whether an action is safe. |

## Raw Poi event

Raw events are JSON objects posted by the Poi plugin to `/poi-event`.

| Field | Meaning | Key condition |
| --- | --- | --- |
| `event` | Plugin-classified event name, e.g. `map_start_request`, `map_next_response`, `sortie_battle_response`, `battle_result`. | **Yes**. Main decoder route selector. |
| `phase` | `request`, `response`, or `poi_battle_result`. | **Yes**. Distinguishes request vs response and identifies Poi battle result events. |
| `path` | KanColle API path, e.g. `/kcsapi/api_req_map/start`. | **Yes**. Used as fallback route selector and `source_kind` derivation. |
| `request_body` | Raw request data from Poi. | **Yes** for map/deck/formation values. |
| `response_body` | Raw response data from Poi. | **Yes** for map node, battle, and night-battle conditions. |
| `battle_result` | Poi `@@BattleResult` summary. | **Yes** for rank/drop/damage extraction. |
| `fleet_snapshot` | Poi store fleet state at export time. | **Yes** for active fleet and damage safety analysis. |
| `sortie_snapshot` | Poi store sortie state at export time. | No; included as context. |
| `seq` / `exported_at` / `source` | Trace metadata. | No; used for logging/correlation. |

## Normalized envelope: `AgentEvent`

Every decoded event uses this common envelope from `kc_core.event_models.AgentEvent`.

| Field | Meaning | Key condition |
| --- | --- | --- |
| `type` | State-machine event type, e.g. `map_node_arrived`. | **Yes**. Dispatch key in `KcAgent.handle_event`. |
| `payload` | Event-specific normalized data. | **Yes**. Holds route conditions and scene requirements. |
| `event_id` | Generated correlation ID. | No; used for tracing. |
| `ts_ms` | Event timestamp in milliseconds. | No. |
| `source` | Producer name, usually `poi`. | No. |
| `correlation` | Optional state/scene correlation data. | Sometimes; scene flow uses `next_scenes` and `next_scene_index`. |

## Normalized payload common fields

| Field | Meaning | Key condition |
| --- | --- | --- |
| `schema` | Payload schema version. Current value: `kc.agent.battle_event.v1`. | No. |
| `event_type` | Same business event type as envelope `type`. | **Yes** for validation/debugging. |
| `phase` | Normalized phase such as `request`, `map`, `formation`, `battle_day`, `result`. | Sometimes, for debugging and future guards. |
| `path` | Original raw API path. | Sometimes, for trace/debugging. |
| `expected.user_input_expected` | Whether UI input is expected after this event. | **Yes**. Agent uses this to decide waiting behavior conceptually. |
| `expected.ui_wait_reason` | Human-readable reason for the next wait. | No; used for logs. |
| `expected.next_scenes` | Ordered scene requirements. | **Yes**. Drives scene wait order and target requirements. |

## Event types and conditions

### `sortie_start_requested`

Created when `event == map_start_request`, or when `path == /kcsapi/api_req_map/start` and `phase == request`.

**Purpose:** initialize sortie context.

**Key payload fields:**

| Field | Meaning | Key condition |
| --- | --- | --- |
| `map.world` | Derived map ID like `1-1`. | No; context. |
| `fleet.deck_id` | Active deck ID. | No; context. |
| `expected.next_scenes` | Always empty. | **Yes**; no UI scene is expected yet. |

### `map_node_arrived`

Created from `map_start_response` or `map_next_response`.

**Purpose:** classify the node and decide which UI scenes should be observed next.

**Key payload fields:**

| Field | Meaning | Key condition |
| --- | --- | --- |
| `map.rashin_flg` | Compass/map-production flag. | **Yes**. `1` adds `compass_or_map_production` to `next_scenes`. |
| `map.is_battle_node` | Derived by node-classification rules. | **Yes**. `true` adds `formation_select` to `next_scenes`. |
| `map.is_boss_node` | Derived by route/color/event rules. | **Yes** for future boss policies; currently context. |
| `map.node_type` | Derived node category, e.g. `battle`, `boss_battle`, `resource`. | **Yes** for future route policies; currently context. |
| `expected.next_scenes` | Ordered waits after map arrival. | **Yes**. Agent waits the first scene, then continues through the list. |

### `formation_selected`

Created when raw `event` ends with `_battle_request`.

**Purpose:** mark that the user selected a formation and battle started.

**Key payload fields:**

| Field | Meaning | Key condition |
| --- | --- | --- |
| `battle.formation.requested` | Requested formation ID from raw request. | No; context/logging. |
| `expected.next_scenes` | Always empty. | **Yes**; agent waits for later battle/result events. |

### `day_battle_received`

Created from `sortie_battle_response`.

**Purpose:** describe battle start and determine whether night battle choice is expected.

**Key payload fields:**

| Field | Meaning | Key condition |
| --- | --- | --- |
| `battle.can_night_battle` | Derived from `response_body.api_midnight_flag == 1`. | **Yes**. `true` adds `night_battle_choice` scene. |
| `expected.next_scenes` | Contains `night_battle_choice` only when night battle is available. | **Yes**. |

### `battle_result`

Created from raw `event == battle_result` or `phase == poi_battle_result`.

**Purpose:** summarize battle result, evaluate damage safety, and decide result/advance UI requirements.

**Key payload fields:**

| Field | Meaning | Key condition |
| --- | --- | --- |
| `damage.has_taiha` | Whether any active fleet ship is at taiha threshold. | **Yes**. Forces retreat-only targets. |
| `damage.taiha_ships` | List of ships that triggered safety latch. | **Yes**. Used in logs/safety payload. |
| `expected.next_scenes[0]` | `battle_result_confirm`. | **Yes**. Agent clicks confirm after scene is ready. |
| `expected.next_scenes[1]` | `advance_or_retreat`. Required targets are retreat-only when taiha exists. | **Yes**. Agent auto-retreats if `taiha_latch` is active. |

## Dynamic decoder configuration

`config/decoder_rules.json` owns route constants, node classification values, and scene target/timeouts. Change the JSON when a route condition or target requirement changes; avoid hard-coding these dynamic arguments in the receiver or agent.
