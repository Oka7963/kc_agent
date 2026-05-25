// poi-plugin-battle-event-exporter/index.es

import { observe, observer } from 'redux-observers'
import { createSelector } from 'reselect'
import { store } from 'views/create-store'
import { extensionSelectorFactory } from 'views/utils/selectors'

const EXPORT_URL = 'http://127.0.0.1:8765/poi-event'
const PLUGIN_LOADED_URL = 'http://127.0.0.1:8765/poi-plugin-loaded'
const EXTENSION_KEY = 'poi-plugin-battle-event-exporter'

let enabled = false
let sequence = 0
let unsubscribeObserve = null

export function pluginDidLoad() {
  enabled = true

  window.addEventListener('game.request', onGameRequest)
  window.addEventListener('game.response', onGameResponse)
  postPluginLoaded()

  /*
   * Use observer for side effect.
   * reducer only updates plugin state.
   */
  unsubscribeObserve = observe(store, [
    observer(
      battleResultSeqSelector,
      (dispatch, current, previous) => {
        if (!enabled) return
        if (current == null || current === previous) return

        const state = store.getState()
        const ext = extensionSelector(state)
        const payload = ext && ext.lastBattleResultEvent

        if (payload) {
          postExport(payload)
        }
      },
    ),
  ])

  console.log('[battle-event-exporter] plugin loaded')
}

export function pluginWillUnload() {
  enabled = false

  window.removeEventListener('game.request', onGameRequest)
  window.removeEventListener('game.response', onGameResponse)

  if (unsubscribeObserve) {
    unsubscribeObserve()
    unsubscribeObserve = null
  }

  console.log('[battle-event-exporter] plugin unloaded')
}

/*
 * Reducer should only update plugin state.
 * poi docs say reducer's third argument is the whole Redux store.
 */
export const reducer = (state = { battleResultSeq: 0 }, action, rootStore) => {
  if (!action || action.type !== '@@BattleResult') {
    return state
  }

  const nextSeq = (state.battleResultSeq || 0) + 1

  return {
    ...state,
    battleResultSeq: nextSeq,
    lastBattleResultEvent: buildBattleResultEvent(action.result, rootStore),
  }
}

/*
 * Selectors for observer.
 */
const extensionSelector = createSelector(
  extensionSelectorFactory(EXTENSION_KEY),
  (state) => state || {},
)

const battleResultSeqSelector = createSelector(
  extensionSelector,
  (state) => state.battleResultSeq,
)

/*
 * game.request and game.response are explicitly documented by poi.
 */
function onGameRequest(e) {
  if (!enabled) return

  const detail = e.detail || {}
  const path = detail.path || ''

  if (!isBattleFlowPath(path)) return

  postExport({
    event: classifyPath(path, 'request'),
    phase: 'request',
    source: 'game.request',
    method: detail.method || null,
    path,
    request_body: sanitizeBody(detail.body),
    response_summary: null,
    response_body: null,
    fleet_snapshot: buildFleetSnapshotFromStore(),
    sortie_snapshot: buildSortieSnapshotFromStore(),
  })
}

function onGameResponse(e) {
  if (!enabled) return

  const detail = e.detail || {}
  const path = detail.path || ''

  if (!isBattleFlowPath(path)) return

  postExport({
    event: classifyPath(path, 'response'),
    phase: 'response',
    source: 'game.response',
    method: detail.method || null,
    path,
    request_body: sanitizeBody(detail.postBody),
    response_summary: summarizeKcsResponse(path, detail.body),
    response_body: sanitizeBody(detail.body),
    fleet_snapshot: buildFleetSnapshotFromStore(),
    sortie_snapshot: buildSortieSnapshotFromStore(),
  })
}

/*
 * This uses path prefix instead of pretending poi docs list every battle endpoint.
 * poi docs guarantee we can get path; path classification is our own logic.
 */
function isBattleFlowPath(path) {
  if (!path) return false

  if (path === '/kcsapi/api_req_map/start') return true
  if (path === '/kcsapi/api_req_map/next') return true

  if (path.indexOf('/kcsapi/api_req_sortie/') === 0) return true
  if (path.indexOf('/kcsapi/api_req_battle_midnight/') === 0) return true
  if (path.indexOf('/kcsapi/api_req_combined_battle/') === 0) return true
  if (path.indexOf('/kcsapi/api_req_practice/') === 0) return true

  /*
   * Not battle API itself, but useful after battle because ship/deck state may update.
   */
  if (path === '/kcsapi/api_get_member/ship2') return true
  if (path === '/kcsapi/api_get_member/ship3') return true
  if (path === '/kcsapi/api_get_member/deck') return true

  return false
}

function classifyPath(path, phase) {
  if (path === '/kcsapi/api_req_map/start') return 'map_start_' + phase
  if (path === '/kcsapi/api_req_map/next') return 'map_next_' + phase

  if (path === '/kcsapi/api_get_member/ship2') return 'ship2_update_' + phase
  if (path === '/kcsapi/api_get_member/ship3') return 'ship3_update_' + phase
  if (path === '/kcsapi/api_get_member/deck') return 'deck_update_' + phase

  if (path.indexOf('/kcsapi/api_req_practice/') === 0) {
    if (path.indexOf('battle_result') >= 0) return 'practice_battle_result_' + phase
    if (path.indexOf('midnight') >= 0) return 'practice_midnight_battle_' + phase
    if (path.indexOf('battle') >= 0) return 'practice_battle_' + phase
    return 'practice_api_' + phase
  }

  if (path.indexOf('/kcsapi/api_req_combined_battle/') === 0) {
    if (path.indexOf('battleresult') >= 0) return 'combined_battle_result_' + phase
    if (path.indexOf('goback_port') >= 0) return 'combined_goback_port_' + phase
    if (path.indexOf('midnight') >= 0 || path.indexOf('sp_midnight') >= 0) {
      return 'combined_midnight_battle_' + phase
    }
    if (path.indexOf('airbattle') >= 0 || path.indexOf('ld_airbattle') >= 0) {
      return 'combined_air_battle_' + phase
    }
    if (path.indexOf('each_battle') >= 0) return 'combined_each_battle_' + phase
    if (path.indexOf('battle_water') >= 0) return 'combined_battle_water_' + phase
    if (path.indexOf('battle') >= 0) return 'combined_battle_' + phase
    return 'combined_battle_api_' + phase
  }

  if (path.indexOf('/kcsapi/api_req_battle_midnight/') === 0) {
    if (path.indexOf('sp_midnight') >= 0) return 'special_midnight_battle_' + phase
    return 'midnight_battle_' + phase
  }

  if (path.indexOf('/kcsapi/api_req_sortie/') === 0) {
    if (path.indexOf('battleresult') >= 0) return 'sortie_battle_result_' + phase
    if (path.indexOf('night_to_day') >= 0) return 'night_to_day_' + phase
    if (path.indexOf('airbattle') >= 0 || path.indexOf('ld_airbattle') >= 0) {
      return 'sortie_air_battle_' + phase
    }
    if (path.indexOf('ld_shooting') >= 0) return 'sortie_ld_shooting_' + phase
    if (path.indexOf('battle') >= 0) return 'sortie_battle_' + phase
    return 'sortie_api_' + phase
  }

  return 'battle_flow_' + phase
}

function buildBattleResultEvent(result, rootStore) {
  return {
    event: 'battle_result',
    phase: 'poi_battle_result',
    source: '@@BattleResult',
    method: null,
    path: null,
    request_body: null,
    response_summary: null,
    response_body: null,
    battle_result: summarizePoiBattleResult(result),
    fleet_snapshot: buildFleetSnapshot(rootStore),
    sortie_snapshot: buildSortieSnapshot(rootStore),
  }
}

function summarizePoiBattleResult(result) {
  if (!result) return null

  return {
    valid: result.valid,
    rank: result.rank,
    boss: result.boss,
    map: result.map,
    map_cell: result.mapCell,
    quest: result.quest,
    enemy: result.enemy,
    combined: result.combined,
    mvp: result.mvp || [],
    drop_item: result.dropItem || null,
    drop_ship_id: result.dropShipId || null,
    deck_ship_id: result.deckShipId || [],
    deck_hp: result.deckHp || [],
    deck_init_hp: result.deckInitHp || [],
    enemy_ship_id: result.enemyShipId || [],
    enemy_formation: result.enemyFormation,
    enemy_hp: result.enemyHp || [],
    event_item: result.eventItem || null,
    time: result.time,
  }
}

function summarizeKcsResponse(path, body) {
  const top = normalizeBody(body)
  const data = top && top.api_data ? top.api_data : null

  const summary = {
    api_result: top ? top.api_result : undefined,
    api_result_msg: top ? top.api_result_msg : undefined,
    has_api_data: !!data,
    api_data_keys: data ? Object.keys(data) : [],
  }

  if (!data) return summary

  if (path === '/kcsapi/api_req_map/start' || path === '/kcsapi/api_req_map/next') {
    summary.map = {
      api_rashin_flg: data.api_rashin_flg,
      api_rashin_id: data.api_rashin_id,
      api_maparea_id: data.api_maparea_id,
      api_mapinfo_no: data.api_mapinfo_no,
      api_no: data.api_no,
      api_color_no: data.api_color_no,
      api_event_id: data.api_event_id,
      api_event_kind: data.api_event_kind,
      api_next: data.api_next,
      api_bosscell_no: data.api_bosscell_no,
      api_bosscomp: data.api_bosscomp,
    }
  }

  if (isBattleApiPath(path)) {
    summary.battle = {
      api_deck_id: data.api_deck_id,
      api_formation: data.api_formation || null,

      api_f_nowhps: data.api_f_nowhps || null,
      api_f_maxhps: data.api_f_maxhps || null,
      api_f_nowhps_combined: data.api_f_nowhps_combined || null,
      api_f_maxhps_combined: data.api_f_maxhps_combined || null,

      api_e_nowhps: data.api_e_nowhps || null,
      api_e_maxhps: data.api_e_maxhps || null,
      api_e_nowhps_combined: data.api_e_nowhps_combined || null,
      api_e_maxhps_combined: data.api_e_maxhps_combined || null,

      api_ship_ke: data.api_ship_ke || null,
      api_ship_ke_combined: data.api_ship_ke_combined || null,

      api_midnight_flag: data.api_midnight_flag,
      api_escape_idx: data.api_escape_idx || null,
      api_escape_idx_combined: data.api_escape_idx_combined || null,

      has_api_kouku: !!data.api_kouku,
      has_api_injection_kouku: !!data.api_injection_kouku,
      has_api_air_base_attack: !!data.api_air_base_attack,
      has_api_support_info: !!data.api_support_info,
      has_api_opening_taisen: !!data.api_opening_taisen,
      has_api_opening_atack: !!data.api_opening_atack,
      has_api_hougeki: !!data.api_hougeki,
      has_api_hougeki1: !!data.api_hougeki1,
      has_api_hougeki2: !!data.api_hougeki2,
      has_api_hougeki3: !!data.api_hougeki3,
      has_api_raigeki: !!data.api_raigeki,
    }
  }

  if (path.indexOf('battleresult') >= 0 || path.indexOf('battle_result') >= 0) {
    summary.raw_battle_result = {
      api_win_rank: data.api_win_rank,
      api_quest_name: data.api_quest_name,
      api_quest_level: data.api_quest_level,
      api_enemy_info: data.api_enemy_info || null,
      api_get_ship: data.api_get_ship || null,
      api_get_eventitem: data.api_get_eventitem || null,
      api_get_useitem: data.api_get_useitem || null,
      api_mvp: data.api_mvp,
      api_mvp_combined: data.api_mvp_combined,
    }
  }

  return summary
}

function isBattleApiPath(path) {
  return (
    path.indexOf('/kcsapi/api_req_sortie/') === 0 ||
    path.indexOf('/kcsapi/api_req_battle_midnight/') === 0 ||
    path.indexOf('/kcsapi/api_req_combined_battle/') === 0 ||
    path.indexOf('/kcsapi/api_req_practice/') === 0
  )
}

function buildFleetSnapshotFromStore() {
  return buildFleetSnapshot(store.getState())
}

function buildSortieSnapshotFromStore() {
  return buildSortieSnapshot(store.getState())
}

function buildFleetSnapshot(rootStore) {
  const info = rootStore && rootStore.info ? rootStore.info : {}
  const fleets = info.fleets || []
  const ships = info.ships || {}

  const result = []

  for (let fleetIndex = 0; fleetIndex < fleets.length; fleetIndex += 1) {
    const fleet = fleets[fleetIndex]
    if (!fleet || !fleet.api_ship) continue

    const shipList = fleet.api_ship
      .filter((id) => id > 0)
      .map((id, index) => {
        const ship = ships[id] || {}

        return {
          position: index + 1,
          instance_id: id,
          ship_id: ship.api_ship_id,
          level: ship.api_lv,
          now_hp: ship.api_nowhp,
          max_hp: ship.api_maxhp,
          cond: ship.api_cond,
          fuel: ship.api_fuel,
          bullet: ship.api_bull,
          slot: ship.api_slot,
        }
      })

    result.push({
      deck_id: fleetIndex + 1,
      fleet_index: fleetIndex,
      name: fleet.api_name,
      mission: fleet.api_mission || null,
      ship_count: shipList.length,
      ships: shipList,
    })
  }

  return result
}

function buildSortieSnapshot(rootStore) {
  const sortie = rootStore && rootStore.sortie ? rootStore.sortie : {}

  return {
    combined_flag: sortie.combinedFlag,
    sortie_status: sortie.sortieStatus || [],
    escaped_pos: sortie.escapedPos || [],
  }
}

function normalizeBody(body) {
  if (!body) return null

  if (typeof body === 'string') {
    try {
      const cleaned = body.replace(/^svdata=/, '')
      return JSON.parse(cleaned)
    } catch (e) {
      return null
    }
  }

  return body
}

function sanitizeBody(body) {
  const normalized = normalizeBody(body)

  if (!normalized || typeof normalized !== 'object') {
    return normalized || null
  }

  return removeSensitiveKeys(normalized)
}

function removeSensitiveKeys(obj) {
  if (Array.isArray(obj)) {
    return obj.map(removeSensitiveKeys)
  }

  if (!obj || typeof obj !== 'object') {
    return obj
  }

  const out = {}

  Object.keys(obj).forEach((key) => {
    if (key === 'api_token') return
    if (key === 'api_token_raw') return
    out[key] = removeSensitiveKeys(obj[key])
  })

  return out
}

function postExport(payload) {
  if (!enabled) return

  sequence += 1

  const finalPayload = {
    ...payload,
    seq: sequence,
    exported_at: Date.now(),
  }

  postJson(EXPORT_URL, finalPayload)
}

function postPluginLoaded() {
  postJson(PLUGIN_LOADED_URL, {
    plugin: EXTENSION_KEY,
    message: 'Poi loaded Battle Event Exporter',
    loaded_at: Date.now(),
  })
}

function postJson(url, payload) {
  try {
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((res) => {
        if (!res.ok) {
          console.warn('[battle-event-exporter] export failed:', res.status, res.statusText)
        }
      })
      .catch((err) => {
        console.warn('[battle-event-exporter] export error:', err)
      })
  } catch (err) {
    console.warn('[battle-event-exporter] fetch failed:', err)
  }
}
