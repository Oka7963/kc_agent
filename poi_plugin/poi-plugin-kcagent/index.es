// poi-plugin-battle-event-exporter/index.es

import { observe, observer } from 'redux-observers'
import { createSelector } from 'reselect'
import { store } from 'views/create-store'
import { extensionSelectorFactory } from 'views/utils/selectors'
import { Fleet, Simulator } from './lib/battle'

const EXPORT_URL = 'http://127.0.0.1:8765/poi-event'
const PLUGIN_LOADED_URL = 'http://127.0.0.1:8765/poi-plugin-loaded'
const EXTENSION_KEY = 'poi-plugin-battle-event-exporter'

let enabled = false
let sequence = 0
let exportQueue = []
let sending = false
let retryTimer = null
let retryAttempt = 0
let unsubscribeObserve = null
let lastBattleResultDrop = null

const EXPORTER_SESSION_ID = createSessionId()
const INITIAL_RETRY_DELAY_MS = 1000
const MAX_RETRY_DELAY_MS = 30000

export function pluginDidLoad() {
  enabled = true

  window.addEventListener('game.response', onGameResponse)
  postPluginLoaded()
  drainExportQueue()

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
          // The normalized result joins the same FIFO used by game.response.
          postExport(payload)
        }
      },
    ),
  ])

  console.log('[battle-event-exporter] plugin loaded')
}

export function pluginWillUnload() {
  enabled = false

  window.removeEventListener('game.response', onGameResponse)

  if (retryTimer) {
    clearTimeout(retryTimer)
    retryTimer = null
  }

  if (unsubscribeObserve) {
    unsubscribeObserve()
    unsubscribeObserve = null
  }

  console.log('[battle-event-exporter] plugin unloaded')
}

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

const extensionSelector = createSelector(
  extensionSelectorFactory(EXTENSION_KEY),
  (state) => state || {},
)

const battleResultSeqSelector = createSelector(
  extensionSelector,
  (state) => state.battleResultSeq,
)

function onGameResponse(e) {
  if (!enabled) return

  const detail = e.detail || {}
  const path = detail.path || ''

  if (!isExportedResponsePath(path)) return

  if (isBattleResultPath(path)) {
    lastBattleResultDrop = extractDropShip(detail.body)
  }

  const rootStore = store.getState()
  const fleetSnapshot = buildFleetSnapshot(rootStore)

  const sourceEvent = postExport({
    event: classifyPath(path, 'response'),
    phase: 'response',
    source: 'game.response',
    method: detail.method || null,
    path,
    request_body: sanitizeBody(detail.postBody),
    response_summary: summarizeKcsResponse(path, detail.body),
    response_body: sanitizeBody(detail.body),
    fleet_snapshot: fleetSnapshot,
    sortie_snapshot: buildSortieSnapshot(rootStore),
  })

  const hpUpdate = buildDayBattleHpEvent(
    path,
    detail.postBody,
    detail.body,
    rootStore,
    fleetSnapshot,
  )
  if (hpUpdate) {
    hpUpdate.day_battle_hp.derived_from_event_id = sourceEvent && sourceEvent.event_id
    postExport(hpUpdate)
  }
}

/*
 * This uses path prefix instead of pretending poi docs list every battle endpoint.
 * poi docs guarantee we can get path; path classification is our own logic.
 */
function isExportedResponsePath(path) {
  if (!path) return false

  // Paths explicitly handled by poi-plugin-prophet's response listener.
  if (path === '/kcsapi/api_start2/getData') return true
  if (path === '/kcsapi/api_port/port') return true
  if (path === '/kcsapi/api_req_map/start') return true
  if (path === '/kcsapi/api_req_map/next') return true
  if (path === '/kcsapi/api_req_map/air_raid') return true
  if (path === '/kcsapi/api_req_map/start_air_base') return true
  if (path === '/kcsapi/api_req_member/get_practice_enemyinfo') return true

  // Paths handled by Prophet's use-item reducer.
  if (path === '/kcsapi/api_get_member/require_info') return true
  if (path === '/kcsapi/api_get_member/useitem') return true
  if (path === '/kcsapi/api_req_kousyou/remodel_slotlist_detail') return true
  if (path === '/kcsapi/api_req_mission/result') return true

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
  if (path === '/kcsapi/api_start2/getData') return 'game_start_' + phase
  if (path === '/kcsapi/api_port/port') return 'port_' + phase
  if (path === '/kcsapi/api_req_map/start') return 'map_start_' + phase
  if (path === '/kcsapi/api_req_map/next') return 'map_next_' + phase
  if (path === '/kcsapi/api_req_map/air_raid') return 'map_air_raid_' + phase
  if (path === '/kcsapi/api_req_map/start_air_base') return 'map_start_air_base_' + phase
  if (path === '/kcsapi/api_req_member/get_practice_enemyinfo') {
    return 'practice_enemy_info_' + phase
  }

  if (path === '/kcsapi/api_get_member/require_info') return 'require_info_' + phase
  if (path === '/kcsapi/api_get_member/useitem') return 'useitem_update_' + phase
  if (path === '/kcsapi/api_req_kousyou/remodel_slotlist_detail') {
    return 'remodel_slotlist_detail_' + phase
  }
  if (path === '/kcsapi/api_req_mission/result') return 'mission_result_' + phase

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

function isBattleResultPath(path) {
  return path.indexOf('battleresult') >= 0 || path.indexOf('battle_result') >= 0
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
    battle_result: summarizePoiBattleResult(result, lastBattleResultDrop),
    fleet_snapshot: buildFleetSnapshot(rootStore),
    sortie_snapshot: buildSortieSnapshot(rootStore),
  }
}

function summarizePoiBattleResult(result, dropShip) {
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
    drop_ship_id: result.dropShipId || (dropShip && dropShip.ship_id) || null,
    drop_ship_name: (dropShip && dropShip.ship_name) || null,
    drop_ship_type: (dropShip && dropShip.ship_type) || null,
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

function extractDropShip(body) {
  const data = getKcsResponseData(body)
  const ship = data && data.api_get_ship

  if (!ship || typeof ship !== 'object') return null

  return {
    ship_id: ship.api_ship_id,
    ship_type: ship.api_ship_type,
    ship_name: ship.api_ship_name,
  }
}

function summarizeKcsResponse(path, body) {
  const top = normalizeBody(body)
  const data = getKcsResponseData(body)

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

const DAY_BATTLE_PATHS = new Set([
  '/kcsapi/api_req_practice/battle',
  '/kcsapi/api_req_sortie/battle',
  '/kcsapi/api_req_sortie/airbattle',
  '/kcsapi/api_req_sortie/ld_airbattle',
  '/kcsapi/api_req_sortie/ld_shooting',
  '/kcsapi/api_req_combined_battle/battle',
  '/kcsapi/api_req_combined_battle/battle_water',
  '/kcsapi/api_req_combined_battle/airbattle',
  '/kcsapi/api_req_combined_battle/ld_airbattle',
  '/kcsapi/api_req_combined_battle/ld_shooting',
  '/kcsapi/api_req_combined_battle/ec_battle',
  '/kcsapi/api_req_combined_battle/each_battle',
  '/kcsapi/api_req_combined_battle/each_battle_water',
  '/kcsapi/api_req_combined_battle/ec_night_to_day',
])

function buildDayBattleHpEvent(path, postBody, body, rootStore, fleetSnapshot) {
  if (!DAY_BATTLE_PATHS.has(path)) return null

  const data = getKcsResponseData(body)
  if (!data || !Array.isArray(data.api_f_nowhps) || !Array.isArray(data.api_e_nowhps)) {
    return null
  }

  try {
    const info = rootStore && rootStore.info ? rootStore.info : {}
    const sortie = rootStore && rootStore.sortie ? rootStore.sortie : {}
    const fleets = info.fleets || []
    const request = normalizeBody(postBody) || {}
    const deckId = Number(data.api_deck_id || request.api_deck_id || 1)
    const combinedFlag = Number(sortie.combinedFlag || 0)
    const mainDeckIndex = combinedFlag > 0 ? 0 : Math.max(deckId - 1, 0)
    const mainDeck = fleets[mainDeckIndex]
    const escortDeck = combinedFlag > 0 ? fleets[1] : null

    if (!mainDeck) return null

    const simulatorFleet = new Fleet({
      type: combinedFlag,
      main: buildSimulatorDeck(mainDeck, info),
      escort: buildSimulatorDeck(escortDeck, info),
      support: null,
      LBAC: null,
    })
    const simulator = new Simulator(simulatorFleet, { usePoiAPI: false })

    syncSimulatorHp(simulator.mainFleet, data.api_f_nowhps)
    syncSimulatorHp(simulator.escortFleet, data.api_f_nowhps_combined)
    simulator.simulate({ ...data, poi_path: path })

    const mainSnapshot = fleetSnapshot.find((fleet) => fleet.deck_id === mainDeckIndex + 1)
    const escortSnapshot = combinedFlag > 0
      ? fleetSnapshot.find((fleet) => fleet.deck_id === 2)
      : null
    const playerMain = summarizeSimulatedFleet(simulator.mainFleet, mainSnapshot, false)
    const playerEscort = summarizeSimulatedFleet(simulator.escortFleet, escortSnapshot, false)
    const enemyMain = summarizeSimulatedFleet(simulator.enemyFleet, null, true)
    const enemyEscort = summarizeSimulatedFleet(simulator.enemyEscort, null, true)

    return {
      event: 'day_battle_hp_update',
      phase: 'derived',
      source: 'game.response+poi-lib-battle',
      method: null,
      path,
      request_body: sanitizeBody(postBody),
      response_summary: null,
      response_body: null,
      day_battle_hp: {
        valid: true,
        simulator: 'poi-lib-battle@2.20.0',
        derived_from_event: classifyPath(path, 'response'),
        deck_id: deckId,
        combined_flag: combinedFlag,
        api_formation: data.api_formation || null,
        api_midnight_flag: data.api_midnight_flag,
        player_main: playerMain,
        player_escort: playerEscort,
        enemy_main: enemyMain,
        enemy_escort: enemyEscort,
      },
      fleet_snapshot: buildCalculatedFleetSnapshot(
        mainSnapshot,
        escortSnapshot,
        playerMain,
        playerEscort,
      ),
      sortie_snapshot: buildSortieSnapshot(rootStore),
    }
  } catch (error) {
    console.warn('[battle-event-exporter] failed to calculate day battle HP', error)
    return null
  }
}

function buildSimulatorDeck(deck, info) {
  if (!deck || !Array.isArray(deck.api_ship)) return null

  const ships = info.ships || {}
  const equips = info.equips || {}

  return deck.api_ship.map((instanceId) => {
    const ship = ships[instanceId]
    if (!ship || instanceId <= 0) return null

    const toSlot = (slotId) => {
      const equip = equips[slotId]
      return equip ? { api_slotitem_id: equip.api_slotitem_id } : null
    }

    return {
      ...ship,
      poi_slot: (ship.api_slot || []).map(toSlot),
      poi_slot_ex: ship.api_slot_ex > 0 ? [toSlot(ship.api_slot_ex)] : [],
    }
  })
}

function syncSimulatorHp(fleet, nowHps) {
  if (!Array.isArray(fleet) || !Array.isArray(nowHps)) return

  fleet.forEach((ship, index) => {
    if (!ship || typeof nowHps[index] !== 'number') return
    ship.nowHP = nowHps[index]
    ship.initHP = nowHps[index]
  })
}

function summarizeSimulatedFleet(fleet, snapshot, enemy) {
  if (!Array.isArray(fleet)) return []
  const snapshotShips = snapshot && snapshot.ships ? snapshot.ships : []

  return fleet.reduce((result, ship, index) => {
    if (!ship) return result
    const initialHp = ship.initHP
    const rawNowHp = ship.nowHP
    const snapshotShip = snapshotShips.find((item) => item.position === index + 1) || {}

    result.push({
      position: index + 1,
      instance_id: enemy ? null : ((ship.raw && ship.raw.api_id) || snapshotShip.instance_id),
      ship_id: ship.id,
      initial_hp: initialHp,
      now_hp: Math.max(0, rawNowHp),
      raw_now_hp: rawNowHp,
      max_hp: ship.maxHP,
      lost_hp: ship.lostHP,
      used_damage_control: ship.useItem || null,
    })
    return result
  }, [])
}

function buildCalculatedFleetSnapshot(
  mainSnapshot,
  escortSnapshot,
  playerMain,
  playerEscort,
) {
  const update = (snapshot, calculated) => {
    if (!snapshot) return null
    return {
      ...snapshot,
      ships: (snapshot.ships || []).map((ship) => {
        const hp = calculated.find((item) => item.instance_id === ship.instance_id)
          || calculated.find((item) => item.position === ship.position)
        return hp ? { ...ship, now_hp: hp.now_hp } : ship
      }),
    }
  }

  return [
    update(mainSnapshot, playerMain),
    update(escortSnapshot, playerEscort),
  ].filter(Boolean)
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

function getKcsResponseData(body) {
  const top = normalizeBody(body)

  if (!top || typeof top !== 'object') return null
  if (Object.prototype.hasOwnProperty.call(top, 'api_data')) {
    return top.api_data
  }

  // poi may emit game.response.detail.body as api_data directly.
  return top
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
  if (!enabled) return null

  sequence += 1

  const finalPayload = {
    ...payload,
    session_id: EXPORTER_SESSION_ID,
    seq: sequence,
    exported_at: Date.now(),
  }

  finalPayload.event_id = EXPORTER_SESSION_ID + ':' + sequence
  exportQueue.push(finalPayload)
  drainExportQueue()
  return finalPayload
}

function postPluginLoaded() {
  postJsonOnce(PLUGIN_LOADED_URL, {
    plugin: EXTENSION_KEY,
    message: 'Poi loaded Battle Event Exporter',
    session_id: EXPORTER_SESSION_ID,
    loaded_at: Date.now(),
  })
}

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') {
    return window.crypto.randomUUID()
  }

  return (
    Date.now().toString(36) +
    '-' +
    Math.random().toString(36).slice(2) +
    Math.random().toString(36).slice(2)
  )
}

function drainExportQueue() {
  if (!enabled || sending || retryTimer || exportQueue.length === 0) return

  const payload = exportQueue[0]
  sending = true

  try {
    fetch(EXPORT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error('HTTP ' + res.status + ' ' + res.statusText)
        }

        // Remove only the acknowledged queue head. No later event can be sent first.
        if (exportQueue[0] === payload) {
          exportQueue.shift()
        }

        sending = false
        retryAttempt = 0
        drainExportQueue()
      })
      .catch((err) => {
        handleExportFailure(payload, err)
      })
  } catch (err) {
    handleExportFailure(payload, err)
  }
}

function handleExportFailure(payload, err) {
  sending = false
  retryAttempt += 1

  const delay = Math.min(
    INITIAL_RETRY_DELAY_MS * Math.pow(2, Math.min(retryAttempt - 1, 5)),
    MAX_RETRY_DELAY_MS,
  )

  console.warn(
    '[battle-event-exporter] export failed; retrying queue head:',
    payload.event_id,
    'in',
    delay,
    'ms',
    err,
  )

  if (!enabled || retryTimer) return

  retryTimer = setTimeout(() => {
    retryTimer = null
    drainExportQueue()
  }, delay)
}

function postJsonOnce(url, payload) {
  try {
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    }).catch((err) => {
      console.warn('[battle-event-exporter] one-shot POST failed:', err)
    })
  } catch (err) {
    console.warn('[battle-event-exporter] one-shot fetch failed:', err)
  }
}
