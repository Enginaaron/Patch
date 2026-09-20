// Shapes of the search/rover API as the frontend consumes them, plus the few
// pure helpers that interpret them. Kept free of React so the rules that
// matter (what is terminal, what an error body means) are easy to read.

export type Box2d = [number, number, number, number] // [ymin, xmin, ymax, xmax], 0-1000 scale

// Recognition lifecycle (database). FOUND means "the user confirmed Patch
// identified the item" -- it never means the rover reached it. Arrival is only
// ever movement.phase === 'arrived'.
export type SearchStatus = 'SEARCHING' | 'CANDIDATE_PENDING' | 'FOUND' | 'CANCELLED'

// What the rover body is doing. Deliberately separate from SearchStatus.
export type MovementPhase =
  | 'idle'
  | 'scanning'
  | 'checking'
  | 'waiting_for_confirmation'
  | 'centering'
  | 'approaching'
  | 'arrived'
  | 'target_lost'
  | 'stopped'
  | 'exhausted'
  | 'error'

export interface MovementState {
  search_id: string | null
  phase: MovementPhase
  message: string
  reason: string | null
  active: boolean // a mission currently owns the wheels (includes waiting_for_confirmation)
  seq: number // bumps on every published change
  updated_at: string
  scan_steps: number
  scan_budget: number
  approach_pulses: number
  candidates_presented: number
  target_offset: number | null
  target_width: number | null
  category: string | null
  vision_mode: string // 'live' | 'simulation'
  driver: string // 'sim' | 'gpio'
}

export interface ReferenceImageOut {
  id: string
  url: string
}

export interface PendingCandidate {
  candidate_id: string
  description: string
  crop_url: string
  image_url: string
  box: Box2d | null // null = Patch saw it but has no location, so it will never drive toward it
  created_at: string
}

export interface AcceptedCandidate {
  candidate_id: string
  description: string
  crop_url: string
}

export interface ResolvedTargetOut {
  category: string | null
  landmarks: string[]
  resolver: string
}

export interface SearchDetail {
  search_id: string
  item_id: string
  target_text: string
  status: SearchStatus
  started_at: string
  ended_at: string | null
  reference_images: ReferenceImageOut[]
  pending_candidate: PendingCandidate | null
  accepted_candidate: AcceptedCandidate | null
  resolved_target: ResolvedTargetOut | null
  movement: MovementState
  can_resume: boolean
  terminal: boolean
}

// What actually arrives over the wire: the fields this page cannot work
// without are required, everything added by the rover work is optional so an
// older/partial backend degrades instead of crashing the page (a crashed page
// has no STOP button).
type RawSearchDetail = Pick<SearchDetail, 'search_id' | 'target_text' | 'status'> &
  Partial<Omit<SearchDetail, 'movement'>> & { movement?: Partial<MovementState> | null }

const IDLE_MOVEMENT: MovementState = {
  search_id: null,
  phase: 'idle',
  message: '',
  reason: null,
  active: false,
  seq: 0,
  updated_at: '',
  scan_steps: 0,
  scan_budget: 0,
  approach_pulses: 0,
  candidates_presented: 0,
  target_offset: null,
  target_width: null,
  category: null,
  vision_mode: 'live',
  driver: 'sim',
}

const SEARCH_STATUSES: readonly string[] = ['SEARCHING', 'CANDIDATE_PENDING', 'FOUND', 'CANCELLED']

// Returns null for a body that is not a search detail at all (e.g. a proxy's
// HTML error page parsed as JSON null); callers treat that like a failed fetch.
export function normalizeSearchDetail(body: unknown): SearchDetail | null {
  if (typeof body !== 'object' || body === null) return null
  const candidate = body as Partial<RawSearchDetail>
  if (
    typeof candidate.search_id !== 'string' ||
    typeof candidate.target_text !== 'string' ||
    typeof candidate.status !== 'string' ||
    !SEARCH_STATUSES.includes(candidate.status)
  ) {
    return null
  }
  const raw = candidate as RawSearchDetail

  const movement: MovementState = raw.movement
    ? { ...IDLE_MOVEMENT, ...raw.movement }
    : {
        ...IDLE_MOVEMENT,
        search_id: raw.search_id,
        // No movement block at all: we cannot prove the rover is at rest, so
        // assume an unfinished search may be moving. That keeps STOP on screen.
        active: raw.status === 'SEARCHING' || raw.status === 'CANDIDATE_PENDING',
      }

  return {
    search_id: raw.search_id,
    item_id: raw.item_id ?? '',
    target_text: raw.target_text,
    status: raw.status,
    started_at: raw.started_at ?? '',
    ended_at: raw.ended_at ?? null,
    reference_images: raw.reference_images ?? [],
    pending_candidate: raw.pending_candidate ?? null,
    accepted_candidate: raw.accepted_candidate ?? null,
    resolved_target: raw.resolved_target ?? null,
    movement,
    can_resume: raw.can_resume ?? false,
    terminal: raw.terminal ?? (raw.status === 'CANCELLED' || movement.phase === 'arrived'),
  }
}

// SSE messages are `{type, payload, terminal?}`. The payload is never merged
// into state (state comes from GET /api/searches/{id} alone); the only thing
// read from a message is whether the stream is finished.
export function isTerminalMessage(data: unknown): boolean {
  if (typeof data !== 'string') return false
  try {
    const parsed: unknown = JSON.parse(data)
    return typeof parsed === 'object' && parsed !== null && (parsed as { terminal?: unknown }).terminal === true
  } catch {
    return false
  }
}

export interface ApiErrorInfo {
  code: string | null
  message: string | null
  fields: Record<string, unknown>
}

// FastAPI error bodies come in three flavours: `{detail: "text"}`,
// `{detail: {code, message, ...}}` (ours) and `{detail: [...]}` (request
// validation). Flatten them so callers never render "[object Object]".
export function readApiError(body: unknown): ApiErrorInfo {
  const empty: ApiErrorInfo = { code: null, message: null, fields: {} }
  if (typeof body !== 'object' || body === null) return empty
  const detail = (body as { detail?: unknown }).detail
  if (typeof detail === 'string') return { ...empty, message: detail }
  if (typeof detail !== 'object' || detail === null || Array.isArray(detail)) return empty
  const fields = detail as Record<string, unknown>
  return {
    code: typeof fields.code === 'string' ? fields.code : null,
    message: typeof fields.message === 'string' ? fields.message : null,
    fields,
  }
}

const PHASE_LABEL: Record<string, string> = {
  idle: 'Idle',
  scanning: 'Scanning',
  checking: 'Checking the view',
  waiting_for_confirmation: 'Waiting for your answer',
  centering: 'Lining up',
  approaching: 'Approaching',
  arrived: 'Arrived',
  target_lost: 'Lost sight of it',
  stopped: 'Stopped',
  exhausted: 'Looked everywhere',
  error: 'Problem',
}

export function phaseLabel(phase: string): string {
  return PHASE_LABEL[phase] ?? phase.replaceAll('_', ' ')
}

// Why the rover came to rest, in plain words. Worded to claim no more than
// the backend knows: arrival is a camera-based estimate (apparent size of the
// item in the frame), never a measured distance.
const REASON_LABEL: Record<string, string> = {
  proximity_confirmed: 'Looks close enough (camera estimate, not a measured distance)',
  not_visible: "Couldn't see the item any more",
  identity_uncertain: "Couldn't be sure it was still the same item",
  no_box: "Saw the item but couldn't locate it, so didn't drive",
  scan_budget: 'Finished a full look around',
  candidate_budget: 'Too many wrong matches',
  user_stop: 'Stopped by you',
  cancelled: 'Search cancelled',
  manual_override: 'Manual driving took over',
  replaced: 'Replaced by a newer search',
  shutdown: 'Patch is shutting down',
  approach_budget: 'Approach limit reached',
  no_progress: "Wasn't getting any closer",
  interrupted: 'Interrupted by a restart',
  stale_frames: 'Camera stopped delivering frames',
  inference_failed: 'Vision service kept failing',
  invalid_detection: 'Detection was unusable',
  unsafe_vision: 'Refused: simulated vision with real motors',
  internal_error: 'Internal error',
}

export function reasonLabel(reason: string): string {
  return REASON_LABEL[reason] ?? reason.replaceAll('_', ' ')
}

// Emergency stop. `keepalive` lets the request outlive the page if the user
// navigates away right after pressing it. If the rover endpoint cannot be
// reached or refuses, fall back to the manual-drive stop, which also zeroes
// the wheels and invalidates the mission. Resolves false only when neither
// endpoint confirmed the stop.
export async function postStop(): Promise<boolean> {
  const attempt = (url: string) =>
    fetch(url, { method: 'POST', keepalive: true }).then(
      (res) => res.ok,
      () => false,
    )
  if (await attempt('/api/rover/stop')) return true
  return attempt('/api/drive/stop')
}
