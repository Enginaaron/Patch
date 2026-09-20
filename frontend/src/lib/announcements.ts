import type { SearchDetail } from './searchTypes'

// Deciding what Patch says out loud, as pure functions of the search detail.
// The page refetches on every SSE message, so the same detail is seen many
// times; the KEY is what makes an announcement happen once (useAnnouncer
// remembers spoken keys). Choosing the key therefore decides how chatty the
// rover is:
//   - a candidate question is keyed by candidate id -> once per candidate;
//   - "on my way" is keyed by the accepted candidate -> once, however often
//     the approach flips between checking / centering / approaching;
//   - resting outcomes are keyed by movement.seq + phase -> once per outcome.
// Scanning/checking flips are never announced.

export interface Announcement {
  key: string
  text: string
}

export function candidateKey(searchId: string, candidateId: string): string {
  return `${searchId}:candidate:${candidateId}`
}

export function acceptedKey(searchId: string, candidateId: string): string {
  return `${searchId}:accepted:${candidateId}`
}

export function movementKey(searchId: string, seq: number, phase: string): string {
  return `${searchId}:${seq}:${phase}`
}

// Used only when the backend sent no message for the phase.
const RESTING_FALLBACK_TEXT: Record<string, string> = {
  arrived: "I'm next to it.",
  target_lost: "I've lost sight of it, so I've stopped.",
  exhausted: "I looked all around and couldn't find it.",
  error: 'Something went wrong, so I have stopped.',
  stopped: 'Stopped.',
}

// Stops the user asked for from a page that is going away (cancel) or that
// belong to a search another one just replaced: nothing useful to say.
const SILENT_STOP_REASONS = new Set(['cancelled', 'replaced'])

export function candidateQuestion(detail: SearchDetail): string {
  const { movement } = detail
  if (movement.phase === 'waiting_for_confirmation' && movement.message.trim()) return movement.message.trim()
  return `Is this "${detail.target_text}"?`
}

export function pickAnnouncement(detail: SearchDetail): Announcement | null {
  const { search_id: searchId, status, movement } = detail
  if (status === 'CANCELLED') return null

  if (status === 'CANDIDATE_PENDING' && detail.pending_candidate) {
    return {
      key: candidateKey(searchId, detail.pending_candidate.candidate_id),
      text: candidateQuestion(detail),
    }
  }

  const fallback = RESTING_FALLBACK_TEXT[movement.phase]
  if (fallback !== undefined) {
    if (movement.phase === 'stopped' && movement.reason !== null && SILENT_STOP_REASONS.has(movement.reason)) {
      return null
    }
    return {
      key: movementKey(searchId, movement.seq, movement.phase),
      text: movement.message.trim() || fallback,
    }
  }

  // Accepted and the rover has taken the job on. Worded as intent, not as a
  // result: whether it gets there is only ever reported by phase 'arrived'.
  if (status === 'FOUND' && detail.accepted_candidate && movement.active) {
    return {
      key: acceptedKey(searchId, detail.accepted_candidate.candidate_id),
      text: "Okay. I'll try to make my way over to it.",
    }
  }

  return null
}
