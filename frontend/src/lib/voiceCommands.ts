// Spoken answers to "Is this your <item>?".
//
// Accepting a candidate makes the rover drive, so a transcript is only ever
// turned into an action through a STRICT allow-list matched against the whole
// utterance. Anything else does nothing. In particular free text is never
// treated as a new target here, and references like "the other one" are not
// resolvable from a transcript alone -- we say so instead of guessing.

export type VoiceCommand = 'accept' | 'reject' | 'stop'

export type VoiceParseResult =
  | { kind: 'command'; command: VoiceCommand }
  | { kind: 'unresolved_reference' }
  | { kind: 'unrecognized' }

export const VOICE_HINT_UNRECOGNIZED = 'Say yes or no'
export const VOICE_HINT_UNRESOLVED_REFERENCE =
  "I can't tell which one you mean from that. Say yes or no about the item shown."

// Stored without apostrophes: normalizeTranscript() strips them, so "that's
// it", "thats it" and "that’s it" all land on the same entry.
const ACCEPT_PHRASES = new Set(['yes', 'yeah', 'yep', 'correct', 'thats it'])
const REJECT_PHRASES = new Set(['no', 'nope', 'not that one', 'wrong'])

// "the other one", "another one", "that one", "the one next to it", bare "it"…
const REFERENCE_PATTERN = /\b(other|another|different|next|same|that|this)\s+one\b|\bthe one\b|^(it|this|that|these|those|them)$/

export function normalizeTranscript(transcript: string): string {
  return transcript
    .toLowerCase()
    .replace(/['’‘`]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function parseVoiceCommand(transcript: string): VoiceParseResult {
  const text = normalizeTranscript(transcript)
  if (!text) return { kind: 'unrecognized' }

  // "stop" is the one deliberately loose match: a false stop leaves the
  // wheels still and the search resumable, so the word anywhere in the
  // utterance wins over everything else ("no stop", "stop stop", "please stop").
  if (/\bstop\b/.test(text)) return { kind: 'command', command: 'stop' }

  if (ACCEPT_PHRASES.has(text)) return { kind: 'command', command: 'accept' }
  if (REJECT_PHRASES.has(text)) return { kind: 'command', command: 'reject' }
  if (REFERENCE_PATTERN.test(text)) return { kind: 'unresolved_reference' }
  return { kind: 'unrecognized' }
}
