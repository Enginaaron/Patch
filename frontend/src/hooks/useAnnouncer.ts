import { useCallback, useRef } from 'react'

const STORAGE_KEY = 'patch.announced'
// Plenty for a session; bounded so the list cannot grow without limit.
const MAX_REMEMBERED_KEYS = 200

// sessionStorage can throw (private mode, storage disabled). Remembering then
// degrades to the in-memory ref, which still covers refetches and StrictMode.
function loadSpokenKeys(): Set<string> {
  try {
    const parsed: unknown = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? '[]')
    if (Array.isArray(parsed)) return new Set(parsed.filter((k): k is string => typeof k === 'string'))
  } catch {
    /* fall through to an empty set */
  }
  return new Set()
}

function saveSpokenKeys(keys: Set<string>): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...keys].slice(-MAX_REMEMBERED_KEYS)))
  } catch {
    /* in-memory dedupe still applies */
  }
}

interface UseAnnouncerResult {
  // Speaks `text` unless `key` was already announced. Returns whether it spoke.
  announce: (key: string, text: string) => boolean
}

// Speak-once announcements. The page state is refetched constantly, effects
// run twice under StrictMode and the user may refresh mid-search; none of
// that may make Patch repeat itself. A key is remembered BEFORE speech starts
// (ref for this mount, sessionStorage across reloads), so the second pass of a
// double effect already sees it. Replacing audio that is still playing is
// speak()'s job (useVoice does that); speech failures are silent -- the same
// text is always on screen.
function useAnnouncer(speak: (text: string) => void): UseAnnouncerResult {
  const spokenRef = useRef<Set<string> | null>(null)

  const announce = useCallback(
    (key: string, text: string): boolean => {
      const sentence = text.trim()
      if (!key || !sentence) return false

      // Loaded on first use rather than during render: reading storage is a
      // side effect, and announce() is only ever called from effects/handlers.
      if (spokenRef.current === null) spokenRef.current = loadSpokenKeys()
      const spoken = spokenRef.current
      if (spoken.has(key)) return false

      spoken.add(key)
      saveSpokenKeys(spoken)
      try {
        speak(sentence)
      } catch {
        /* silent by design */
      }
      return true
    },
    [speak],
  )

  return { announce }
}

export default useAnnouncer
