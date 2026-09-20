// One speaker shared by search announcements and chat. All audio comes from
// the existing OMNI endpoint; there is no browser text-to-speech substitute.
import { useSyncExternalStore } from 'react'

type SpeakerState = { enabled: boolean; busy: boolean; error: string | null; text: string }
let state: SpeakerState = { enabled: false, busy: false, error: null, text: '' }
const listeners = new Set<() => void>()
const SPEECH_REQUEST_TIMEOUT_MS = 45_000
let context: AudioContext | null = null
let source: AudioBufferSourceNode | null = null
let request: AbortController | null = null
let generation = 0

function update(next: Partial<SpeakerState>) {
  state = { ...state, ...next }
  listeners.forEach((listener) => listener())
}

export function stopOmniSpeech() {
  generation += 1
  request?.abort()
  request = null
  if (source) {
    source.onended = null
    source.stop()
    source.disconnect()
    source = null
  }
  update({ busy: false })
}

export function speakOmni(text: string) {
  stopOmniSpeech()
  const phrase = text.trim()
  update({ text: phrase, error: null })
  // Keep the latest relevant reply until the user enables playback.
  if (!state.enabled || !phrase) return
  const token = generation
  const abort = new AbortController()
  request = abort
  update({ busy: true })
  void (async () => {
    const deadline = setTimeout(() => abort.abort(), SPEECH_REQUEST_TIMEOUT_MS)
    try {
      const response = await fetch('/api/speech/synthesize', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: phrase }), signal: abort.signal,
      })
      if (!response.ok) throw new Error('OMNI voice is unavailable. Check the connection and try again.')
      const data: { audio_url: string } = await response.json()
      const audio = await fetch(data.audio_url, { signal: abort.signal })
      if (!audio.ok) throw new Error('The voice recording could not be loaded.')
      if (!context) throw new Error('Enable voice to hear Patch.')
      const buffer = await context.decodeAudioData(await audio.arrayBuffer())
      if (token !== generation) return
      if (context.state !== 'running') throw new Error('Playback was paused by the browser. Click Retry voice.')
      source = context.createBufferSource()
      source.buffer = buffer
      source.connect(context.destination)
      source.onended = () => {
        if (token !== generation) return
        source?.disconnect()
        source = null
        update({ busy: false })
      }
      source.start()
    } catch (error) {
      if (token !== generation) return
      update({ busy: false, error: error instanceof Error ? error.message : 'Voice playback failed. Try again.' })
    } finally {
      clearTimeout(deadline)
    }
  })()
}

export async function enableOmniSpeech() {
  // Called directly by a click: unlock browser audio before the cloud request.
  try {
    context ??= new AudioContext()
    await context.resume()
    update({ enabled: true, error: null })
    speakOmni(state.text || "Hi, I'm Patch. My voice is ready.")
  } catch {
    update({ error: 'Audio could not start. Check your browser sound permissions.' })
  }
}

export function muteOmniSpeech() {
  stopOmniSpeech()
  update({ enabled: false, error: null })
}

export function useOmniSpeaker() {
  return useSyncExternalStore((listener) => {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }, getOmniSpeakerState)
}

export function getOmniSpeakerState() { return state }
