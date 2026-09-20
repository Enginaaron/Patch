import { useCallback, useRef, useState } from 'react'

// Safety cap so a stuck recording can't run forever if the user forgets to
// release the mic button. Real silence-based auto-stop (VAD) is a later
// refinement -- release-to-stop is the primary path for now.
const MAX_RECORDING_MS = 10_000

interface UseVoiceOptions {
  onTranscript?: (transcript: string) => void
}

interface UseVoiceResult {
  isRecording: boolean
  isTranscribing: boolean
  isSpeaking: boolean
  startRecording: () => Promise<void>
  stopRecording: () => void
  speak: (text: string) => void
}

function useVoice({ onTranscript }: UseVoiceOptions = {}): UseVoiceResult {
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const maxDurationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const stopRecording = useCallback(() => {
    if (maxDurationTimerRef.current) {
      clearTimeout(maxDurationTimerRef.current)
      maxDurationTimerRef.current = null
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    setIsRecording(false)
  }, [])

  const startRecording = useCallback(async () => {
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (err) {
      // Spec 21: mic permission denied / no device -- fail silently back to
      // the tap-based flow rather than leaving the caller stuck. isRecording
      // never flips true, so the mic button just looks like it did nothing.
      console.warn('microphone unavailable', err)
      return
    }
    streamRef.current = stream

    const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : ''
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
    chunksRef.current = []

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data)
    }

    recorder.onstop = async () => {
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null

      const usedMimeType = recorder.mimeType || 'audio/webm'
      const blob = new Blob(chunksRef.current, { type: usedMimeType })
      const ext = usedMimeType.includes('ogg') ? 'ogg' : 'webm'

      setIsTranscribing(true)
      try {
        const formData = new FormData()
        formData.append('audio', blob, `recording.${ext}`)
        const res = await fetch('/api/speech/transcribe', { method: 'POST', body: formData })
        if (res.ok) {
          const data = await res.json()
          onTranscript?.(data.transcript)
        }
        // Spec 21: a non-ok response (no speech detected, malformed output,
        // etc.) just means onTranscript never fires -- the text field (or
        // whatever the caller was populating) is simply left as-is for
        // manual input, no error surfaced.
      } catch (err) {
        // Network failure reaching the backend at all -- same graceful
        // fallback as a non-ok response above.
        console.warn('transcription failed', err)
      } finally {
        setIsTranscribing(false)
      }
    }

    mediaRecorderRef.current = recorder
    recorder.start()
    setIsRecording(true)

    maxDurationTimerRef.current = setTimeout(stopRecording, MAX_RECORDING_MS)
  }, [onTranscript, stopRecording])

  // Fire-and-forget by design: the caller never awaits this. It shows
  // whatever text it already has immediately, and audio plays independently
  // whenever the synthesis round-trip finishes -- never blocking the UI.
  const speak = useCallback((text: string) => {
    setIsSpeaking(true)
    fetch('/api/speech/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data: { audio_url: string }) => {
        const audio = new Audio(data.audio_url)
        audio.onended = () => setIsSpeaking(false)
        audio.onerror = () => setIsSpeaking(false)
        return audio.play()
      })
      .catch(() => setIsSpeaking(false))
  }, [])

  return { isRecording, isTranscribing, isSpeaking, startRecording, stopRecording, speak }
}

export default useVoice
