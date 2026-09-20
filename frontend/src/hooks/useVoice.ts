import { useCallback, useEffect, useRef, useState } from 'react'
import { speakOmni, stopOmniSpeech, useOmniSpeaker } from '../lib/omniSpeaker'

// Safety cap so a stuck recording can't run forever if the user forgets to
// release the mic button. Real silence-based auto-stop (VAD) is a later
// refinement -- release-to-stop is the primary path for now.
const MAX_RECORDING_MS = 10_000

interface UseVoiceOptions {
  onTranscript?: (transcript: string) => void
  // Optional: the transcription request failed. Lets a caller say "didn't
  // catch that" instead of silently doing nothing.
  onTranscriptError?: () => void
}

interface UseVoiceResult {
  isRecording: boolean
  isTranscribing: boolean
  isSpeaking: boolean
  startRecording: () => Promise<void>
  stopRecording: () => void
  speak: (text: string) => void
  stopSpeaking: () => void
}

function useVoice({ onTranscript, onTranscriptError }: UseVoiceOptions = {}): UseVoiceResult {
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const { busy: isSpeaking } = useOmniSpeaker()

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const maxDurationTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Bumped on every press and every release. getUserMedia is async, so a
  // quick tap can release (or press again) before the stream exists; a stream
  // that arrives for an outdated press is discarded instead of starting a
  // recording nobody is holding the button for.
  const recordAttemptRef = useRef(0)
  const stopRecording = useCallback(() => {
    recordAttemptRef.current += 1
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
    stopOmniSpeech()
    const attempt = ++recordAttemptRef.current
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    if (attempt !== recordAttemptRef.current) {
      stream.getTracks().forEach((track) => track.stop())
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
        } else {
          onTranscriptError?.()
        }
      } catch {
        onTranscriptError?.()
      } finally {
        setIsTranscribing(false)
      }
    }

    mediaRecorderRef.current = recorder
    recorder.start()
    setIsRecording(true)

    maxDurationTimerRef.current = setTimeout(stopRecording, MAX_RECORDING_MS)
  }, [onTranscript, onTranscriptError, stopRecording])

  const speak = speakOmni
  const stopSpeaking = stopOmniSpeech

  // Leaving the page releases the microphone. Shared speech survives route
  // mounts (including StrictMode); a new reply or explicit stop replaces it. A recording
  // abandoned this way is discarded, not transcribed: its answer would land
  // on a page the user has already left.
  useEffect(() => {
    return () => {
      recordAttemptRef.current += 1
      if (maxDurationTimerRef.current) clearTimeout(maxDurationTimerRef.current)
      const recorder = mediaRecorderRef.current
      if (recorder && recorder.state !== 'inactive') {
        recorder.onstop = null
        recorder.stop()
      }
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
  }, [])

  return { isRecording, isTranscribing, isSpeaking, startRecording, stopRecording, speak, stopSpeaking }
}

export default useVoice
