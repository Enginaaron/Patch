import { useState } from 'react'
import useVoice from '../hooks/useVoice'

// Isolated test surface for Spec 8, same principle as Spec 7's debug script:
// exercises both directions of useVoice() outside any real product flow, and
// makes the non-blocking speak() pattern visible (text renders immediately,
// audio plays independently whenever synthesis lands).
function DevVoicePage() {
  const [transcript, setTranscript] = useState('')
  const [speakText, setSpeakText] = useState('Still there')
  const [textShownAt, setTextShownAt] = useState<number | null>(null)

  const { isRecording, isTranscribing, isSpeaking, startRecording, stopRecording, speak } = useVoice({
    onTranscript: setTranscript,
  })

  const handleSpeak = () => {
    setTextShownAt(Date.now())
    speak(speakText)
  }

  return (
    <main style={{ maxWidth: 480, margin: '0 auto', padding: 24, fontFamily: 'system-ui' }}>
      <h1>Voice dev test (Spec 8)</h1>

      <section style={{ marginTop: 24 }}>
        <h2>Speech-in</h2>
        <button
          type="button"
          onMouseDown={startRecording}
          onMouseUp={stopRecording}
          onMouseLeave={() => isRecording && stopRecording()}
          style={{ padding: '12px 24px', fontSize: 16 }}
        >
          {isRecording ? 'Recording… release to stop' : 'Hold to speak'}
        </button>
        <p>{isTranscribing ? 'Transcribing…' : `Transcript: ${transcript || '(none yet)'}`}</p>
      </section>

      <section style={{ marginTop: 24 }}>
        <h2>Speech-out (non-blocking)</h2>
        <input
          value={speakText}
          onChange={(e) => setSpeakText(e.target.value)}
          style={{ padding: 8, fontSize: 16, width: '70%' }}
        />
        <button type="button" onClick={handleSpeak} style={{ padding: '8px 16px', marginLeft: 8 }}>
          Speak
        </button>
        <p>
          Text shown at: {textShownAt ? new Date(textShownAt).toLocaleTimeString() : '—'}
          {' — '}
          {isSpeaking ? 'audio pending/playing…' : 'idle'}
        </p>
      </section>
    </main>
  )
}

export default DevVoicePage
