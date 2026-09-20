import { enableOmniSpeech, muteOmniSpeech, useOmniSpeaker } from '../lib/omniSpeaker'
import './VoiceOutput.css'

export default function VoiceOutput() {
  const voice = useOmniSpeaker()
  return (
    <aside className="voice-output" aria-label="Patch voice">
      <button type="button" onClick={() => voice.enabled && !voice.error ? muteOmniSpeech() : void enableOmniSpeech()}>
        {!voice.enabled ? "Enable Patch's voice" : voice.error ? 'Retry voice' : voice.busy ? 'Patch voice · Mute' : 'Voice on · Mute'}
      </button>
      {voice.enabled && !voice.busy && !voice.error && (
        <button type="button" onClick={() => void enableOmniSpeech()}>Replay reply</button>
      )}
      {voice.enabled && voice.error && <button type="button" onClick={muteOmniSpeech}>Mute</button>}
      {voice.error && <p role="alert">{voice.error}</p>}
    </aside>
  )
}
