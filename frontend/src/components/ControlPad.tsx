import { useEffect, useRef, useState } from 'react'
import {
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Gamepad2,
  Radar,
  Square,
  X,
} from 'lucide-react'
import './ControlPad.css'

interface ControlPadProps {
  searchId?: string
  targetText?: string
}

interface SearchStatus {
  active: boolean
  phase: string
  message: string
  confidence: number
  mode: string
}

const DRIVE_SPEED = 0.6

async function postJson(url: string, body?: unknown) {
  return fetch(url, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  })
}

function ControlPad({ searchId, targetText }: ControlPadProps) {
  const [open, setOpen] = useState(false)
  const [auto, setAuto] = useState(false)
  const [status, setStatus] = useState<SearchStatus | null>(null)
  const pollRef = useRef<number | null>(null)

  const drive = (command: string) => {
    setAuto(false)
    postJson('/api/drive', { command, speed: DRIVE_SPEED }).catch(() => {})
  }
  const stopDrive = () => {
    postJson('/api/drive', { command: 'stop' }).catch(() => {})
  }

  const stopPolling = () => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const startAuto = async () => {
    if (!searchId) return
    setAuto(true)
    await postJson(`/api/searches/${searchId}/start-auto`, {
      target_text: targetText ?? '',
    }).catch(() => {})
    stopPolling()
    pollRef.current = window.setInterval(async () => {
      try {
        const res = await fetch('/api/searches/auto-status')
        const data: SearchStatus = await res.json()
        setStatus(data)
        if (!data.active) {
          setAuto(false)
          stopPolling()
        }
      } catch {
        /* ignore poll errors */
      }
    }, 600)
  }

  const stopAuto = async () => {
    setAuto(false)
    stopPolling()
    await postJson('/api/searches/stop-auto').catch(() => {})
  }

  useEffect(() => stopPolling, [])

  if (!open) {
    return (
      <button type="button" className="control-fab" onClick={() => setOpen(true)} aria-label="Controls">
        <Gamepad2 size={22} color="white" strokeWidth={2} />
        Controls
      </button>
    )
  }

  const phaseLabel = status?.active ? status.phase : 'idle'

  return (
    <section className="control-pad" aria-label="Drive controls">
      <header className="control-pad__header">
        <span className="control-pad__title">
          <Gamepad2 size={18} /> Controls
        </span>
        <button type="button" className="control-pad__close" onClick={() => setOpen(false)} aria-label="Close controls">
          <X size={18} />
        </button>
      </header>

      <button
        type="button"
        className={`control-pad__auto ${auto ? 'control-pad__auto--on' : ''}`}
        onClick={auto ? stopAuto : startAuto}
        disabled={!searchId}
      >
        {auto ? <Square size={16} /> : <Radar size={16} />}
        {auto ? 'Stop search' : 'Auto search'}
      </button>

      <div className="control-pad__status">
        <span className={`control-pad__phase control-pad__phase--${phaseLabel}`}>{phaseLabel}</span>
        {status?.mode && <span className="control-pad__mode">{status.mode}</span>}
        {status?.active && <span className="control-pad__conf">{Math.round((status.confidence ?? 0) * 100)}%</span>}
      </div>
      {status?.message && <p className="control-pad__message">{status.message}</p>}

      <div className="dpad">
        <button
          type="button"
          className="dpad__btn dpad__btn--up"
          onPointerDown={() => drive('forward')}
          onPointerUp={stopDrive}
          onPointerLeave={stopDrive}
          aria-label="Forward"
        >
          <ArrowUp size={22} />
        </button>
        <button
          type="button"
          className="dpad__btn dpad__btn--left"
          onPointerDown={() => drive('turn_left')}
          onPointerUp={stopDrive}
          onPointerLeave={stopDrive}
          aria-label="Turn left"
        >
          <ArrowLeft size={22} />
        </button>
        <button type="button" className="dpad__btn dpad__btn--stop" onClick={stopDrive} aria-label="Stop">
          <Square size={18} />
        </button>
        <button
          type="button"
          className="dpad__btn dpad__btn--right"
          onPointerDown={() => drive('turn_right')}
          onPointerUp={stopDrive}
          onPointerLeave={stopDrive}
          aria-label="Turn right"
        >
          <ArrowRight size={22} />
        </button>
        <button
          type="button"
          className="dpad__btn dpad__btn--down"
          onPointerDown={() => drive('backward')}
          onPointerUp={stopDrive}
          onPointerLeave={stopDrive}
          aria-label="Backward"
        >
          <ArrowDown size={22} />
        </button>
      </div>
    </section>
  )
}

export default ControlPad
