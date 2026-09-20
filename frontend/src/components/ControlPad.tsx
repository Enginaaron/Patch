import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Gamepad2, Square, X } from 'lucide-react'
import { phaseLabel } from '../lib/searchTypes'
import './ControlPad.css'

interface ControlPadProps {
  // Current rover movement phase, owned by the page (from GET /api/searches/{id}).
  phase?: string
  // True while an autonomous mission owns the wheels.
  autonomous?: boolean
}

type DriveCommand = 'forward' | 'backward' | 'turn_left' | 'turn_right'

const DRIVE_SPEED = 0.6

// The server stops the wheels on its own `drive_manual_ttl_seconds` (2 s by
// default) after the last drive command -- a deadman for a dropped connection
// or a frozen tab. While a button is held we therefore keep re-sending the
// command, comfortably inside that window.
const RESEND_INTERVAL_MS = 500

function postDrive(command: DriveCommand): Promise<Response> {
  return fetch('/api/drive', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command, speed: DRIVE_SPEED }),
  })
}

function postDriveStop(): void {
  // keepalive: a stop sent while the page is going away must still leave the
  // browser. (Drive commands deliberately do NOT get it -- a "go" that
  // outlives the page is the opposite of what we want.)
  fetch('/api/drive/stop', { method: 'POST', keepalive: true }).catch(() => {})
}

function ControlPad({ phase, autonomous = false }: ControlPadProps) {
  const [open, setOpen] = useState(false)
  const [held, setHeld] = useState<DriveCommand | null>(null)

  // The active hold (re-send timer) and the drive request currently in flight.
  const holdTimerRef = useRef<number | null>(null)
  const inFlightRef = useRef<Promise<void> | null>(null)

  const sendDrive = useCallback((command: DriveCommand) => {
    const request: Promise<void> = postDrive(command)
      .then(
        () => {},
        () => {},
      )
      .finally(() => {
        if (inFlightRef.current === request) inFlightRef.current = null
      })
    inFlightRef.current = request
  }, [])

  const stopWheels = useCallback(() => {
    postDriveStop()
    // Requests can overtake each other. A drive re-send still in flight could
    // be handled AFTER the stop above and restart the wheels, so stop once
    // more as soon as it has settled -- unless the user has pressed a button
    // again by then. After a release the last word is always "stop".
    const pending = inFlightRef.current
    if (pending) {
      void pending.then(() => {
        if (holdTimerRef.current === null) postDriveStop()
      })
    }
  }, [])

  const clearHold = useCallback(() => {
    if (holdTimerRef.current !== null) clearInterval(holdTimerRef.current)
    holdTimerRef.current = null
    setHeld(null)
  }, [])

  const endHold = useCallback(() => {
    // Only a real press may touch the rover. pointerleave also fires when the
    // mouse merely passes over a button, and a stray stop here would halt an
    // autonomous search the user never meant to interrupt.
    if (holdTimerRef.current === null) return
    stopWheels()
    clearHold()
  }, [stopWheels, clearHold])

  const beginHold = (command: DriveCommand) => {
    // A second finger on another button replaces the first hold; any release
    // then stops everything.
    if (holdTimerRef.current !== null) clearInterval(holdTimerRef.current)
    sendDrive(command)
    holdTimerRef.current = window.setInterval(() => {
      // Skip a beat rather than queue commands on a slow link: if the link is
      // bad enough to miss the window, the server-side deadman stops the wheels.
      if (inFlightRef.current === null) sendDrive(command)
    }, RESEND_INTERVAL_MS)
    setHeld(command)
  }

  // Anything that means "the finger may no longer be on the button" ends the
  // hold: the tab losing focus or being hidden, the page going away, the pad
  // being closed or unmounted. (Pointer up/leave/cancel are wired per button.)
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.hidden) endHold()
    }
    window.addEventListener('blur', endHold)
    window.addEventListener('pagehide', endHold)
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      window.removeEventListener('blur', endHold)
      window.removeEventListener('pagehide', endHold)
      document.removeEventListener('visibilitychange', onVisibilityChange)
      endHold()
    }
  }, [endHold])

  const close = () => {
    endHold()
    setOpen(false)
  }

  const handleStopClick = () => {
    // Explicit stop: always sent, even with no hold active, and sent before
    // anything else happens.
    stopWheels()
    clearHold()
  }

  if (!open) {
    return (
      <button type="button" className="control-fab" onClick={() => setOpen(true)} aria-label="Controls">
        <Gamepad2 size={22} color="white" strokeWidth={2} />
        Controls
      </button>
    )
  }

  const holdProps = (command: DriveCommand) => ({
    onPointerDown: (e: React.PointerEvent) => {
      if (e.button === 0) beginHold(command)
    },
    onPointerUp: endHold,
    onPointerLeave: endHold,
    onPointerCancel: endHold,
    // Long-press on touch screens opens a context menu, which swallows the
    // pointerup that would have stopped the rover.
    onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
  })

  const heldClass = (command: DriveCommand) => (held === command ? ' dpad__btn--held' : '')
  const phaseName = phase ?? 'idle'

  return (
    <section className="control-pad" aria-label="Drive controls">
      <header className="control-pad__header">
        <span className="control-pad__title">
          <Gamepad2 size={18} /> Controls
        </span>
        <button type="button" className="control-pad__close" onClick={close} aria-label="Close controls">
          <X size={18} />
        </button>
      </header>

      <div className="control-pad__status">
        <span className={`control-pad__phase control-pad__phase--${phaseName}`}>{phaseLabel(phaseName)}</span>
        <span className="control-pad__mode">{held ? 'manual' : autonomous ? 'autonomous' : 'at rest'}</span>
      </div>
      <p className={`control-pad__note${autonomous ? ' control-pad__note--warn' : ''}`}>
        {autonomous
          ? 'Patch is searching on its own. Driving manually stops the autonomous search; use Resume to continue it.'
          : 'Driving manually stops any autonomous search. Hold a button to move, release to stop.'}
      </p>

      <div className="dpad">
        <button
          type="button"
          className={`dpad__btn dpad__btn--up${heldClass('forward')}`}
          aria-label="Forward"
          {...holdProps('forward')}
        >
          <ArrowUp size={22} />
        </button>
        <button
          type="button"
          className={`dpad__btn dpad__btn--left${heldClass('turn_left')}`}
          aria-label="Turn left"
          {...holdProps('turn_left')}
        >
          <ArrowLeft size={22} />
        </button>
        <button type="button" className="dpad__btn dpad__btn--stop" onClick={handleStopClick} aria-label="Stop">
          <Square size={18} />
        </button>
        <button
          type="button"
          className={`dpad__btn dpad__btn--right${heldClass('turn_right')}`}
          aria-label="Turn right"
          {...holdProps('turn_right')}
        >
          <ArrowRight size={22} />
        </button>
        <button
          type="button"
          className={`dpad__btn dpad__btn--down${heldClass('backward')}`}
          aria-label="Backward"
          {...holdProps('backward')}
        >
          <ArrowDown size={22} />
        </button>
      </div>
    </section>
  )
}

export default ControlPad
