import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Image, Mic, Play, Plus, Search, Square } from 'lucide-react'
import BoundingBoxOverlay from '../components/BoundingBoxOverlay'
import CameraFeed from '../components/CameraFeed'
import ChatPanel from '../components/ChatPanel'
import ControlPad from '../components/ControlPad'
import useAnnouncer from '../hooks/useAnnouncer'
import useVoice from '../hooks/useVoice'
import { formatDate, formatTime } from '../utils/datetime'
import { candidateQuestion, pickAnnouncement } from '../lib/announcements'
import {
  isTerminalMessage,
  normalizeSearchDetail,
  phaseLabel,
  postStop,
  readApiError,
  reasonLabel,
} from '../lib/searchTypes'
import type { SearchDetail } from '../lib/searchTypes'
import { parseVoiceCommand, VOICE_HINT_UNRECOGNIZED, VOICE_HINT_UNRESOLVED_REFERENCE } from '../lib/voiceCommands'
import './SearchPage.css'

type LoadState = 'loading' | 'loaded' | 'not_found' | 'error'
type Decision = 'accept' | 'reject'

// A note shown inside the candidate popup. It remembers which candidate it was
// about, so it disappears by itself when the next candidate is shown.
interface PopupNote {
  candidateId: string
  text: string
  tone: 'hint' | 'error'
}

// Recognition lifecycle labels. FOUND is "you confirmed the item", never "the
// rover reached it" -- arrival is only ever the movement phase 'arrived'.
const STATUS_LABEL: Record<string, string> = {
  SEARCHING: 'Searching',
  CANDIDATE_PENDING: 'Reviewing a match',
  FOUND: 'Item confirmed',
  CANCELLED: 'Cancelled',
}

// Statuses where the AI loop may be working -- these get the "thinking" dots,
// but only while a mission is actually running (a stopped or exhausted rover
// is still SEARCHING in the database).
const ACTIVE_STATUSES = new Set(['SEARCHING', 'CANDIDATE_PENDING'])

// SSE is the fast path; this slow poll is the safety net for a refetch that
// failed or an event stream that died, so the page can never sit on a stale
// "approaching" forever.
const SAFETY_POLL_MS = 5000
// EventSource retries network drops by itself but gives up for good on an HTTP
// error (e.g. a proxy 502 while the backend restarts); then we reopen it.
const SSE_REOPEN_MS = 3000

function SearchPage() {
  const { searchId } = useParams<{ searchId: string }>()
  // Keyed by id: request tickets, the event stream, in-flight flags and popup
  // notes all belong to ONE search and reset together if the route changes.
  return <SearchView key={searchId ?? ''} searchId={searchId} />
}

function SearchView({ searchId }: { searchId: string | undefined }) {
  const navigate = useNavigate()
  const [search, setSearch] = useState<SearchDetail | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [connectionLost, setConnectionLost] = useState(false)
  const [sseEpoch, setSseEpoch] = useState(0)
  const [stopping, setStopping] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [resuming, setResuming] = useState(false)
  const [answering, setAnswering] = useState<Decision | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [popupNote, setPopupNote] = useState<PopupNote | null>(null)
  // Candidate id whose popup shows the whole frame (with the box drawn on it)
  // instead of the crop. Tied to the id so the next candidate starts on its crop.
  const [sceneFor, setSceneFor] = useState<string | null>(null)

  const aliveRef = useRef(false)
  // Out-of-order guard: every request that returns a search detail takes a
  // ticket; a response is applied only if nothing newer is already on screen.
  const ticketRef = useRef(0)
  const appliedTicketRef = useRef(0)
  const answeringRef = useRef(false)
  const resumingRef = useRef(false)
  // The candidate that was on screen when the user started talking. A
  // transcript takes seconds; it may only ever answer THAT candidate.
  const recordingForRef = useRef<string | null>(null)
  const transcriptHandlerRef = useRef<(transcript: string | null) => void>(() => {})
  const sceneImageRef = useRef<HTMLImageElement>(null)

  const onTranscript = useCallback((transcript: string) => transcriptHandlerRef.current(transcript), [])
  const onTranscriptError = useCallback(() => transcriptHandlerRef.current(null), [])
  const { isRecording, isTranscribing, startRecording, stopRecording, speak, stopSpeaking } = useVoice({
    onTranscript,
    onTranscriptError,
  })
  const { announce } = useAnnouncer(speak)

  useEffect(() => {
    aliveRef.current = true
    return () => {
      aliveRef.current = false
    }
  }, [])

  const applyDetail = useCallback((ticket: number, body: unknown): boolean => {
    const detail = normalizeSearchDetail(body)
    if (!detail) return false
    if (!aliveRef.current || ticket < appliedTicketRef.current) return true
    appliedTicketRef.current = ticket
    setSearch(detail)
    setLoadState('loaded')
    setConnectionLost(false)
    return true
  }, [])

  const refetch = useCallback(async () => {
    if (!searchId) return
    const ticket = ++ticketRef.current
    try {
      const res = await fetch(`/api/searches/${searchId}`)
      if (res.status === 404) {
        if (aliveRef.current && ticket > appliedTicketRef.current) setLoadState('not_found')
        return
      }
      if (!res.ok) throw new Error('error')
      if (!applyDetail(ticket, await res.json())) throw new Error('error')
    } catch {
      if (!aliveRef.current) return
      setConnectionLost(true)
      setLoadState((prev) => (prev === 'loading' ? 'error' : prev))
    }
  }, [searchId, applyDetail])

  // Initial load. Deferred by one task so React StrictMode's throwaway first
  // mount cancels its request before it is ever sent (one GET, not two).
  useEffect(() => {
    const timer = window.setTimeout(() => void refetch(), 0)
    return () => clearTimeout(timer)
  }, [refetch])

  const terminal = search?.terminal ?? false
  // Nothing more will ever arrive for a finished or unknown search.
  const streamDone = terminal || loadState === 'not_found'

  // Spec 13: candidate detection and status changes appear without a manual
  // refresh. Every event is treated purely as a "something changed, go
  // refetch" signal rather than a payload to merge into state directly --
  // status stays sourced from GET /api/searches/{id} alone (Spec 6), SSE
  // just tells us when to ask again. That also makes missed events harmless:
  // after a reconnect the server's first message triggers a refetch.
  useEffect(() => {
    if (!searchId || streamDone) return

    const source = new EventSource(`/api/searches/${searchId}/events`)
    let reopenTimer: number | null = null

    source.onmessage = (event) => {
      void refetch()
      // The server ends the stream after a terminal event. Left alone,
      // EventSource would treat that as a dropped connection and reconnect
      // forever, so a finished search closes its own stream.
      if (isTerminalMessage(event.data)) source.close()
    }
    source.onerror = () => {
      void refetch()
      if (source.readyState === EventSource.CLOSED && reopenTimer === null) {
        reopenTimer = window.setTimeout(() => setSseEpoch((n) => n + 1), SSE_REOPEN_MS)
      }
    }

    return () => {
      // Also what makes React StrictMode's mount/unmount/mount harmless: the
      // first stream is closed before the second one opens.
      source.close()
      if (reopenTimer !== null) clearTimeout(reopenTimer)
    }
  }, [searchId, streamDone, sseEpoch, refetch])

  useEffect(() => {
    if (!searchId || streamDone) return
    const timer = window.setInterval(() => {
      if (!document.hidden) void refetch()
    }, SAFETY_POLL_MS)
    // Phones suspend timers and streams in the background; catch up at once.
    const onVisibilityChange = () => {
      if (!document.hidden) void refetch()
    }
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => {
      clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [searchId, streamDone, refetch])

  // --- rover actions -------------------------------------------------------
  // STOP and CANCEL: the request leaves first, synchronously in the click
  // handler. They never wait for speech, a transcript or another request, and
  // they are never disabled -- pressing again simply sends again (both
  // endpoints are idempotent).

  const handleStop = () => {
    const request = postStop()
    stopSpeaking()
    setStopping(true)
    setActionError(null)
    void request.then((confirmed) => {
      if (!aliveRef.current) return
      setStopping(false)
      if (!confirmed) setActionError("Stop wasn't confirmed. Press again, or switch the motors off.")
      void refetch()
    })
  }

  const handleCancel = () => {
    if (!searchId) return
    const request = fetch(`/api/searches/${searchId}/cancel`, { method: 'POST', keepalive: true })
    stopSpeaking()
    setCancelling(true)
    setActionError(null)
    request
      .then((res) => {
        if (!res.ok) throw new Error('failed to cancel')
        if (aliveRef.current) navigate('/')
      })
      .catch(() => {
        if (!aliveRef.current) return
        setCancelling(false)
        setActionError("Couldn't cancel the search. Press again.")
        void refetch()
      })
  }

  const handleResume = async () => {
    if (!searchId || resumingRef.current) return
    resumingRef.current = true
    setResuming(true)
    setActionError(null)
    const ticket = ++ticketRef.current
    try {
      const res = await fetch(`/api/searches/${searchId}/resume`, { method: 'POST' })
      const body: unknown = await res.json().catch(() => null)
      if (!res.ok || !applyDetail(ticket, body)) {
        const { code, message } = readApiError(body)
        if (aliveRef.current) {
          setActionError(
            code === 'rover_busy' ? 'Patch is busy with another search.' : (message ?? "Can't resume right now."),
          )
        }
        void refetch()
      }
    } catch {
      if (aliveRef.current) setActionError("Couldn't reach Patch. Try again.")
    } finally {
      resumingRef.current = false
      if (aliveRef.current) setResuming(false)
    }
  }

  const answerCandidate = async (decision: Decision, candidateId: string) => {
    if (!searchId || answeringRef.current) return
    answeringRef.current = true
    setAnswering(decision)
    setPopupNote(null)
    setActionError(null)
    stopSpeaking() // the question has been answered; stop asking it
    const ticket = ++ticketRef.current
    try {
      const res = await fetch(`/api/searches/${searchId}/candidates/${candidateId}/${decision}`, { method: 'POST' })
      const body: unknown = await res.json().catch(() => null)
      if (!res.ok || !applyDetail(ticket, body)) {
        const { code, message } = readApiError(body)
        // The popup is about to disappear for these, so the message goes next
        // to the action buttons instead of inside it.
        if (aliveRef.current) {
          setActionError(
            code === 'invalid_transition'
              ? 'That match is no longer waiting for an answer.'
              : code === 'rover_busy'
                ? 'Patch is busy with another search.'
                : (message ?? "Couldn't send your answer. Try again."),
          )
        }
        void refetch()
      }
    } catch {
      if (aliveRef.current) setPopupNote({ candidateId, text: "Couldn't reach Patch. Try again.", tone: 'error' })
    } finally {
      answeringRef.current = false
      if (aliveRef.current) setAnswering(null)
    }
  }

  // --- derived view state --------------------------------------------------

  const movement = search?.movement ?? null
  // Popup only for a candidate that is really waiting for an answer.
  const pending = search && search.status === 'CANDIDATE_PENDING' ? search.pending_candidate : null
  const pendingId = pending?.candidate_id ?? null
  const visibleNote = pending && popupNote?.candidateId === pending.candidate_id ? popupNote : null
  const simulated = movement?.vision_mode === 'simulation'
  // With two similar items a crop cannot say WHICH one Patch means; the frame
  // it came from, with the box drawn on it, can.
  const sceneBox = pending?.image_url ? pending.box : null
  const showScene = pending !== null && sceneFor === pending.candidate_id

  // STOP shows whenever a mission owns the wheels -- and also whenever we
  // cannot tell (still loading, load failed, connection lost): an unneeded
  // STOP button is harmless, a missing one is not.
  const showStop = search ? search.movement.active || connectionLost : loadState !== 'not_found'
  const showCancel = search !== null && !search.terminal
  const showResume = search?.can_resume ?? false

  // --- speech --------------------------------------------------------------

  const announcement = search ? pickAnnouncement(search) : null
  const announcementKey = announcement?.key ?? null
  const announcementText = announcement?.text ?? null
  useEffect(() => {
    if (announcementKey && announcementText) announce(announcementKey, announcementText)
  }, [announcementKey, announcementText, announce])

  // Re-pointed after every render so a transcript that arrives seconds later
  // is judged against what is on screen NOW, not when the recording started.
  useEffect(() => {
    transcriptHandlerRef.current = (transcript) => {
      if (!aliveRef.current) return
      const heardFor = recordingForRef.current
      recordingForRef.current = null

      const parsed = transcript === null ? null : parseVoiceCommand(transcript)
      if (parsed?.kind === 'command' && parsed.command === 'stop') {
        handleStop()
        return
      }
      // Never answer a candidate the user was not looking at when they spoke.
      if (!pending || heardFor !== pending.candidate_id) return

      const candidateId = pending.candidate_id
      if (parsed === null) {
        setPopupNote({ candidateId, text: "Didn't catch that. Say yes or no, or use the buttons.", tone: 'hint' })
      } else if (parsed.kind === 'command') {
        void answerCandidate(parsed.command === 'accept' ? 'accept' : 'reject', candidateId)
      } else if (parsed.kind === 'unresolved_reference') {
        setPopupNote({ candidateId, text: VOICE_HINT_UNRESOLVED_REFERENCE, tone: 'hint' })
      } else {
        setPopupNote({ candidateId, text: VOICE_HINT_UNRECOGNIZED, tone: 'hint' })
      }
    }
  })

  // The mic button lives inside the popup; if the popup goes away mid-press
  // its pointerup never fires, so stop listening when the candidate changes.
  useEffect(() => stopRecording, [pendingId, stopRecording])

  const beginVoiceAnswer = (candidateId: string) => {
    recordingForRef.current = candidateId
    stopSpeaking() // don't record Patch's own voice asking the question
    setPopupNote(null)
    startRecording().catch(() => {
      if (aliveRef.current) setPopupNote({ candidateId, text: 'Microphone unavailable. Use the buttons.', tone: 'error' })
    })
  }

  // --- render --------------------------------------------------------------

  const displayText = search?.target_text ?? '…'
  const referenceUrl = search?.reference_images[0]?.url
  const searchRunning = search !== null && ACTIVE_STATUSES.has(search.status) && search.movement.active
  const statusLabel = !search
    ? ''
    : search.status === 'SEARCHING' && !search.movement.active
      ? 'Search paused'
      : (STATUS_LABEL[search.status] ?? search.status)
  const showScanProgress = search?.status === 'SEARCHING' && movement !== null && movement.scan_budget > 0
  const scanSteps = movement ? Math.min(movement.scan_steps, movement.scan_budget) : 0

  return (
    <main className="live-screen">
      <CameraFeed className="camera-feed--full" />

      <div className="live-screen__top-left">
        <button type="button" className="live-screen__back" onClick={() => navigate('/')}>
          <ArrowLeft size={20} color="white" strokeWidth={2} />
          Back
        </button>

        {showStop && (
          // The label never changes: an emergency button should not shift or
          // re-word itself under the finger. "Sending" is shown by colour only.
          <button
            type="button"
            className={`live-screen__stop${stopping ? ' live-screen__stop--sending' : ''}`}
            onClick={handleStop}
            aria-busy={stopping}
          >
            <Square size={18} fill="currentColor" />
            STOP
          </button>
        )}
        {showResume && (
          <button type="button" className="live-screen__resume" onClick={handleResume} disabled={resuming}>
            <Play size={16} fill="currentColor" />
            {resuming ? 'Resuming…' : 'Resume'}
          </button>
        )}
        {showCancel && (
          <button type="button" className="live-screen__cancel" onClick={handleCancel}>
            {cancelling ? 'CANCELLING…' : 'CANCEL SEARCH'}
          </button>
        )}
        {actionError && (
          <p className="live-screen__action-error" role="alert">
            {actionError}
          </p>
        )}
      </div>

      {loadState === 'not_found' && <div className="live-screen__notice">Search not found.</div>}
      {loadState === 'error' && <div className="live-screen__notice">Couldn't load this search.</div>}

      {search && pending && (
        <section className="live-screen__popup" aria-label="Possible match" aria-live="polite">
          {simulated && <span className="live-screen__sim-tag">Simulated detection</span>}
          {sceneBox && showScene ? (
            <div className="live-screen__popup-scene">
              <img ref={sceneImageRef} src={pending.image_url} alt="The view this match was found in" />
              <BoundingBoxOverlay box2d={sceneBox} mediaRef={sceneImageRef} objectFit="contain" />
            </div>
          ) : (
            <img className="live-screen__popup-crop" src={pending.crop_url} alt={pending.description || 'Possible match'} />
          )}
          {sceneBox && (
            <button
              type="button"
              className="live-screen__popup-toggle"
              onClick={() => setSceneFor(showScene ? null : pending.candidate_id)}
            >
              {showScene ? 'Show the close-up' : 'Show where it is'}
            </button>
          )}
          {pending.description && <p className="live-screen__popup-description">{pending.description}</p>}
          <p className="live-screen__popup-question">{candidateQuestion(search)}</p>
          {!pending.box && (
            <p className="live-screen__popup-caveat">
              Patch can't tell where this is in the view, so it won't drive to it.
            </p>
          )}
          <div className="live-screen__popup-actions">
            <button
              type="button"
              className="popup-button popup-button--yes"
              onClick={() => answerCandidate('accept', pending.candidate_id)}
              disabled={answering !== null}
            >
              {answering === 'accept' ? 'Sending…' : 'Yes'}
            </button>
            <button
              type="button"
              className="popup-button popup-button--no"
              onClick={() => answerCandidate('reject', pending.candidate_id)}
              disabled={answering !== null}
            >
              {answering === 'reject' ? 'Sending…' : 'No'}
            </button>
          </div>
          <button
            type="button"
            className={`popup-mic${isRecording ? ' popup-mic--recording' : ''}`}
            disabled={isTranscribing}
            onPointerDown={(e) => {
              if (e.button === 0) beginVoiceAnswer(pending.candidate_id)
            }}
            onPointerUp={stopRecording}
            onPointerLeave={stopRecording}
            onPointerCancel={stopRecording}
            onContextMenu={(e) => e.preventDefault()}
          >
            <Mic size={14} />
            {isRecording ? 'Listening… release to send' : isTranscribing ? 'Working it out…' : 'Hold to answer by voice'}
          </button>
          {visibleNote && (
            <p className={`live-screen__popup-note live-screen__popup-note--${visibleNote.tone}`} role="status">
              {visibleNote.text}
            </p>
          )}
        </section>
      )}

      {search && movement && (
        <div className="live-screen__bottom">
          {search.find && (
            <section className="live-screen__card" aria-label="Saved find">
              <p>{movement.phase === 'arrived' ? 'Arrived at your item' : 'Item identified and saved'}</p>
              <img src={search.find.crop_url} alt={displayText} style={{ maxWidth: 120, maxHeight: 100, objectFit: 'contain' }} />
              <p>{formatDate(search.find.found_at)} · {formatTime(search.find.found_at)}</p>
              <button type="button" onClick={() => navigate('/memory')}>View memory</button>
            </section>
          )}
          {connectionLost && (
            <div className="live-screen__offline" role="status">
              Connection lost. Retrying…
            </div>
          )}

          <section className="live-screen__card live-screen__card--movement" aria-label="Rover movement">
            <div className="live-screen__movement-head">
              <span className={`live-screen__phase live-screen__phase--${movement.phase}`}>
                {phaseLabel(movement.phase)}
              </span>
              {simulated && (
                <span
                  className="live-screen__badge live-screen__badge--sim"
                  title="Detections come from the built-in simulation, not from the camera."
                >
                  SIMULATION
                </span>
              )}
            </div>
            {movement.message && <p className="live-screen__movement-message">{movement.message}</p>}
            {!movement.active && movement.reason && (
              <p className="live-screen__movement-reason">{reasonLabel(movement.reason)}</p>
            )}
            {showScanProgress && (
              <div className="live-screen__progress">
                <span>
                  Scan {scanSteps}/{movement.scan_budget}
                </span>
                <span
                  className="live-screen__progress-track"
                  role="progressbar"
                  aria-label="Scan progress"
                  aria-valuemin={0}
                  aria-valuemax={movement.scan_budget}
                  aria-valuenow={scanSteps}
                >
                  <span style={{ width: `${(scanSteps / movement.scan_budget) * 100}%` }} />
                </span>
              </div>
            )}
            {search.status === 'FOUND' && movement.approach_pulses > 0 && (
              <p className="live-screen__movement-reason">Approach moves: {movement.approach_pulses}</p>
            )}
            <div className="live-screen__badges">
              {!simulated && <span className="live-screen__badge">LIVE VISION</span>}
              <span className={`live-screen__badge${movement.driver === 'gpio' ? ' live-screen__badge--real' : ''}`}>
                {movement.driver === 'sim'
                  ? 'SIM MOTORS'
                  : movement.driver === 'gpio'
                    ? 'REAL MOTORS'
                    : movement.driver.toUpperCase()}
              </span>
            </div>
          </section>

          <div className="live-screen__card live-screen__card--search">
            <div className="live-screen__status">
              <span
                className={`live-screen__status-dot live-screen__status-dot--${search.status.toLowerCase()}${
                  searchRunning ? ' live-screen__status-dot--active' : ''
                }`}
              />
              {statusLabel}
              {searchRunning && (
                <span className="live-screen__status-thinking" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
              )}
            </div>

            <p className="live-screen__label">You're looking for:</p>
            <div className="live-screen__target-pill">
              <Search size={16} />
              <span>{displayText}</span>
            </div>

            <div className="live-screen__photos">
              <div className="live-screen__photo-slot">
                {referenceUrl ? (
                  <img src={referenceUrl} alt="" />
                ) : (
                  <Image size={20} color="#999" strokeWidth={1.5} />
                )}
              </div>
              <button
                type="button"
                className="live-screen__photo-slot live-screen__photo-slot--add"
                disabled
                title="Editing reference photos mid-search isn't available yet"
              >
                <Plus size={18} />
              </button>
            </div>

            <div className="live-screen__match">
              <span className="live-screen__label">Match Level:</span>
              <span className="live-screen__match-badge">
                <span className="live-screen__match-dot" />
                &mdash;
              </span>
            </div>
          </div>
        </div>
      )}

      <ControlPad phase={movement?.phase} autonomous={movement?.active ?? false} />
      <ChatPanel targetText={search?.target_text} />
    </main>
  )
}

export default SearchPage
