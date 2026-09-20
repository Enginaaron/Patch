import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Image, Mic, Plus, Search } from 'lucide-react'
import CameraFeed from '../components/CameraFeed'
import BoundingBoxOverlay from '../components/BoundingBoxOverlay'
import useVoice from '../hooks/useVoice'
import { formatDate, formatTime } from '../utils/datetime'
import './SearchPage.css'

interface ReferenceImageOut {
  id: string
  url: string
}

interface PendingCandidate {
  candidate_id: string
  description: string
  image_url: string
  crop_url: string
  box: [number, number, number, number] | null
  created_at: string
}

interface FindData {
  image_url: string
  crop_url: string
  found_at: string
}

interface SearchDetail {
  search_id: string
  target_text: string
  status: string
  reference_images: ReferenceImageOut[]
  pending_candidate: PendingCandidate | null
  find: FindData | null
}

type LoadState = 'loading' | 'loaded' | 'not_found' | 'error'

const STATUS_LABEL: Record<string, string> = {
  SEARCHING: 'Searching',
  CANDIDATE_PENDING: 'Reviewing a match',
  FOUND: 'Found',
  CANCELLED: 'Cancelled',
}

// Statuses where the AI loop is actively working -- these get the "thinking"
// dots. FOUND/CANCELLED are terminal, so they stay static.
const ACTIVE_STATUSES = new Set(['SEARCHING', 'CANDIDATE_PENDING'])

// Spec 14: a simple, reliable yes/no keyword check -- deliberately not a
// general conversational parser.
function matchConfirmIntent(transcript: string): 'confirm' | 'reject' | null {
  const t = transcript.toLowerCase()
  if (/\b(yes|yeah|yep|yup|that'?s it|correct)\b/.test(t)) return 'confirm'
  if (/\b(no|nope|not mine|incorrect|wrong)\b/.test(t)) return 'reject'
  return null
}

function SearchPage() {
  const navigate = useNavigate()
  const { searchId } = useParams<{ searchId: string }>()
  const [search, setSearch] = useState<SearchDetail | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [cancelling, setCancelling] = useState(false)
  const [deciding, setDeciding] = useState(false)
  const [visionError, setVisionError] = useState(false)
  const evidenceImgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    if (!searchId) return
    let cancelled = false

    fetch(`/api/searches/${searchId}`)
      .then((res) => {
        if (res.status === 404) throw new Error('not_found')
        if (!res.ok) throw new Error('error')
        return res.json() as Promise<SearchDetail>
      })
      .then((data) => {
        if (cancelled) return
        setSearch(data)
        setLoadState('loaded')
      })
      .catch((err) => {
        if (cancelled) return
        setLoadState(err instanceof Error && err.message === 'not_found' ? 'not_found' : 'error')
      })

    return () => {
      cancelled = true
    }
  }, [searchId])

  // Spec 13: candidate detection and status changes appear without a manual
  // refresh. Every event is treated purely as a "something changed, go
  // refetch" signal rather than a payload to merge into state directly --
  // status stays sourced from GET /api/searches/{id} alone (Spec 6), SSE
  // just tells us when to ask again.
  //
  // Spec 21: vision_error/vision_recovered are the exception -- they're
  // shown as a local nonfatal banner instead, since the search's actual
  // status hasn't changed and there's nothing new to refetch for them.
  useEffect(() => {
    if (!searchId) return

    const source = new EventSource(`/api/searches/${searchId}/events`)
    source.onopen = () => setVisionError(false)
    source.onmessage = (event) => {
      let type: string | undefined
      try {
        type = JSON.parse(event.data).type
      } catch {
        type = undefined
      }

      if (type === 'vision_error') {
        setVisionError(true)
        return
      }
      if (type === 'vision_recovered') {
        setVisionError(false)
        return
      }

      setVisionError(false)
      fetch(`/api/searches/${searchId}`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res)))
        .then((data: SearchDetail) => setSearch(data))
        .catch(() => {})
    }

    return () => {
      source.close()
    }
  }, [searchId])

  const handleCancel = async () => {
    if (!searchId || cancelling) return
    setCancelling(true)
    try {
      const res = await fetch(`/api/searches/${searchId}/cancel`, { method: 'POST' })
      if (!res.ok) throw new Error('failed to cancel')
      navigate('/')
    } catch {
      setCancelling(false)
    }
  }

  // Spec 14: confirm/reject works identically whether triggered by tap or
  // by voice -- both the buttons below and the voice keyword match call
  // this exact same function.
  const decide = async (action: 'confirm' | 'reject') => {
    if (!searchId || !search?.pending_candidate || deciding) return
    setDeciding(true)
    try {
      const res = await fetch(
        `/api/candidates/${search.pending_candidate.candidate_id}/${action}`,
        { method: 'POST' },
      )
      if (res.ok) {
        const data = (await res.json()) as SearchDetail
        setSearch(data)
      }
    } finally {
      setDeciding(false)
    }
  }

  const { isRecording, isTranscribing, startRecording, stopRecording, speak } = useVoice({
    onTranscript: (transcript) => {
      const intent = matchConfirmIntent(transcript)
      if (intent) decide(intent)
    },
  })

  const displayText = search?.target_text ?? '…'
  const referenceUrl = search?.reference_images[0]?.url
  const statusLabel = search ? (STATUS_LABEL[search.status] ?? search.status) : ''
  const candidate = search?.pending_candidate ?? null
  const showingCandidate = search?.status === 'CANDIDATE_PENDING' && candidate !== null
  const find = search?.find ?? null
  const showingFound = search?.status === 'FOUND' && find !== null

  // Spec 16: spoken confirmation fires once, the moment this search lands on
  // FOUND -- not on every refetch (SSE keep-alives, re-mounts, etc.).
  const spokenFoundRef = useRef<string | null>(null)
  useEffect(() => {
    if (!showingFound || !searchId) return
    if (spokenFoundRef.current === searchId) return
    spokenFoundRef.current = searchId
    speak('Found it!')
  }, [showingFound, searchId, speak])

  return (
    <main className="live-screen">
      <CameraFeed className="camera-feed--full" />

      <button type="button" className="live-screen__back" onClick={() => navigate('/')}>
        <ArrowLeft size={20} color="white" strokeWidth={2} />
        Back
      </button>

      {loadState === 'not_found' && <div className="live-screen__notice">Search not found.</div>}
      {loadState === 'error' && <div className="live-screen__notice">Couldn't load this search.</div>}

      {loadState === 'loaded' && search && (
        <>
          {/* Spec 14: this overlay clearly interrupts the normal searching
              state -- it replaces the status card/cancel button entirely
              rather than coexisting alongside them. */}
          {showingCandidate && candidate && (
            <div className="candidate-overlay">
              <div className="candidate-card">
                <h2 className="candidate-card__title">Possible match found</h2>

                <img src={candidate.crop_url} alt="" className="candidate-card__crop" />

                {candidate.box && (
                  <div className="candidate-card__evidence">
                    <img
                      ref={evidenceImgRef}
                      src={candidate.image_url}
                      alt=""
                      className="candidate-card__evidence-img"
                    />
                    <BoundingBoxOverlay
                      box2d={candidate.box}
                      mediaRef={evidenceImgRef}
                      objectFit="contain"
                      color="#2ecc71"
                      label="confirmed"
                    />
                  </div>
                )}

                <p className="candidate-card__question">Is this what you're looking for?</p>

                <div className="candidate-card__actions">
                  <button
                    type="button"
                    className={`candidate-mic${isRecording ? ' candidate-mic--recording' : ''}`}
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    onMouseLeave={() => isRecording && stopRecording()}
                    onTouchStart={(e) => {
                      e.preventDefault()
                      startRecording()
                    }}
                    onTouchEnd={(e) => {
                      e.preventDefault()
                      stopRecording()
                    }}
                    disabled={deciding || isTranscribing}
                    aria-label="Hold to answer by voice"
                    title="Hold to say yes or no"
                  >
                    <Mic size={20} color="white" strokeWidth={1.875} />
                  </button>
                  <button
                    type="button"
                    className="candidate-button candidate-button--reject"
                    onClick={() => decide('reject')}
                    disabled={deciding}
                  >
                    NOT MINE
                  </button>
                  <button
                    type="button"
                    className="candidate-button candidate-button--confirm"
                    onClick={() => decide('confirm')}
                    disabled={deciding}
                  >
                    THAT'S IT
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Spec 16: same full-interrupt pattern as the candidate card --
              replaces the status card/cancel button entirely. */}
          {showingFound && find && (
            <div className="found-overlay">
              <div className="found-card">
                <p className="found-card__eyebrow">FOUND</p>
                <h2 className="found-card__title">{displayText}</h2>
                <img src={find.crop_url} alt="" className="found-card__photo" />
                <p className="found-card__found-at-label">Found at</p>
                <p className="found-card__time">{formatTime(find.found_at)}</p>
                <p className="found-card__date">{formatDate(find.found_at)}</p>
                <button type="button" className="found-card__memory" onClick={() => navigate('/memory')}>
                  VIEW MEMORY
                </button>
              </div>
            </div>
          )}

          {!showingCandidate && !showingFound && (
            <div className="live-screen__bottom">
              {/* Spec 21: nonfatal -- the worker is backing off and retrying
                  on its own; this just keeps the user informed while it does. */}
              {visionError && (
                <div className="live-screen__vision-notice">
                  Vision temporarily unavailable.
                  <br />
                  Retrying...
                </div>
              )}

              <div className="live-screen__card">
                <div className="live-screen__status">
                  <span
                    className={`live-screen__status-dot live-screen__status-dot--${search.status.toLowerCase()}${
                      ACTIVE_STATUSES.has(search.status) ? ' live-screen__status-dot--active' : ''
                    }`}
                  />
                  {statusLabel}
                  {ACTIVE_STATUSES.has(search.status) && (
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

              {search.status === 'SEARCHING' && (
                <button type="button" className="live-screen__cancel" onClick={handleCancel} disabled={cancelling}>
                  {cancelling ? 'Cancelling…' : 'CANCEL SEARCH'}
                </button>
              )}
            </div>
          )}
        </>
      )}
    </main>
  )
}

export default SearchPage
