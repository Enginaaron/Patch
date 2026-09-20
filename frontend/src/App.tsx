import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AlignLeft, ArrowUp, CirclePlus, Mic, X } from 'lucide-react'
import logoMark from './assets/logo-mark.svg'
import useVoice from './hooks/useVoice'
import { readApiError } from './lib/searchTypes'
import './App.css'

type ConnectionStatus = 'checking' | 'connected' | 'disconnected'

const MAX_REFERENCE_IMAGES = 3

interface ReferenceImageSlot {
  file: File
  previewUrl: string
}

// 409 rover_busy: Patch has one body, so one search at a time. Starting
// another means stopping and cancelling the running one -- the user decides.
interface BusyPrompt {
  activeSearchId: string | null
  activeTargetText: string
}

const DEFAULT_CLARIFICATION = 'Which item do you mean? Tell me what it is.'

function App() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const [targetText, setTargetText] = useState('')
  const [images, setImages] = useState<ReferenceImageSlot[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busyPrompt, setBusyPrompt] = useState<BusyPrompt | null>(null)
  const [clarification, setClarification] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Both prompts are about the text that was submitted; once it changes they
  // no longer apply.
  const updateTargetText = (text: string) => {
    setTargetText(text)
    setBusyPrompt(null)
    setClarification(null)
  }

  const { isRecording, isTranscribing, startRecording, stopRecording, speak, stopSpeaking } = useVoice({
    onTranscript: (transcript) => {
      if (transcript) updateTargetText(transcript)
    },
  })

  const beginRecording = () => {
    stopSpeaking() // don't record Patch's own question back into the search box
    startRecording().catch(() => setError('Microphone unavailable'))
  }

  useEffect(() => {
    fetch('/api/health')
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => setStatus(data.status === 'ok' ? 'connected' : 'disconnected'))
      .catch(() => setStatus('disconnected'))
  }, [])

  const remainingSlots = MAX_REFERENCE_IMAGES - images.length

  const handleAddPhotoClick = () => {
    if (remainingSlots <= 0) return
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []).slice(0, remainingSlots)
    e.target.value = ''
    if (files.length === 0) return
    setImages((prev) => [...prev, ...files.map((file) => ({ file, previewUrl: URL.createObjectURL(file) }))])
  }

  const handleRemoveImage = (index: number) => {
    setImages((prev) => {
      URL.revokeObjectURL(prev[index].previewUrl)
      return prev.filter((_, i) => i !== index)
    })
  }

  const submitSearch = async (replaceActive = false) => {
    const text = targetText.trim()
    if (!text || submitting) return

    setSubmitting(true)
    setError(null)
    setBusyPrompt(null)
    setClarification(null)
    try {
      const formData = new FormData()
      formData.append('target_text', text)
      images.forEach((img) => formData.append('reference_images', img.file))
      if (replaceActive) formData.append('replace_active', 'true')

      const res = await fetch('/api/searches', { method: 'POST', body: formData })
      if (!res.ok) {
        const body: unknown = await res.json().catch(() => null)
        const { code, message, fields } = readApiError(body)

        if (res.status === 409 && code === 'rover_busy') {
          // Nothing was created. Ask before taking the rover away from the
          // search it is on; "replace" resubmits with replace_active=true.
          setBusyPrompt({
            activeSearchId: typeof fields.active_search_id === 'string' ? fields.active_search_id : null,
            activeTargetText: typeof fields.active_target_text === 'string' ? fields.active_target_text : '',
          })
          setSubmitting(false)
          return
        }

        if (res.status === 422 && code === 'needs_clarification') {
          // "the other one", "that one"… -- a transcript cannot tell us which
          // object that is, so ask instead of searching for a guess. No search
          // exists yet: stay here.
          const question =
            typeof fields.question === 'string' && fields.question ? fields.question : DEFAULT_CLARIFICATION
          setClarification(question)
          speak(question)
          setSubmitting(false)
          return
        }

        throw new Error(message ?? 'Failed to start search')
      }
      const data = await res.json()
      navigate(`/search/${data.search_id}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start search')
      setSubmitting(false)
    }
  }

  return (
    <main className="home">
      <span
        className={`status-dot status-dot--${status}`}
        title={`Backend: ${status}`}
        aria-label={`Backend status: ${status}`}
      />

      <Link to="/memory" className="menu-button" aria-label="Open memory">
        <AlignLeft size={20} color="white" strokeWidth={2} />
      </Link>

      <div className="home__center">
        <img src={logoMark} alt="" className="home__logo" />
        <h1 className="home__title">Patch</h1>
        <p className="home__subtitle">What are we looking for today?</p>
      </div>

      <form
        className="search-bar"
        onSubmit={(e) => {
          e.preventDefault()
          submitSearch()
        }}
      >
        <textarea
          className="search-bar__input"
          placeholder="i'm looking for my...."
          value={targetText}
          onChange={(e) => updateTargetText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              submitSearch()
            }
          }}
          rows={1}
        />

        {images.length > 0 && (
          <div className="reference-preview">
            {images.map((img, index) => (
              <div key={img.previewUrl} className="reference-preview__item">
                <img src={img.previewUrl} alt="" />
                <button
                  type="button"
                  className="reference-preview__remove"
                  onClick={() => handleRemoveImage(index)}
                  aria-label="Remove photo"
                >
                  <X size={12} color="white" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="search-bar__actions">
          <button
            type="button"
            className="icon-button icon-button--plus"
            aria-label="Add reference photo"
            onClick={handleAddPhotoClick}
            disabled={remainingSlots <= 0}
          >
            <CirclePlus size={45} strokeWidth={1.5} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="visually-hidden"
            onChange={handleFileChange}
          />
          <div className="search-bar__actions-right">
            <button
              type="button"
              className={`icon-button icon-button--circle${isRecording ? ' icon-button--recording' : ''}`}
              aria-label={isRecording ? 'Recording… release to stop' : 'Hold to speak'}
              title={isTranscribing ? 'Transcribing…' : 'Hold to speak'}
              disabled={isTranscribing}
              onMouseDown={beginRecording}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
              onTouchStart={(e) => {
                e.preventDefault()
                beginRecording()
              }}
              onTouchEnd={(e) => {
                e.preventDefault()
                stopRecording()
              }}
            >
              <Mic size={24} color="white" strokeWidth={1.875} />
            </button>
            <button
              type="submit"
              className="icon-button icon-button--circle"
              aria-label="Search"
              disabled={submitting}
            >
              <ArrowUp size={24} color="white" strokeWidth={1.875} />
            </button>
          </div>
        </div>
      </form>

      {clarification && (
        <p className="form-clarify" role="status">
          {clarification}
        </p>
      )}

      {busyPrompt && (
        <div className="busy-prompt" role="alertdialog" aria-label="Patch is already searching">
          <p className="busy-prompt__text">
            {busyPrompt.activeTargetText
              ? `Patch is already looking for "${busyPrompt.activeTargetText}".`
              : 'Patch is already on another search.'}{' '}
            Stop that search and start this one?
          </p>
          <div className="busy-prompt__actions">
            <button
              type="button"
              className="busy-prompt__button busy-prompt__button--primary"
              onClick={() => submitSearch(true)}
              disabled={submitting}
            >
              Stop it and start this one
            </button>
            {busyPrompt.activeSearchId && (
              <button
                type="button"
                className="busy-prompt__button"
                onClick={() => navigate(`/search/${busyPrompt.activeSearchId}`)}
              >
                View current search
              </button>
            )}
            <button type="button" className="busy-prompt__button" onClick={() => setBusyPrompt(null)}>
              Never mind
            </button>
          </div>
        </div>
      )}

      {error && <p className="form-error">{error}</p>}
    </main>
  )
}

export default App
