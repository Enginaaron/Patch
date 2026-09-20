import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AlignLeft, ArrowUp, CirclePlus, Mic, X } from 'lucide-react'
import logoMark from './assets/logo-mark.svg'
import './App.css'

type ConnectionStatus = 'checking' | 'connected' | 'disconnected'

const MAX_REFERENCE_IMAGES = 3

interface ReferenceImageSlot {
  file: File
  previewUrl: string
}

function App() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const [targetText, setTargetText] = useState('')
  const [images, setImages] = useState<ReferenceImageSlot[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  const submitSearch = async () => {
    const text = targetText.trim()
    if (!text || submitting) return

    setSubmitting(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('target_text', text)
      images.forEach((img) => formData.append('reference_images', img.file))

      const res = await fetch('/api/searches', { method: 'POST', body: formData })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail ?? 'Failed to start search')
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
          onChange={(e) => setTargetText(e.target.value)}
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
              className="icon-button icon-button--circle"
              aria-label="Voice input"
              disabled
              title="Voice input isn't available yet"
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

      {error && <p className="form-error">{error}</p>}
    </main>
  )
}

export default App
