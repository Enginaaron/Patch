import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Image, Plus, Search } from 'lucide-react'
import CameraFeed from '../components/CameraFeed'
import './SearchPage.css'

interface ReferenceImageOut {
  id: string
  url: string
}

interface SearchDetail {
  search_id: string
  target_text: string
  status: string
  reference_images: ReferenceImageOut[]
}

type LoadState = 'loading' | 'loaded' | 'not_found' | 'error'

const STATUS_LABEL: Record<string, string> = {
  SEARCHING: 'Searching…',
  CANDIDATE_PENDING: 'Reviewing a match…',
  FOUND: 'Found',
  CANCELLED: 'Cancelled',
}

function SearchPage() {
  const navigate = useNavigate()
  const { searchId } = useParams<{ searchId: string }>()
  const [search, setSearch] = useState<SearchDetail | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [cancelling, setCancelling] = useState(false)

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

  const displayText = search?.target_text ?? '…'
  const referenceUrl = search?.reference_images[0]?.url
  const statusLabel = search ? (STATUS_LABEL[search.status] ?? search.status) : ''

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
          <div className="live-screen__popup">
            <p className="live-screen__popup-question">Is this "{displayText}"?</p>
            <div className="live-screen__popup-actions">
              <button
                type="button"
                className="popup-button popup-button--yes"
                disabled
                title="Candidate confirmation isn't available yet"
              >
                Yes
              </button>
              <button
                type="button"
                className="popup-button popup-button--no"
                disabled
                title="Candidate confirmation isn't available yet"
              >
                No
              </button>
            </div>
          </div>

          <div className="live-screen__bottom">
            <div className="live-screen__card">
              <div className="live-screen__status">
                <span
                  className={`live-screen__status-dot live-screen__status-dot--${search.status.toLowerCase()}`}
                />
                {statusLabel}
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
        </>
      )}
    </main>
  )
}

export default SearchPage
