import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Mic } from 'lucide-react'
import useVoice from '../hooks/useVoice'
import { formatDate, formatTime } from '../utils/datetime'
import './ItemPage.css'

interface ItemReferenceImage {
  id: string
  url: string
}

interface ItemFind {
  find_id: string
  image_url: string
  crop_url: string
  found_at: string
}

interface ItemDetail {
  item_id: string
  name: string
  reference_images: ItemReferenceImage[]
  finds: ItemFind[]
}

interface QuickCheckResult {
  visible: boolean
  checked_at: string
}

type LoadState = 'loading' | 'loaded' | 'not_found' | 'error'

function ItemPage() {
  const navigate = useNavigate()
  const { itemId } = useParams<{ itemId: string }>()
  const [item, setItem] = useState<ItemDetail | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [findingAgain, setFindingAgain] = useState(false)
  const [checking, setChecking] = useState(false)
  const [quickCheckResult, setQuickCheckResult] = useState<QuickCheckResult | null>(null)
  const [quickCheckError, setQuickCheckError] = useState<string | null>(null)

  useEffect(() => {
    if (!itemId) return
    let cancelled = false

    fetch(`/api/items/${itemId}`)
      .then((res) => {
        if (res.status === 404) throw new Error('not_found')
        if (!res.ok) throw new Error('error')
        return res.json() as Promise<ItemDetail>
      })
      .then((data) => {
        if (cancelled) return
        setItem(data)
        setLoadState('loaded')
      })
      .catch((err) => {
        if (cancelled) return
        setLoadState(err instanceof Error && err.message === 'not_found' ? 'not_found' : 'error')
      })

    return () => {
      cancelled = true
    }
  }, [itemId])

  const latest = item?.finds[0] ?? null
  const hasReferencePhotos = (item?.reference_images.length ?? 0) > 0

  const handleFindAgain = async () => {
    if (!itemId || findingAgain) return
    setFindingAgain(true)
    try {
      const res = await fetch(`/api/items/${itemId}/search`, { method: 'POST' })
      if (!res.ok) throw new Error('failed to start search')
      const data = (await res.json()) as { search_id: string }
      navigate(`/search/${data.search_id}`)
    } catch {
      setFindingAgain(false)
    }
  }

  // Spec 19: voice-first, text-first -- the on-screen result is set the
  // instant this (the vision call) returns. speak() below is fire-and-forget
  // (Spec 8), so it never delays that update; it's a follow-on, not
  // something the UI waits on.
  const runQuickCheck = async () => {
    if (!itemId || !hasReferencePhotos || checking) return
    setChecking(true)
    setQuickCheckError(null)
    try {
      const res = await fetch(`/api/items/${itemId}/quick-check`, { method: 'POST' })
      if (!res.ok) {
        const body = await res.json().catch(() => null)
        throw new Error(body?.detail ?? 'Quick check failed')
      }
      const data = (await res.json()) as QuickCheckResult
      setQuickCheckResult(data)
      speak(data.visible ? 'Yes, still there' : 'Not visible right now')
    } catch (err) {
      setQuickCheckError(err instanceof Error ? err.message : 'Quick check failed')
    } finally {
      setChecking(false)
    }
  }

  // Spec 19: this mic is scoped to this item's page -- any transcript
  // triggers the check for this item, no keyword parsing needed since the
  // page context already says which item "check my keys" means.
  const { isRecording, isTranscribing, startRecording, stopRecording, speak } = useVoice({
    onTranscript: () => {
      runQuickCheck()
    },
  })

  return (
    <main className="item-page">
      <button type="button" className="item-page__back" onClick={() => navigate('/memory')} aria-label="Back">
        <ArrowLeft size={20} color="white" strokeWidth={2} />
      </button>

      {loadState === 'loading' && <p className="item-page__notice">Loading…</p>}
      {loadState === 'not_found' && <p className="item-page__notice">Item not found.</p>}
      {loadState === 'error' && <p className="item-page__notice">Couldn't load this item.</p>}

      {loadState === 'loaded' && item && (
        <>
          <h1 className="item-page__title">{item.name}</h1>

          {latest && <img src={latest.crop_url} alt="" className="item-page__photo" />}

          {latest && (
            <p className="item-page__last-found">
              Last found&nbsp;&nbsp;{formatTime(latest.found_at)} · {formatDate(latest.found_at)}
            </p>
          )}

          {item.reference_images.length > 0 && (
            <div className="item-page__section">
              <p className="item-page__section-label">Reference photos</p>
              <div className="item-page__reference-photos">
                {item.reference_images.map((ref) => (
                  <img key={ref.id} src={ref.url} alt="" className="item-page__reference-photo" />
                ))}
              </div>
            </div>
          )}

          <div className="item-page__section">
            <p className="item-page__section-label">Previous Finds</p>
            {item.finds.length === 0 ? (
              <p className="item-page__notice">No finds yet.</p>
            ) : (
              <ul className="item-page__finds-list">
                {item.finds.map((find) => (
                  <li key={find.find_id} className="item-page__finds-row">
                    <img src={find.crop_url} alt="" className="item-page__finds-photo" />
                    <span className="item-page__finds-time">
                      {formatTime(find.found_at)} · {formatDate(find.found_at)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {quickCheckError && <p className="item-page__notice item-page__notice--error">{quickCheckError}</p>}

          {/* Spec 19: text is shown the instant the vision call returns --
              the "checked just now" copy is literal, not a live timestamp,
              since the result is always fresh at the moment it renders. */}
          {quickCheckResult && (
            <div
              className={`quick-check-result${quickCheckResult.visible ? '' : ' quick-check-result--negative'}`}
            >
              {quickCheckResult.visible ? (
                <>
                  <p className="quick-check-result__status">✓ Still there</p>
                  <p className="quick-check-result__meta">Checked just now</p>
                </>
              ) : (
                <>
                  <p className="quick-check-result__status">Not visible right now</p>
                  <button
                    type="button"
                    className="quick-check-result__search-button"
                    onClick={handleFindAgain}
                    disabled={findingAgain}
                  >
                    {findingAgain ? 'STARTING…' : 'SEARCH FOR IT'}
                  </button>
                </>
              )}
            </div>
          )}

          {!hasReferencePhotos && (
            <p className="item-page__notice">
              Quick Check needs at least one reference photo -- teach this item first.
            </p>
          )}

          <div className="item-page__actions">
            {hasReferencePhotos && (
              <button
                type="button"
                className={`item-page__mic${isRecording ? ' item-page__mic--recording' : ''}`}
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
                disabled={checking || isTranscribing}
                aria-label="Hold to quick check by voice"
                title="Hold and say something, e.g. &quot;check my keys&quot;"
              >
                <Mic size={18} color="white" strokeWidth={1.875} />
              </button>
            )}
            <button
              type="button"
              className="item-page__action"
              onClick={handleFindAgain}
              disabled={findingAgain}
            >
              {findingAgain ? 'STARTING…' : 'FIND AGAIN'}
            </button>
            <button
              type="button"
              className="item-page__action"
              onClick={runQuickCheck}
              disabled={!hasReferencePhotos || checking}
              title={hasReferencePhotos ? undefined : 'Teach this item with a reference photo first'}
            >
              {checking ? 'CHECKING…' : 'QUICK CHECK'}
            </button>
          </div>
        </>
      )}
    </main>
  )
}

export default ItemPage
