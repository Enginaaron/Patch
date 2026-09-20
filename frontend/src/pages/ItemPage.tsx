import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
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

type LoadState = 'loading' | 'loaded' | 'not_found' | 'error'

function ItemPage() {
  const navigate = useNavigate()
  const { itemId } = useParams<{ itemId: string }>()
  const [item, setItem] = useState<ItemDetail | null>(null)
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [findingAgain, setFindingAgain] = useState(false)

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

          <div className="item-page__actions">
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
              disabled
              title="Quick check isn't wired up yet"
            >
              QUICK CHECK
            </button>
          </div>
        </>
      )}
    </main>
  )
}

export default ItemPage
