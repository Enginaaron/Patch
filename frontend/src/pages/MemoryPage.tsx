import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft, Search } from 'lucide-react'
import { formatTime } from '../utils/datetime'
import './MemoryPage.css'

interface FindListItem {
  find_id: string
  item_id: string
  item_name: string
  image_url: string
  crop_url: string
  found_at: string
}

type LoadState = 'loading' | 'loaded' | 'error'

function MemoryPage() {
  const navigate = useNavigate()
  const [finds, setFinds] = useState<FindListItem[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [query, setQuery] = useState('')

  useEffect(() => {
    let cancelled = false
    fetch('/api/finds')
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data: FindListItem[]) => {
        if (cancelled) return
        setFinds(data)
        setLoadState('loaded')
      })
      .catch(() => {
        if (!cancelled) setLoadState('error')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const filteredFinds = finds.filter((find) =>
    find.item_name.toLowerCase().includes(query.trim().toLowerCase()),
  )

  return (
    <main className="memory">
      <button type="button" className="memory__back" onClick={() => navigate('/')} aria-label="Back">
        <ArrowLeft size={20} color="white" strokeWidth={2} />
      </button>

      <header className="memory__header">
        <h1 className="memory__title">Memory</h1>
        <div className="memory__view-toggle" aria-label="View mode">
          <span>grid</span>
          <span className="memory__view-toggle-muted"> / list</span>
        </div>
      </header>

      <label className="memory__search">
        <Search size={24} strokeWidth={2} aria-hidden="true" />
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="search"
          aria-label="Search memory"
        />
      </label>

      {loadState === 'loading' && <p className="memory__notice">Loading…</p>}
      {loadState === 'error' && <p className="memory__notice">Couldn't load memory.</p>}
      {loadState === 'loaded' && filteredFinds.length === 0 && (
        <p className="memory__notice">Nothing found yet.</p>
      )}

      {filteredFinds.length > 0 && (
        <ul className="memory__grid">
          {filteredFinds.map((find) => (
            <li key={find.find_id}>
              <Link to={`/items/${find.item_id}`} className="memory__card">
                <img src={find.crop_url} alt="" className="memory__card-photo" />
                <span className="memory__card-caption">
                  {formatTime(find.found_at)} - {find.item_name}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  )
}

export default MemoryPage
