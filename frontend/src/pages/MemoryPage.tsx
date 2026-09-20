import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { dayLabel, formatTime } from '../utils/datetime'
import './MemoryPage.css'

interface FindListItem {
  find_id: string
  item_id: string
  item_name: string
  image_url: string
  crop_url: string
  found_at: string
}

interface FindGroup {
  label: string
  items: FindListItem[]
}

type LoadState = 'loading' | 'loaded' | 'error'

function groupByDay(finds: FindListItem[]): FindGroup[] {
  const groups: FindGroup[] = []
  for (const find of finds) {
    const label = dayLabel(find.found_at)
    const current = groups[groups.length - 1]
    if (current && current.label === label) {
      current.items.push(find)
    } else {
      groups.push({ label, items: [find] })
    }
  }
  return groups
}

function MemoryPage() {
  const navigate = useNavigate()
  const [finds, setFinds] = useState<FindListItem[]>([])
  const [loadState, setLoadState] = useState<LoadState>('loading')

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

  const groups = groupByDay(finds)

  return (
    <main className="memory">
      <button type="button" className="memory__back" onClick={() => navigate('/')} aria-label="Back">
        <ArrowLeft size={20} color="white" strokeWidth={2} />
      </button>

      <h1 className="memory__title">MEMORY</h1>

      {loadState === 'loading' && <p className="memory__notice">Loading…</p>}
      {loadState === 'error' && <p className="memory__notice">Couldn't load memory.</p>}
      {loadState === 'loaded' && finds.length === 0 && (
        <p className="memory__notice">Nothing found yet.</p>
      )}

      {groups.map((group) => (
        <section key={group.label} className="memory__group">
          <h2 className="memory__group-label">{group.label}</h2>
          <ul className="memory__list">
            {group.items.map((find) => (
              <li key={find.find_id}>
                <Link to={`/items/${find.item_id}`} className="memory__row">
                  <img src={find.crop_url} alt="" className="memory__row-photo" />
                  <span className="memory__row-name">{find.item_name}</span>
                  <span className="memory__row-time">{formatTime(find.found_at)}</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </main>
  )
}

export default MemoryPage
