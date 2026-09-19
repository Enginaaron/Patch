import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AlignLeft, ArrowUp, CirclePlus, Mic } from 'lucide-react'
import './App.css'

type ConnectionStatus = 'checking' | 'connected' | 'disconnected'

function App() {
  const [status, setStatus] = useState<ConnectionStatus>('checking')
  const [query, setQuery] = useState('')
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => setStatus(data.status === 'ok' ? 'connected' : 'disconnected'))
      .catch(() => setStatus('disconnected'))
  }, [])

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
        <h1 className="home__title">Patch</h1>
        <p className="home__subtitle">what are we looking for today?</p>
      </div>

      <form
        className="search-bar"
        onSubmit={(e) => {
          e.preventDefault()
        }}
      >
        <textarea
          ref={inputRef}
          className="search-bar__input"
          placeholder="i'm looking for my...."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={1}
        />
        <div className="search-bar__actions">
          <button type="button" className="icon-button icon-button--plus" aria-label="Add reference photo">
            <CirclePlus size={45} strokeWidth={1.5} />
          </button>
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
              disabled={query.trim().length === 0}
            >
              <ArrowUp size={24} color="white" strokeWidth={1.875} />
            </button>
          </div>
        </div>
      </form>
    </main>
  )
}

export default App
