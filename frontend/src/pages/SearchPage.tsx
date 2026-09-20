import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Image, Plus, Search } from 'lucide-react'
import CameraFeed from '../components/CameraFeed'
import ChatPanel from '../components/ChatPanel'
import ControlPad from '../components/ControlPad'
import './SearchPage.css'

interface SearchNavState {
  targetText?: string
  previewUrl?: string
}

function SearchPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { searchId } = useParams()
  const { targetText, previewUrl } = (location.state as SearchNavState | null) ?? {}
  const displayText = targetText ?? 'your item'

  return (
    <main className="live-screen">
      <CameraFeed className="camera-feed--full" />

      <button type="button" className="live-screen__back" onClick={() => navigate('/')}>
        <ArrowLeft size={20} color="white" strokeWidth={2} />
        Back
      </button>

      <div className="live-screen__popup">
        <p className="live-screen__popup-question">Is this "{displayText}"?</p>
        <div className="live-screen__popup-actions">
          <button type="button" className="popup-button popup-button--yes" disabled title="Candidate confirmation isn't available yet">
            Yes
          </button>
          <button type="button" className="popup-button popup-button--no" disabled title="Candidate confirmation isn't available yet">
            No
          </button>
        </div>
      </div>

      <div className="live-screen__card">
        <p className="live-screen__label">You're looking for:</p>
        <div className="live-screen__target-pill">
          <Search size={16} />
          <span>{displayText}</span>
        </div>

        <div className="live-screen__photos">
          <div className="live-screen__photo-slot">
            {previewUrl ? (
              <img src={previewUrl} alt="" />
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

      <ControlPad searchId={searchId} targetText={targetText} />
      <ChatPanel targetText={targetText} />
    </main>
  )
}

export default SearchPage
