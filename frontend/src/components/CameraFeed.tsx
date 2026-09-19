import { useState } from 'react'
import './CameraFeed.css'

type FeedStatus = 'loading' | 'loaded' | 'error'

interface CameraFeedProps {
  src?: string
  className?: string
}

function CameraFeed({ src = '/video', className = '' }: CameraFeedProps) {
  const [status, setStatus] = useState<FeedStatus>('loading')

  return (
    <div className={`camera-feed ${className}`.trim()}>
      {status !== 'loaded' && (
        <div className="camera-feed__overlay">
          {status === 'loading' ? 'Connecting to camera…' : 'Camera unavailable'}
        </div>
      )}
      <img
        className="camera-feed__video"
        src={src}
        alt="Live camera feed"
        style={{ display: status === 'error' ? 'none' : 'block' }}
        onLoad={() => setStatus('loaded')}
        onError={() => setStatus('error')}
      />
    </div>
  )
}

export default CameraFeed
