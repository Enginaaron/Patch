import { useEffect, useState } from 'react'
import type { RefObject } from 'react'

type Box2d = [number, number, number, number] // [ymin, xmin, ymax, xmax], 0-1000 scale

interface BoundingBoxOverlayProps {
  box2d: Box2d
  mediaRef: RefObject<HTMLImageElement | HTMLVideoElement | null>
  objectFit?: 'cover' | 'contain'
  color?: string
  label?: string
}

interface RenderedRect {
  left: number
  top: number
  width: number
  height: number
}

function getNaturalSize(el: HTMLImageElement | HTMLVideoElement): { width: number; height: number } | null {
  if (el instanceof HTMLVideoElement) {
    if (!el.videoWidth || !el.videoHeight) return null
    return { width: el.videoWidth, height: el.videoHeight }
  }
  if (!el.naturalWidth || !el.naturalHeight) return null
  return { width: el.naturalWidth, height: el.naturalHeight }
}

// Renders a box positioned over an <img>/<video> at its CURRENT RENDERED
// size, never burning it into the source. The tricky part: when the element
// uses object-fit (cover/contain), the visible content is scaled and
// possibly cropped relative to its own box, so naive percentage positioning
// misaligns as soon as the container's aspect ratio differs from the media's
// natural aspect ratio. This replicates the actual object-fit math to find
// where the natural-pixel box lands in the rendered box.
//
// Usage: place this as a sibling of the <img>/<video> inside a
// position:relative wrapper that matches the media element's box.
function BoundingBoxOverlay({ box2d, mediaRef, objectFit = 'cover', color = '#2ecc71', label }: BoundingBoxOverlayProps) {
  const [rect, setRect] = useState<RenderedRect | null>(null)

  useEffect(() => {
    const el = mediaRef.current
    if (!el) return

    const update = () => {
      const natural = getNaturalSize(el)
      const containerWidth = el.clientWidth
      const containerHeight = el.clientHeight
      if (!natural || !containerWidth || !containerHeight) {
        setRect(null)
        return
      }

      const scale =
        objectFit === 'cover'
          ? Math.max(containerWidth / natural.width, containerHeight / natural.height)
          : Math.min(containerWidth / natural.width, containerHeight / natural.height)

      const renderedWidth = natural.width * scale
      const renderedHeight = natural.height * scale
      const offsetX = (containerWidth - renderedWidth) / 2
      const offsetY = (containerHeight - renderedHeight) / 2

      const [ymin, xmin, ymax, xmax] = box2d
      const nx1 = (xmin / 1000) * natural.width
      const ny1 = (ymin / 1000) * natural.height
      const nx2 = (xmax / 1000) * natural.width
      const ny2 = (ymax / 1000) * natural.height

      setRect({
        left: offsetX + nx1 * scale,
        top: offsetY + ny1 * scale,
        width: (nx2 - nx1) * scale,
        height: (ny2 - ny1) * scale,
      })
    }

    update()

    const resizeObserver = new ResizeObserver(update)
    resizeObserver.observe(el)
    el.addEventListener('loadedmetadata', update)
    el.addEventListener('load', update)

    return () => {
      resizeObserver.disconnect()
      el.removeEventListener('loadedmetadata', update)
      el.removeEventListener('load', update)
    }
  }, [box2d, mediaRef, objectFit])

  if (!rect) return null

  return (
    <div
      style={{
        position: 'absolute',
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        border: `2px solid ${color}`,
        borderRadius: 4,
        pointerEvents: 'none',
        boxSizing: 'border-box',
      }}
    >
      {label && (
        <span
          style={{
            position: 'absolute',
            top: -22,
            left: 0,
            background: color,
            color: 'white',
            fontSize: 12,
            padding: '2px 6px',
            borderRadius: 3,
            whiteSpace: 'nowrap',
          }}
        >
          {label}
        </span>
      )}
    </div>
  )
}

export default BoundingBoxOverlay
