import { useRef, useState } from 'react'
import BoundingBoxOverlay from '../components/BoundingBoxOverlay'

// Isolated test harness for Spec 10, same principle as Specs 7/8's debug
// scripts: verifies the overlay aligns correctly before it's ever wired to a
// real detection. Renders the same image + box across several fixed
// container sizes (phone/tablet/desktop widths) and both object-fit modes
// side by side, so alignment across sizes is visible in one screenshot
// instead of requiring the browser window to actually be resized.
const SIZES = [
  { label: 'Phone (320x240)', width: 320, height: 240 },
  { label: 'Tablet (600x400)', width: 600, height: 400 },
  { label: 'Desktop (900x300)', width: 900, height: 300 },
]

// Known-good box from Spec 7 testing: bounds the silver "CLE USA" key in
// dev-bbox-test.jpg, verified against the backend crop in Spec 10.
const DEFAULT_BOX: [number, number, number, number] = [284, 261, 793, 535]

function Preview({
  label,
  width,
  height,
  box,
  objectFit,
}: {
  label: string
  width: number
  height: number
  box: [number, number, number, number]
  objectFit: 'cover' | 'contain'
}) {
  const imgRef = useRef<HTMLImageElement>(null)

  return (
    <div style={{ marginBottom: 12 }}>
      <p style={{ margin: '0 0 4px', fontSize: 13, fontFamily: 'monospace' }}>
        {label} — object-fit: {objectFit}
      </p>
      <div style={{ position: 'relative', width, height, background: '#111', overflow: 'hidden' }}>
        <img
          ref={imgRef}
          src="/dev-bbox-test.jpg"
          alt=""
          style={{ width: '100%', height: '100%', objectFit, display: 'block' }}
        />
        <BoundingBoxOverlay box2d={box} mediaRef={imgRef} objectFit={objectFit} label="silver key" />
      </div>
    </div>
  )
}

function DevBBoxPage() {
  const [box, setBox] = useState<[number, number, number, number]>(DEFAULT_BOX)

  const updateBox = (index: number, value: number) => {
    setBox((prev) => {
      const next = [...prev] as [number, number, number, number]
      next[index] = value
      return next
    })
  }

  return (
    <main style={{ maxWidth: 1000, margin: '0 auto', padding: 24, fontFamily: 'system-ui' }}>
      <h1>Bounding box overlay dev test (Spec 10)</h1>

      <section style={{ marginBottom: 24 }}>
        <h2>box_2d (0-1000 scale): [ymin, xmin, ymax, xmax]</h2>
        {(['ymin', 'xmin', 'ymax', 'xmax'] as const).map((name, i) => (
          <label key={name} style={{ marginRight: 12 }}>
            {name}:{' '}
            <input
              type="number"
              min={0}
              max={1000}
              value={box[i]}
              onChange={(e) => updateBox(i, Number(e.target.value))}
              style={{ width: 70 }}
            />
          </label>
        ))}
      </section>

      {SIZES.map((size) => (
        <div key={size.label} style={{ display: 'flex', gap: 24 }}>
          <Preview label={size.label} width={size.width} height={size.height} box={box} objectFit="cover" />
          <Preview label={size.label} width={size.width} height={size.height} box={box} objectFit="contain" />
        </div>
      ))}
    </main>
  )
}

export default DevBBoxPage
