import { useEffect, useRef, useState } from 'react'
import type { ReconstructionEvent } from './types'

/** The car's front and rear (metres from the barrier face, + towards the camera's right side of travel) at time t,
 *  interpolated between recorded frames; a missing edge is the other one ± the car's length. */
function edgesAt(r: ReconstructionEvent, t: number): [number, number] | null {
  const f = r.frames
  if (!f.length || t < f[0][0]) return null
  let i = 0
  while (i < f.length - 2 && f[i + 1][0] <= t) i++
  const a = f[i], b = f[Math.min(i + 1, f.length - 1)]
  const k = b[0] > a[0] ? Math.min(Math.max((t - a[0]) / (b[0] - a[0]), 0), 1) : 0
  const lerp = (p: number | null, q: number | null) => (p === null ? q : q === null ? p : p + (q - p) * k)
  let front = lerp(a[1], b[1]), rear = lerp(a[2], b[2])
  if (front === null && rear === null) return null
  if (front === null) front = (rear as number) - r.length_m
  if (rear === null) rear = (front as number) + r.length_m
  return [front as number, Math.max(rear as number, (front as number) + 0.3 * r.length_m)]
}

/** A car's front behaves roughly like a spring: for the same car, crush and average deceleration scale with impact
 *  speed, and the crash energy with its square. Used only for the what-if, and labelled as a model. */
function ghostAt(r: ReconstructionEvent, t: number, v2: number, crush: number): [number, number] | null {
  const v = r.speed_ms, L = r.length_m
  if (!crush || !v) return null
  const d2 = crush * (v2 / v)
  const w = v / crush                                   // same stiffness: ω = v/d for the real crash
  const dt = t - r.impact_t
  const e = r.rebound_ms ? r.rebound_ms / v : 0.1
  if (dt < 0) { const front = v2 * -dt; return [front, front + L] }
  const tMax = Math.PI / 2 / w
  if (dt <= tMax) { const c = d2 * Math.sin(w * dt); return [0, L - c] }
  const back = e * v2 * (dt - tMax)
  return [back, back + L - 0.85 * d2]
}

function drawCar(g: CanvasRenderingContext2D, x0: number, x1: number, ground: number, hPx: number, color: string, ghost: boolean) {
  const len = x1 - x0
  g.save()
  g.globalAlpha = ghost ? 0.55 : 1
  g.strokeStyle = color
  g.fillStyle = ghost ? 'transparent' : color + '33'
  g.lineWidth = 2
  if (ghost) g.setLineDash([6, 5])
  // Body: a low box with a cabin.
  const body = hPx * 0.55, cabinH = hPx * 0.45
  g.beginPath(); g.roundRect(x0, ground - hPx * 0.18 - body, len, body, 8); g.fill(); g.stroke()
  g.beginPath()
  g.moveTo(x0 + len * 0.22, ground - hPx * 0.18 - body)
  g.lineTo(x0 + len * 0.36, ground - hPx * 0.18 - body - cabinH)
  g.lineTo(x0 + len * 0.82, ground - hPx * 0.18 - body - cabinH)
  g.lineTo(x0 + len * 0.95, ground - hPx * 0.18 - body)
  g.stroke()
  const wr = hPx * 0.19
  for (const fx of [0.2, 0.8]) { g.beginPath(); g.arc(x0 + len * fx, ground - wr, wr, 0, Math.PI * 2); g.stroke() }
  g.restore()
}

export default function ReconstructionPanel({ recon, time }: { recon: ReconstructionEvent; time: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [whatKmh, setWhatKmh] = useState<number | null>(null)
  const realKmh = recon.speed_ms * 3.6
  const crush = recon.crush_m || 0

  useEffect(() => {
    const c = ref.current
    if (!c) return
    const dpr = window.devicePixelRatio || 1
    const W = c.clientWidth, H = c.clientHeight
    if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr }
    const g = c.getContext('2d')!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    g.clearRect(0, 0, W, H)
    const span = recon.length_m * 1.9
    const ppm = Math.min((W * 0.86) / span, (H - 60) / (recon.height_m * 1.15))   // fit the length and the height
    const X = (m: number) => W * 0.1 + m * ppm
    const ground = H - 28
    const hPx = recon.height_m * ppm

    // Floor with metre ticks from the barrier face.
    g.strokeStyle = 'rgba(233, 238, 245, 0.35)'; g.lineWidth = 1
    g.beginPath(); g.moveTo(0, ground); g.lineTo(W, ground); g.stroke()
    g.fillStyle = 'rgba(233, 238, 245, 0.5)'; g.font = '10px Inter, system-ui, sans-serif'
    for (let m = 0; m <= span; m++) { g.fillRect(X(m), ground, 1, 5); g.fillText(`${m} m`, X(m) - 8, ground + 17) }
    // The barrier.
    g.fillStyle = 'rgba(170, 180, 195, 0.25)'; g.fillRect(X(-0.6), ground - hPx * 1.1, X(0) - X(-0.6), hPx * 1.1)
    g.strokeStyle = 'rgba(200, 210, 225, 0.8)'; g.strokeRect(X(-0.6), ground - hPx * 1.1, X(0) - X(-0.6), hPx * 1.1)
    for (let k = 0; k < 6; k++) { g.beginPath(); g.moveTo(X(-0.6), ground - k * hPx * 0.2); g.lineTo(X(0), ground - k * hPx * 0.2 - hPx * 0.15); g.stroke() }
    g.fillStyle = 'rgba(233, 238, 245, 0.7)'; g.fillText('rigid barrier', X(-0.6), ground - hPx * 1.1 - 6)

    // What-if ghost (behind the real car).
    if (whatKmh !== null) {
      const gh = ghostAt(recon, time, whatKmh / 3.6, crush)
      if (gh) drawCar(g, X(gh[0]), X(gh[1]), ground, hPx, '#ff5ad1', true)
    }
    // The real car, as recorded.
    const e = edgesAt(recon, time)
    if (e) {
      const [front, rear] = e
      const crushed = Math.max(0, recon.length_m - (rear - front))
      drawCar(g, X(front), X(rear), ground, hPx, '#3ee6ff', false)
      if (crushed > 0.03) {
        g.fillStyle = 'rgba(255, 93, 115, 0.45)'
        g.fillRect(X(front), ground - hPx * 0.75, crushed * ppm, hPx * 0.57)
        g.fillStyle = '#ff8a9a'; g.font = '600 11px Inter, system-ui, sans-serif'
        g.fillText(`crushed ${crushed.toFixed(2)} m`, X(front) + 4, ground - hPx * 0.8)
      }
      // Speed from the recorded motion around now.
      const e2 = edgesAt(recon, time + 0.02)
      const v = e2 ? Math.abs(e2[1] - rear) / 0.02 : 0
      g.fillStyle = '#e8f6ff'; g.font = '600 22px Inter, system-ui, sans-serif'
      g.fillText(`${(v * 3.6).toFixed(0)} km/h`, W - 140, 34)
      g.font = '11px Inter, system-ui, sans-serif'; g.fillStyle = 'rgba(233, 238, 245, 0.6)'
      g.fillText('real car (video)', W - 140, 50)
    }
    // Impact flash.
    const dt = time - recon.impact_t
    if (dt >= 0 && dt < 0.25) {
      const k = 1 - dt / 0.25
      g.strokeStyle = `rgba(255, 200, 87, ${k})`; g.lineWidth = 3
      for (let i = 0; i < 10; i++) {
        const a = (i / 10) * Math.PI - Math.PI / 2, r0 = 10, r1 = 18 + 50 * (1 - k)
        g.beginPath(); g.moveTo(X(0) + Math.cos(a) * r0, ground - hPx * 0.4 + Math.sin(a) * r0)
        g.lineTo(X(0) + Math.cos(a) * r1, ground - hPx * 0.4 + Math.sin(a) * r1); g.stroke()
      }
      g.fillStyle = `rgba(255, 200, 87, ${k})`; g.font = '700 14px Inter, system-ui, sans-serif'
      g.fillText(`IMPACT · ${realKmh.toFixed(1)} km/h`, X(0) + 12, 22)
    }
  }, [recon, time, whatKmh, crush, realKmh])

  const ratio = whatKmh !== null ? whatKmh / realKmh : 1
  return (
    <div className="recon">
      <div className="recon-head">
        <span className="label">Reconstruction <span className="knowledge rec">from the record, to scale</span></span>
        <label className="recon-slider">
          What if it had been going
          <input type="range" min={10} max={100} step={1} value={whatKmh ?? Math.round(realKmh)}
            onChange={(e) => setWhatKmh(Number(e.target.value))} />
          <b>{whatKmh ?? Math.round(realKmh)} km/h</b>
          {whatKmh !== null && <button className="chip" onClick={() => setWhatKmh(null)}>off</button>}
        </label>
      </div>
      <canvas ref={ref} />
      {whatKmh !== null && crush > 0 && (
        <div className="recon-note">
          At {whatKmh} km/h: crush ≈ {(crush * ratio).toFixed(2)} m, average deceleration ≈ {(((recon.speed_ms * recon.speed_ms) / crush / 9.81) * (2 / Math.PI) * ratio).toFixed(0)} g,
          crash energy {(ratio * ratio * 100).toFixed(0)}% of the real one.
          <span className="knowledge"> model: the car's front acts like a spring (same car, same barrier)</span>
        </div>
      )}
    </div>
  )
}
