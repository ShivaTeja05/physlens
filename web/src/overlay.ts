import type { Path, StageState } from './derive'
import type { PredictionEvent } from './types'

export type View = { scale: number; ox: number; oy: number; pxPerM: number; vw: number; vh: number }

const CYAN = '#3ee6ff'
const HIT = '#4ade80'
const MISS = '#ff5d73'
const GHOST = 'rgba(190, 220, 255, 0.85)'

// Tracker coordinates put pixel centres on integers; on screen pixel i spans [i, i+1), so add half a pixel.
const toCanvas = (v: View, x: number, y: number): [number, number] => [v.ox + (x + 0.5) * v.scale, v.oy + (y + 0.5) * v.scale]

/** Slow → cool cyan, fast → hot coral. */
function speedColor(speed: number, alpha: number) {
  const k = Math.min(speed / 4.5, 1)
  const hue = 190 - 180 * k
  return `hsla(${hue}, 95%, 62%, ${alpha})`
}

function arrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: string, width = 2.5) {
  const ang = Math.atan2(y1 - y0, x1 - x0)
  const head = 9 + width * 1.5
  ctx.strokeStyle = color
  ctx.fillStyle = color
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.moveTo(x0, y0)
  ctx.lineTo(x1 - Math.cos(ang) * head * 0.6, y1 - Math.sin(ang) * head * 0.6)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x1, y1)
  ctx.lineTo(x1 - head * Math.cos(ang - 0.42), y1 - head * Math.sin(ang - 0.42))
  ctx.lineTo(x1 - head * Math.cos(ang + 0.42), y1 - head * Math.sin(ang + 0.42))
  ctx.closePath()
  ctx.fill()
}

function pill(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = '500 13px Inter, system-ui, sans-serif'
  const w = ctx.measureText(text).width + 18
  ctx.fillStyle = 'rgba(8, 12, 18, 0.72)'
  ctx.strokeStyle = color
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.roundRect(x, y - 13, w, 26, 13)
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = '#e8f6ff'
  ctx.fillText(text, x + 9, y + 4.5)
}

/** Each tracked object gets its own colour; single-object experiments keep the signature cyan. */
const OBJECT_COLORS: Record<string, string> = { red: '#ff6b7d', blue: '#5aa9ff', ref: '#4ade80', mystery: '#ff6b7d' }
const objectColor = (id: string) => OBJECT_COLORS[id] ?? CYAN

function reticle(ctx: CanvasRenderingContext2D, x: number, y: number, tone: string, r = 16) {
  ctx.strokeStyle = tone
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.moveTo(x - r - 10, y); ctx.lineTo(x - r + 8, y)
  ctx.moveTo(x + r - 8, y); ctx.lineTo(x + r + 10, y)
  ctx.moveTo(x, y - r - 10); ctx.lineTo(x, y - r + 8)
  ctx.stroke()
}

/** A tape measure on the floor from the table edge, a tick every 5 cm and a number every 10. */
function tape(ctx: CanvasRenderingContext2D, x0: number, x1: number, y: number, cmToCanvas: number) {
  const dir = Math.sign(x1 - x0) || 1
  const len = Math.abs(x1 - x0) + 12 * cmToCanvas
  ctx.save()
  ctx.fillStyle = 'rgba(255, 200, 87, 0.16)'
  ctx.fillRect(Math.min(x0, x0 + dir * len), y, len, 12)
  ctx.strokeStyle = 'rgba(255, 200, 87, 0.75)'
  ctx.fillStyle = 'rgba(255, 220, 150, 0.9)'
  ctx.lineWidth = 1
  ctx.font = '500 9px Inter, system-ui, sans-serif'
  ctx.textAlign = 'center'
  for (let cm = 0; cm * cmToCanvas <= len; cm += 5) {
    const x = x0 + dir * cm * cmToCanvas
    const major = cm % 10 === 0
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x, y + (major ? 7 : 4))
    ctx.stroke()
    if (major && cmToCanvas * 10 > 18) ctx.fillText(String(cm), x, y + 20)
  }
  ctx.restore()
}

/** Chequered finish line through the predicted finish points, extended past both lanes. */
function finishLine(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number]) {
  const len = Math.hypot(b[0] - a[0], b[1] - a[1])
  if (len < 1) return
  const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len
  const ext = 34, sq = 6
  const nx = -uy, ny = ux
  ctx.save()
  for (let d = -ext, k = 0; d < len + ext; d += sq, k++) {
    for (const row of [0, 1]) {
      ctx.fillStyle = (k + row) % 2 ? 'rgba(255, 255, 255, 0.85)' : 'rgba(10, 12, 16, 0.85)'
      const cx = a[0] + ux * d + nx * (row - 1) * sq, cy = a[1] + uy * d + ny * (row - 1) * sq
      ctx.beginPath()
      ctx.moveTo(cx, cy)
      ctx.lineTo(cx + ux * sq, cy + uy * sq)
      ctx.lineTo(cx + ux * sq + nx * sq, cy + uy * sq + ny * sq)
      ctx.lineTo(cx + nx * sq, cy + ny * sq)
      ctx.closePath()
      ctx.fill()
    }
  }
  ctx.restore()
  const top = a[1] < b[1] ? a : b
  pill(ctx, 'FINISH', top[0] - ux * ext * Math.sign(uy || 1) - 30, top[1] - 26, 'rgba(255, 255, 255, 0.7)')
}

function bandFill(verdictHit: boolean | undefined) {
  if (verdictHit === undefined) return 'rgba(62, 230, 255, 0.16)'
  return verdictHit ? 'rgba(74, 222, 128, 0.22)' : 'rgba(255, 93, 115, 0.22)'
}

function drawPrediction(ctx: CanvasRenderingContext2D, v: View, p: PredictionEvent, s: StageState, time: number) {
  if (!p.ghost || p.ghost.length < 2) return
  const verdict = s.verdicts[p.id]
  const tone = verdict ? (verdict.hit ? HIT : MISS) : p.object ? objectColor(p.object) : GHOST
  const fmt = (x: number) => (p.unit === 's' ? `${x.toFixed(3)} s` : `${x.toFixed(1)} ${p.unit ?? ''}`)

  // Predicted path.
  ctx.save()
  ctx.setLineDash([6, 7])
  ctx.lineWidth = 2
  ctx.strokeStyle = verdict ? tone : p.object ? objectColor(p.object) + 'aa' : 'rgba(190, 220, 255, 0.55)'
  ctx.beginPath()
  p.ghost.forEach(([, x, y], i) => {
    const [cx, cy] = toCanvas(v, x, y)
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy)
  })
  ctx.stroke()
  ctx.restore()

  const cmToCanvas = (v.pxPerM / 100) * v.scale
  const label = verdict ? `${verdict.hit ? '✓' : '✗'} ${fmt(verdict.actual)}` : fmt(p.value)
  if (p.quantity === 'peak_height') {
    // Bounce: reticle on the apex with the 90% band in height.
    let apex = p.ghost[0]
    for (const g of p.ghost) if (g[2] < apex[2]) apex = g
    const [ax, ay] = toCanvas(v, apex[1], apex[2])
    const top = ay - (p.ci90[1] - p.value) * cmToCanvas
    const bottom = ay + (p.value - p.ci90[0]) * cmToCanvas
    const half = Math.max((bottom - top) / 2, 3)
    ctx.fillStyle = bandFill(verdict?.hit)
    ctx.fillRect(ax - 70, (top + bottom) / 2 - half, 140, half * 2)
    reticle(ctx, ax, ay, tone)
    pill(ctx, label, ax + 34, ay, tone)
  } else if (p.target_px) {
    // Landing spot on the floor, with the 90% band along the floor: where to put the cup.
    const [tx, ty] = toCanvas(v, p.target_px[0], p.target_px[1])
    tape(ctx, toCanvas(v, p.ghost[0][1], p.ghost[0][2])[0], tx, ty + 6, cmToCanvas)
    const left = tx - (p.value - p.ci90[0]) * cmToCanvas
    const right = tx + (p.ci90[1] - p.value) * cmToCanvas
    ctx.fillStyle = bandFill(verdict?.hit)
    ctx.fillRect(Math.min(left, tx - 3), ty - 26, Math.max(right - left, 6), 30)
    reticle(ctx, tx, ty - 10, tone, 14)
    pill(ctx, `cup here · ${label}`, tx + 24, ty + 22, tone)
  } else {
    // Timed calls (finish line, next peak): mark where the path ends, labelled with the time.
    const end = p.ghost[p.ghost.length - 1]
    const [ex, ey] = toCanvas(v, end[1], end[2])
    reticle(ctx, ex, ey, tone, 12)
    pill(ctx, label, ex + 20, ey + 26, tone)   // below, clear of the object's own label above it
  }

  // Ghost ball travelling along the prediction, so you can watch reality chase it.
  const t0 = p.ghost[0][0], t1 = p.ghost[p.ghost.length - 1][0]
  if (time >= t0 && time <= t1) {
    const k = ((time - t0) / (t1 - t0)) * (p.ghost.length - 1)
    const i = Math.min(Math.floor(k), p.ghost.length - 2)
    const f = k - i
    const gx = p.ghost[i][1] + (p.ghost[i + 1][1] - p.ghost[i][1]) * f
    const gy = p.ghost[i][2] + (p.ghost[i + 1][2] - p.ghost[i][2]) * f
    const [cx, cy] = toCanvas(v, gx, gy)
    const owner = s.track?.objects.find((o) => o.id === p.object) ?? s.track?.objects[0]
    const r = (owner?.r_px ?? 14) * v.scale
    ctx.strokeStyle = 'rgba(220, 235, 255, 0.9)'
    ctx.setLineDash([4, 4])
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(cx, cy, r + 3, 0, Math.PI * 2)
    ctx.stroke()
    ctx.setLineDash([])
  }
}

export type PathMode = 'sofar' | 'whole' | 'off'

export type SimOverlay = { color: string; label: string; points: [number, number, number][] }   // [t, x_px, y_px]

export function drawOverlay(ctx: CanvasRenderingContext2D, s: StageState, v: View, time: number,
                            paths: Path[] = [], pathMode: PathMode = 'sofar', sims: SimOverlay[] = []) {
  const { width, height } = ctx.canvas
  ctx.clearRect(0, 0, width, height)
  // Keep every overlay on the picture, never in the letterbox bars.
  ctx.save()
  ctx.beginPath()
  ctx.rect(v.ox, v.oy, v.vw * v.scale, v.vh * v.scale)
  ctx.clip()
  if (pathMode !== 'off') for (const p of paths) drawPath(ctx, v, p, pathMode === 'whole' ? Infinity : time)
  for (const sim of sims) drawSim(ctx, v, sim, time)
  drawScene(ctx, s, v, time)
  ctx.restore()
}

/** A simulated path (replay or what-if) with a ghost ball moving in time with the video. */
function drawSim(ctx: CanvasRenderingContext2D, v: View, sim: SimOverlay, time: number) {
  const pts = sim.points
  if (pts.length < 2) return
  ctx.save()
  ctx.strokeStyle = sim.color
  ctx.globalAlpha = 0.85
  ctx.lineWidth = 2
  ctx.setLineDash(sim.label.startsWith('replay') ? [4, 5] : [])
  ctx.beginPath()
  pts.forEach(([, x, y], i) => { const [cx, cy] = toCanvas(v, x, y); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy) })
  ctx.stroke()
  ctx.setLineDash([])
  if (time >= pts[0][0] && time <= pts[pts.length - 1][0]) {
    let i = 0
    while (i < pts.length - 2 && pts[i + 1][0] < time) i++
    const f = (time - pts[i][0]) / Math.max(pts[i + 1][0] - pts[i][0], 1e-9)
    const [cx, cy] = toCanvas(v, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f, pts[i][2] + (pts[i + 1][2] - pts[i][2]) * f)
    ctx.globalAlpha = 1
    ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.lineWidth = 2; ctx.stroke()
    ctx.font = '600 10px Inter, system-ui, sans-serif'; ctx.fillStyle = sim.color
    ctx.fillText(sim.label, cx + 12, cy - 10)
  }
  ctx.restore()
}

/** What happened: the object's whole path up to `until`, a dot every 0.1 s (a strobe photo: wider gaps = faster),
 *  and a marker at each bounce and peak. */
function drawPath(ctx: CanvasRenderingContext2D, v: View, p: Path, until: number) {
  const pts = p.points.filter((q) => q.t <= until)
  if (pts.length < 2) return
  const color = objectColor(p.id)
  ctx.save()
  ctx.strokeStyle = color + '8c'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  pts.forEach((q, i) => {
    const [x, y] = toCanvas(v, q.x, q.y)
    // A gap of more than a few frames is a tracking loss: don't draw a straight line across it.
    if (i === 0 || q.t - pts[i - 1].t > 0.1) ctx.moveTo(x, y); else ctx.lineTo(x, y)
  })
  ctx.stroke()
  let next = pts[0].t
  ctx.fillStyle = color
  for (const q of pts) {
    if (q.t + 1e-6 < next) continue
    next = q.t + 0.1
    const [x, y] = toCanvas(v, q.x, q.y)
    ctx.globalAlpha = 0.85
    ctx.beginPath(); ctx.arc(x, y, 2.6, 0, Math.PI * 2); ctx.fill()
  }
  ctx.globalAlpha = 1
  let n = 0
  for (const turn of p.turns) {
    if (turn.t > until) continue
    const [x, y] = toCanvas(v, turn.x, turn.y)
    ctx.strokeStyle = turn.kind === 'bounce' ? '#ffc857' : '#e8f6ff'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    if (turn.kind === 'bounce') { ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y); ctx.closePath() }
    else { ctx.moveTo(x - 6, y + 4); ctx.lineTo(x, y - 6); ctx.lineTo(x + 6, y + 4); ctx.closePath() }
    ctx.stroke()
    ctx.font = '600 10px Inter, system-ui, sans-serif'
    ctx.fillStyle = ctx.strokeStyle
    ctx.fillText(turn.kind === 'bounce' ? `bounce ${++n} · ${turn.t.toFixed(2)} s` : `peak · ${turn.t.toFixed(2)} s`,
      x + 10, turn.kind === 'bounce' ? y + 16 : y - 8)
  }
  ctx.restore()
}

function drawScene(ctx: CanvasRenderingContext2D, s: StageState, v: View, time: number) {
  const finishes = s.predictions.filter((p) => p.quantity === 'finish_time' && p.ghost && p.ghost.length > 1)
  if (finishes.length >= 2) {
    const ends = finishes.map((p) => { const g = p.ghost![p.ghost!.length - 1]; return toCanvas(v, g[1], g[2]) })
    finishLine(ctx, ends[0], ends[ends.length - 1])
  }
  for (const p of s.predictions) if (p.ghost) drawPrediction(ctx, v, p, s, time)
  const objects = s.track?.objects ?? []
  const single = objects.length <= 1

  // Comet trails: coloured by speed for a single object, by object when several are tracked.
  for (const obj of objects) {
    const pts = s.trail.map((e) => ({ t: e.t, o: e.objects.find((o) => o.id === obj.id) })).filter((p) => p.o)
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1].o!, b = pts[i].o!
      const alpha = Math.max(0, 1 - (time - pts[i].t) / 0.45)
      const speed = Math.hypot(b.vx ?? 0, b.vy ?? 0)
      const [x0, y0] = toCanvas(v, a.x_px, a.y_px)
      const [x1, y1] = toCanvas(v, b.x_px, b.y_px)
      ctx.strokeStyle = single ? speedColor(speed, alpha * 0.95) : objectColor(obj.id) + Math.round(alpha * 230).toString(16).padStart(2, '0')
      ctx.lineWidth = 2 + 7 * alpha
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(x0, y0)
      ctx.lineTo(x1, y1)
      ctx.stroke()
    }
  }

  for (const obj of objects) {
    const pose = s.poses[obj.id] ?? { x_px: obj.x_px, y_px: obj.y_px, dx: 0, dy: 0 }
    const [cx, cy] = toCanvas(v, pose.x_px, pose.y_px)
    const speed = Math.hypot(obj.vx ?? 0, obj.vy ?? 0)
    const color = objectColor(obj.id)

    // Glowing mask outline.
    ctx.save()
    ctx.shadowColor = color
    ctx.shadowBlur = 18
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.fillStyle = color + '1f'
    ctx.beginPath()
    if (obj.outline && obj.outline.length > 2) {
      obj.outline.forEach(([x, y], i) => {
        const [px, py] = toCanvas(v, x + pose.dx, y + pose.dy)
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
      })
      ctx.closePath()
    } else {
      ctx.arc(cx, cy, obj.r_px * v.scale, 0, Math.PI * 2)
    }
    ctx.fill()
    ctx.stroke()
    ctx.restore()

    // Velocity arrow: where the object will be in 60 ms at this speed.
    if (speed > 0.15) {
      const k = 0.06 * v.pxPerM * v.scale
      arrow(ctx, cx, cy, cx + (obj.vx ?? 0) * k, cy - (obj.vy ?? 0) * k, single ? speedColor(speed, 1) : color, 3)
    }
    // Gravity arrow once a fall law has been fitted (drop experiment).
    const g = s.fit?.params.g?.[0]
    if (g && single) {
      const len = Math.min(30 + g * 6, 110) * v.scale
      arrow(ctx, cx - 28 * v.scale, cy, cx - 28 * v.scale, cy + len, 'rgba(255, 200, 87, 0.9)', 2)
    }
    pill(ctx, `${obj.label} · ${speed.toFixed(2)} m/s`, cx + obj.r_px * v.scale + 14, cy - obj.r_px * v.scale - 14, color)

    // A called value with no path to draw (a mass) hangs as a tag under its object.
    for (const p of s.predictions) {
      if (p.object !== obj.id || p.ghost) continue
      const verdict = s.verdicts[p.id]
      const text = verdict
        ? `${verdict.hit ? '✓' : '✗'} called ${p.value.toFixed(0)} ${p.unit} · scale ${verdict.actual.toFixed(0)} ${p.unit}`
        : `called ${p.value.toFixed(0)} ${p.unit}`
      ctx.font = '500 13px Inter, system-ui, sans-serif'
      const w = ctx.measureText(text).width + 18
      // Left of the object: its own label sits up-right and timed calls hang below-right.
      pill(ctx, text, cx - obj.r_px * v.scale - 16 - w, cy + 4, verdict ? (verdict.hit ? HIT : MISS) : color)
    }
  }
}
