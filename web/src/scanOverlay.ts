import type { ScanPart, ScanResult, ScanStage } from './scanTypes'

export type ScanView = { scale: number; ox: number; oy: number; w: number; h: number }
export type ScanAnim = {
  stage: ScanStage
  now: number                       // ms
  stageAt: number                   // when the current stage began
  partsAt: number | null            // when outlines arrived
  edges: Float32Array | null        // edge points of the picture, x,y pairs in image pixels
  highlight: Set<string> | null     // parts the narrator is talking about
}

const CYAN = '#3ee6ff'
const KIND_COLORS: Record<string, string> = {
  load: '#ffc857', force: '#e8f6ff', energy: '#ff6b7d', light: '#fff27a', motion: CYAN, signal: '#b18cff', flow: '#5aa9ff',
}
const SWEEP_MS = 2400

const px = (v: ScanView, x: number, y: number): [number, number] => [v.ox + x * v.scale, v.oy + y * v.scale]

/** Edge points of an image (Sobel, strongest ~9%), computed once per picture on a small copy. */
export function edgePoints(img: HTMLImageElement): Float32Array {
  const W = 320, H = Math.max(1, Math.round((img.naturalHeight / img.naturalWidth) * W))
  const c = document.createElement('canvas')
  c.width = W; c.height = H
  const g = c.getContext('2d', { willReadFrequently: true })!
  g.drawImage(img, 0, 0, W, H)
  const d = g.getImageData(0, 0, W, H).data
  const lum = new Float32Array(W * H)
  for (let i = 0; i < W * H; i++) lum[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]
  const mag = new Float32Array(W * H)
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x
    const gx = lum[i - W + 1] + 2 * lum[i + 1] + lum[i + W + 1] - lum[i - W - 1] - 2 * lum[i - 1] - lum[i + W - 1]
    const gy = lum[i + W - 1] + 2 * lum[i + W] + lum[i + W + 1] - lum[i - W - 1] - 2 * lum[i - W] - lum[i - W + 1]
    mag[i] = Math.hypot(gx, gy)
  }
  const sorted = Array.from(mag).sort((a, b) => b - a)
  const thr = sorted[Math.floor(sorted.length * 0.09)] || 1
  const pts: number[] = []
  const k = img.naturalWidth / W
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) if (mag[y * W + x] >= thr) pts.push(x * k, y * k)
  return Float32Array.from(pts)
}

function pill(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, alignRight = false, size = 12) {
  ctx.font = `600 ${size}px Inter, system-ui, sans-serif`
  const w = ctx.measureText(text).width + 16
  const left = alignRight ? x - w : x
  ctx.fillStyle = 'rgba(4, 10, 16, 0.78)'
  ctx.strokeStyle = color
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.roundRect(left, y - 11, w, 22, 4)
  ctx.fill()
  ctx.stroke()
  ctx.fillStyle = '#e8f6ff'
  ctx.fillText(text, left + 8, y + 4)
  return { left, right: left + w }
}

function anchorOf(p: ScanPart): [number, number] | null {
  if (p.union_box) return [(p.union_box[0] + p.union_box[2]) / 2, (p.union_box[1] + p.union_box[3]) / 2]
  return p.instances[0]?.center ?? null
}

function pathLength(pts: [number, number][]) {
  let L = 0
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length]
    L += Math.hypot(b[0] - a[0], b[1] - a[1])
  }
  return L
}

function headline(p: ScanPart): string {
  const count = p.measures.find((m) => m.label === 'count')
  const rest = p.measures.filter((m) => m.label !== 'count' && m.label !== 'columns')
  let s = p.name.toUpperCase()
  if (count) s += ` ×${count.value}`
  const key = rest[0]
  if (key) s += ` · ${key.label === 'rows' ? `${key.value} rows` : `${key.value}${key.unit}`}`
  return s
}

export function drawScan(ctx: CanvasRenderingContext2D, r: ScanResult, v: ScanView, a: ScanAnim) {
  const { width, height } = ctx.canvas
  ctx.clearRect(0, 0, width, height)
  const W = r.image.width, H = r.image.height
  const scanning = a.stage !== 'done' && a.stage !== 'error'
  const t = a.now - a.stageAt

  ctx.save()
  ctx.beginPath()
  ctx.rect(v.ox, v.oy, W * v.scale, H * v.scale)
  ctx.clip()

  // Dim the photo slightly so the overlay reads like a heads-up display.
  ctx.fillStyle = scanning ? 'rgba(2, 8, 14, 0.38)' : 'rgba(2, 8, 14, 0.22)'
  ctx.fillRect(v.ox, v.oy, W * v.scale, H * v.scale)

  // Grid.
  ctx.strokeStyle = scanning ? 'rgba(62, 230, 255, 0.07)' : 'rgba(62, 230, 255, 0.035)'
  ctx.lineWidth = 1
  const step = 40
  for (let x = v.ox; x < v.ox + W * v.scale; x += step) { ctx.beginPath(); ctx.moveTo(x, v.oy); ctx.lineTo(x, v.oy + H * v.scale); ctx.stroke() }
  for (let y = v.oy; y < v.oy + H * v.scale; y += step) { ctx.beginPath(); ctx.moveTo(v.ox, y); ctx.lineTo(v.ox + W * v.scale, y); ctx.stroke() }

  // Sweep line, revealing the edge wireframe as it passes.
  const sweepY = scanning ? ((a.now % SWEEP_MS) / SWEEP_MS) * H : H
  if (a.edges) {
    ctx.fillStyle = CYAN
    for (let i = 0; i < a.edges.length; i += 2) {
      const ex = a.edges[i], ey = a.edges[i + 1]
      if (ey > sweepY) continue
      const near = scanning ? Math.max(0, 1 - (sweepY - ey) / (H * 0.25)) : 0
      ctx.globalAlpha = scanning ? 0.18 + 0.6 * near : 0.13
      const [cx, cy] = px(v, ex, ey)
      ctx.fillRect(cx, cy, 1.4, 1.4)
    }
    ctx.globalAlpha = 1
  }
  if (scanning) {
    const [, sy] = px(v, 0, sweepY)
    const grad = ctx.createLinearGradient(0, sy - 60, 0, sy)
    grad.addColorStop(0, 'rgba(62, 230, 255, 0)')
    grad.addColorStop(1, 'rgba(62, 230, 255, 0.22)')
    ctx.fillStyle = grad
    ctx.fillRect(v.ox, sy - 60, W * v.scale, 60)
    ctx.strokeStyle = 'rgba(160, 245, 255, 0.95)'
    ctx.lineWidth = 2
    ctx.shadowColor = CYAN
    ctx.shadowBlur = 16
    ctx.beginPath(); ctx.moveTo(v.ox, sy); ctx.lineTo(v.ox + W * v.scale, sy); ctx.stroke()
    ctx.shadowBlur = 0
  }

  const parts = (r.parts ?? []).filter((p) => p.instances.length)
  const since = a.partsAt === null ? -1 : a.now - a.partsAt
  const lit = (p: ScanPart) => !a.highlight || a.highlight.has(p.id)

  // Outlines, drawn in one after another.
  parts.forEach((p, k) => {
    const prog = Math.min(Math.max((since - k * 280) / 700, 0), 1)
    if (prog <= 0) return
    const alpha = lit(p) ? 1 : 0.28
    ctx.globalAlpha = alpha
    for (const inst of p.instances) {
      if (inst.outline.length < 3) continue
      const pts = inst.outline.map(([x, y]) => px(v, x, y))
      const L = pathLength(pts as [number, number][])
      ctx.save()
      ctx.strokeStyle = p.color
      ctx.fillStyle = p.color + (a.highlight?.has(p.id) ? '33' : '1a')
      ctx.lineWidth = a.highlight?.has(p.id) ? 2.4 : 1.6
      ctx.shadowColor = p.color
      ctx.shadowBlur = a.highlight?.has(p.id) ? 18 + 6 * Math.sin(a.now / 180) : 10
      ctx.setLineDash([L * prog, L])
      ctx.beginPath()
      pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)))
      ctx.closePath()
      if (prog >= 1) ctx.fill()
      ctx.stroke()
      ctx.restore()
    }
    // Floors / rows of repeated parts.
    if (prog >= 1 && p.row_lines && p.union_box && p.row_lines.length >= 2) {
      const [x0, , x1] = p.union_box
      const rows = [...p.row_lines].sort((m, n) => n - m)   // bottom first
      ctx.setLineDash([5, 6])
      ctx.strokeStyle = p.color + 'aa'
      ctx.lineWidth = 1
      rows.forEach((ry, i) => {
        const [lx, ly] = px(v, x0 - 14, ry), [rx] = px(v, x1 + 14, ry)
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ly); ctx.stroke()
        ctx.font = '600 9px Inter, system-ui, sans-serif'
        ctx.fillStyle = p.color
        ctx.fillText(`R${i + 1}`, rx + 3, ly + 3)
      })
      ctx.setLineDash([])
    }
    // A round body's full disc (fitted to its sunlit edge), dashed: the circle the lit share is measured against.
    if (prog >= 1 && p.disc) {
      const [dx, dy] = px(v, p.disc[0], p.disc[1])
      const lit = p.measures.find((m) => m.label === 'disc lit')
      ctx.strokeStyle = p.color
      ctx.lineWidth = 1.5
      ctx.setLineDash([7, 7])
      ctx.lineDashOffset = -a.now / 60
      ctx.beginPath(); ctx.arc(dx, dy, p.disc[2] * v.scale, 0, Math.PI * 2); ctx.stroke()
      ctx.setLineDash([]); ctx.lineDashOffset = 0
      ctx.beginPath(); ctx.moveTo(dx - 8, dy); ctx.lineTo(dx + 8, dy); ctx.moveTo(dx, dy - 8); ctx.lineTo(dx, dy + 8); ctx.stroke()
      if (lit) pill(ctx, `${lit.value.toFixed(1)}% of the disc lit`, dx + 12, dy + p.disc[2] * v.scale * 0.72, p.color)
    }
    // Lean: the part's long axis against true vertical, with the angle.
    const tilt = p.measures.find((m) => m.label === 'tilt from upright')
    const anchor = anchorOf(p)
    if (prog >= 1 && tilt && p.axis && anchor && p.union_box) {
      const [cx, cy] = px(v, anchor[0], anchor[1])
      const len = ((p.union_box[3] - p.union_box[1]) / 2) * v.scale * 0.9
      let [ax, ay] = p.axis
      if (ay > 0) { ax = -ax; ay = -ay }   // point upwards
      const upright = Math.abs(ay) >= Math.abs(ax)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      if (upright) { ctx.moveTo(cx, cy + len); ctx.lineTo(cx, cy - len) } else { ctx.moveTo(cx - len, cy); ctx.lineTo(cx + len, cy) }
      ctx.stroke()
      ctx.setLineDash([])
      ctx.strokeStyle = p.color
      ctx.lineWidth = 2
      ctx.beginPath(); ctx.moveTo(cx - ax * len, cy - ay * len); ctx.lineTo(cx + ax * len, cy + ay * len); ctx.stroke()
      const base = upright ? -Math.PI / 2 : 0
      const ang = Math.atan2(ay, ax)
      ctx.beginPath(); ctx.arc(cx, cy, Math.min(46, len * 0.6), Math.min(base, ang), Math.max(base, ang)); ctx.stroke()
      pill(ctx, `${tilt.value.toFixed(1)}°`, cx + 10, cy - Math.min(46, len * 0.6) - 14, p.color)
    }
    ctx.globalAlpha = 1
  })

  // Relations: flowing arrows between parts (load, light, energy...), with a contact check.
  const byId = Object.fromEntries(parts.map((p) => [p.id, p]))
  if (since > parts.length * 280 + 500) {
    for (const rel of r.relations ?? []) {
      const pa = byId[rel.from], pb = byId[rel.to]
      const A = pa && anchorOf(pa), B = pb && anchorOf(pb)
      if (!A || !B) continue
      const [x0, y0] = px(v, A[0], A[1]), [x1, y1] = px(v, B[0], B[1])
      const color = KIND_COLORS[rel.kind] ?? CYAN
      ctx.globalAlpha = lit(pa) || lit(pb) ? 1 : 0.25
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([10, 8])
      ctx.lineDashOffset = -a.now / 30
      // Motion in a still photo can only be inferred: draw it as the arc a thrown thing follows under gravity
      // (bowed upwards), not a straight line, and say it's inferred.
      const arc = rel.kind === 'motion'
      const cxq = (x0 + x1) / 2, cyq = Math.min(y0, y1) - 0.35 * Math.hypot(x1 - x0, y1 - y0)
      ctx.beginPath(); ctx.moveTo(x0, y0)
      if (arc) ctx.quadraticCurveTo(cxq, cyq, x1, y1); else ctx.lineTo(x1, y1)
      ctx.stroke()
      ctx.setLineDash([]); ctx.lineDashOffset = 0
      if (arc) {   // a ghost of the object travelling along its likely path
        const u = (a.now / 1600) % 1
        const gx = (1 - u) ** 2 * x0 + 2 * (1 - u) * u * cxq + u ** 2 * x1
        const gy = (1 - u) ** 2 * y0 + 2 * (1 - u) * u * cyq + u ** 2 * y1
        ctx.beginPath(); ctx.arc(gx, gy, 5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.globalAlpha *= 0.8; ctx.fill(); ctx.globalAlpha = 1
      }
      const ang = arc ? Math.atan2(y1 - cyq, x1 - cxq) : Math.atan2(y1 - y0, x1 - x0)
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x1 - 12 * Math.cos(ang - 0.4), y1 - 12 * Math.sin(ang - 0.4))
      ctx.lineTo(x1 - 12 * Math.cos(ang + 0.4), y1 - 12 * Math.sin(ang + 0.4))
      ctx.fill()
      const label = arc ? `${rel.label} · likely path, inferred (one photo can't show motion)`
        : rel.contact === undefined ? rel.label : `${rel.label}${rel.contact ? ' · ✓ contact' : ' · no contact seen'}`
      pill(ctx, label, arc ? cxq - 60 : (x0 + x1) / 2 + 6, arc ? (cyq + Math.min(y0, y1)) / 2 : (y0 + y1) / 2, color, false, 11)
      ctx.globalAlpha = 1
    }
  }
  ctx.restore()

  // Call-outs in the margins, stacked so they never overlap.
  const sides: Record<'L' | 'R', { p: ScanPart; ay: number; ax: number }[]> = { L: [], R: [] }
  parts.forEach((p, k) => {
    if (since - k * 280 < 700) return
    const anc = anchorOf(p)
    if (!anc) return
    const box = p.union_box ?? p.instances[0].box
    const [ax, ay] = px(v, anc[0] < W / 2 ? box[0] : box[2], (box[1] + box[3]) / 2)
    sides[anc[0] < W / 2 ? 'L' : 'R'].push({ p, ay, ax })
  })
  for (const side of ['L', 'R'] as const) {
    const list = sides[side].sort((m, n) => m.ay - n.ay)
    let lastY = -Infinity
    const x = side === 'L' ? Math.max(v.ox - 8, 12) : Math.min(v.ox + W * v.scale + 8, width - 12)
    for (const { p, ay, ax } of list) {
      const y = Math.max(ay, lastY + 30)
      lastY = y
      ctx.globalAlpha = lit(p) ? 1 : 0.3
      ctx.strokeStyle = p.color
      ctx.lineWidth = 1
      const lx = side === 'L' ? x + 6 : x - 6
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(lx + (side === 'L' ? 40 : -40), y); ctx.lineTo(lx, y); ctx.stroke()
      ctx.beginPath(); ctx.arc(ax, ay, 3, 0, Math.PI * 2); ctx.fillStyle = p.color; ctx.fill()
      pill(ctx, headline(p), side === 'L' ? x + 10 : x - 10, y, p.color, side === 'R')
      ctx.globalAlpha = 1
    }
  }

  // Corner brackets: the scanner's frame.
  const x0 = v.ox, y0 = v.oy, x1 = v.ox + W * v.scale, y1 = v.oy + H * v.scale, c = 26
  ctx.strokeStyle = scanning ? `rgba(62, 230, 255, ${0.6 + 0.4 * Math.sin(t / 220)})` : 'rgba(62, 230, 255, 0.8)'
  ctx.lineWidth = 2
  for (const [bx, by, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]] as const) {
    ctx.beginPath(); ctx.moveTo(bx + dx * c, by); ctx.lineTo(bx, by); ctx.lineTo(bx, by + dy * c); ctx.stroke()
  }
}
