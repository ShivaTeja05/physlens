import type {
  CalibrationEvent, ExplainEvent, FitEvent, HypothesisEvent, LawEvent, MeasurementEvent, MemoryVsEyesEvent, NarrationEvent, PhysEvent,
  PredictionEvent, TrackEvent, VerdictEvent,
} from './types'

/** Everything the stage shows at one moment of video time. */
export type StageState = {
  calibration?: CalibrationEvent
  track?: TrackEvent
  /** Where the first object is at the exact video time, interpolated between track events. */
  pose?: Pose
  /** The same for every tracked object, by id. */
  poses: Record<string, Pose>
  trail: TrackEvent[]
  hypothesis?: HypothesisEvent
  bounceHypothesis?: HypothesisEvent
  /** Every hypothesis race by topic (motion, bounce, mass:red, …), latest of each. */
  races: Record<string, HypothesisEvent>
  measurement?: MeasurementEvent
  /** Discovered equations of motion, by id. */
  laws: Record<string, LawEvent>
  memories: Record<string, MemoryVsEyesEvent>
  fit?: FitEvent
  memory?: MemoryVsEyesEvent
  predictions: PredictionEvent[]
  verdicts: Record<string, VerdictEvent>
  explain?: ExplainEvent
  narration?: NarrationEvent
}

export type Pose = { x_px: number; y_px: number; dx: number; dy: number }

const TRAIL_SECONDS = 0.45

/** Index of the last event with t ≤ time (events are sorted by t). */
export function cursor(events: PhysEvent[], time: number): number {
  let lo = 0, hi = events.length - 1, ans = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (events[mid].t <= time) { ans = mid; lo = mid + 1 } else hi = mid - 1
  }
  return ans
}

function interpolatePoses(events: PhysEvent[], upTo: number, prev: TrackEvent, time: number): Record<string, Pose> {
  let next: TrackEvent | undefined
  for (let i = upTo + 1; i < events.length; i++) {
    const e = events[i]
    if (e.type === 'track') { next = e; break }
  }
  const f = next && next.t > prev.t ? Math.min(Math.max((time - prev.t) / (next.t - prev.t), 0), 1) : 0
  const out: Record<string, Pose> = {}
  for (const a of prev.objects) {
    const b = next?.objects.find((o) => o.id === a.id)
    const x = b ? a.x_px + (b.x_px - a.x_px) * f : a.x_px
    const y = b ? a.y_px + (b.y_px - a.y_px) * f : a.y_px
    out[a.id] = { x_px: x, y_px: y, dx: x - a.x_px, dy: y - a.y_px }
  }
  return out
}

export function deriveState(events: PhysEvent[], upTo: number, time?: number): StageState {
  const s: StageState = { trail: [], predictions: [], verdicts: {}, races: {}, memories: {}, poses: {}, laws: {} }
  for (let i = 0; i <= upTo; i++) {
    const e = events[i]
    switch (e.type) {
      case 'calibration': s.calibration = e; break
      case 'track': s.track = e; s.trail.push(e); break
      case 'hypothesis':
        if (e.topic === 'bounce') s.bounceHypothesis = e
        else if (!e.topic || e.topic === 'motion') s.hypothesis = e
        s.races[e.topic ?? 'motion'] = e
        break
      case 'measurement': s.measurement = e; break
      case 'law': s.laws[e.id] = e; break
      case 'fit': if (e.id !== 'fit_all') s.fit = e; break
      case 'memory_vs_eyes': s.memory = e; s.memories[e.quantity] = e; break
      case 'prediction': s.predictions.push(e); break
      case 'verdict': s.verdicts[e.id] = e; break
      case 'explain': s.explain = e; break
      case 'narration': s.narration = e; break
    }
  }
  const now = s.track?.t ?? 0
  s.trail = s.trail.filter((e) => now - e.t <= TRAIL_SECONDS)
  if (s.track && time !== undefined) {
    s.poses = interpolatePoses(events, upTo, s.track, time)
    const first = s.track.objects[0]
    if (first) s.pose = s.poses[first.id]
  }
  return s
}

export type PathPoint = { t: number; x: number; y: number }
export type Turn = PathPoint & { kind: 'bounce' | 'peak' }
export type Path = { id: string; label: string; points: PathPoint[]; turns: Turn[] }

/** Every tracked object's whole path through the clip, with its turning points: a bounce (lowest point, falling →
 *  rising) or a peak (highest point, rising → falling). Image y grows downwards. */
export function objectPaths(events: PhysEvent[]): Path[] {
  const byId = new Map<string, Path>()
  for (const e of events) {
    if (e.type !== 'track') continue
    for (const o of e.objects) {
      let p = byId.get(o.id)
      if (!p) { p = { id: o.id, label: o.label, points: [], turns: [] }; byId.set(o.id, p) }
      p.points.push({ t: e.t, x: o.x_px, y: o.y_px })
    }
  }
  for (const p of byId.values()) {
    const pts = p.points
    const k = 3          // compare a few samples either side so tracker jitter isn't a turn
    for (let i = k; i < pts.length - k; i++) {
      const before = pts[i].y - pts[i - k].y, after = pts[i + k].y - pts[i].y
      const isMax = pts.slice(i - k, i + k + 1).every((q) => q.y <= pts[i].y)
      const isMin = pts.slice(i - k, i + k + 1).every((q) => q.y >= pts[i].y)
      if (isMax && before > 2 && after < -2) p.turns.push({ ...pts[i], kind: 'bounce' })
      else if (isMin && before < -2 && after > 2) p.turns.push({ ...pts[i], kind: 'peak' })
    }
  }
  return [...byId.values()]
}

/** Timeline events in video order; run_state messages are status, not timeline. */
export function timeline(events: PhysEvent[]): PhysEvent[] {
  return events.filter((e) => e.type !== 'run_state').sort((a, b) => a.t - b.t)
}
