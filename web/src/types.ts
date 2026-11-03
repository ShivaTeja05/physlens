// Mirrors ../../contract/events.schema.json. Keep in sync with backend/physlens/events.py.

type Base = { run_id: string; t: number }

export type VideoInfo = { url: string; fps: number; width: number; height: number; duration: number }

export type RunStateEvent = Base & {
  type: 'run_state'
  state: 'created' | 'tracking' | 'measuring' | 'hypothesizing' | 'locked' | 'verifying' | 'done' | 'error'
  message?: string
  video?: VideoInfo
  /** Set when the engine detected the experiment itself (auto-detect). */
  experiment?: string
}
export type CalibrationEvent = Base & { type: 'calibration'; method: string; px_per_m: number; rel_uncertainty: number }
export type TrackedObject = {
  id: string; label: string; x_px: number; y_px: number; r_px: number
  x_m?: number; y_m?: number; vx?: number; vy?: number; outline?: [number, number][]
}
export type TrackEvent = Base & { type: 'track'; objects: TrackedObject[] }
export type Candidate = { name: string; equation: string; score: number; aic?: number }
export type HypothesisEvent = Base & { type: 'hypothesis'; topic?: string; label?: string; candidates: Candidate[] }
export type FitEvent = Base & {
  type: 'fit'; id: string; model: string; params: Record<string, [number, number]>; rmse_m?: number; window?: [number, number]
}
export type MemoryVsEyesEvent = Base & {
  type: 'memory_vs_eyes'; quantity: string; unit?: string; memory_value: number; eyes_value: number
  eyes_err?: number; memory_text?: string; eyes_text?: string
}
export type PredictionEvent = Base & {
  type: 'prediction'; id: string; text: string; quantity?: string; value: number; unit?: string
  ci90: [number, number]; target_t: number; based_on_until_t: number; ghost?: [number, number, number][]
  hash: string; locked_at: string
  object?: string; target_px?: [number, number]; winner?: string
}
export type VerdictEvent = Base & { type: 'verdict'; id: string; actual: number; hit: boolean; error?: number }
export type ExplainEvent = Base & { type: 'explain'; text: string; evidence: string[]; verified: boolean; source?: string }
export type NarrationEvent = Base & { type: 'narration'; text: string }
export type MeasurementItem = { label: string; value: number; err: number | null; unit: string }
export type MeasurementEvent = Base & { type: 'measurement'; items: MeasurementItem[] }

/** An equation of motion found by searching combinations of force terms (not picked from a fixed list). */
export type LawEvent = Base & {
  type: 'law'; id: string; label: string; axis: string; equation: string; object?: string
  terms: { term: string; value: number; err: number }[]; meaning: string[]; searched: number; samples: number
  rmse: number; alternatives: Candidate[]
}

/** A key moment an analysis reports itself (impact, max crush, rebound...). */
export type MomentEvent = Base & { type: 'moment'; object: string; kind: string; speed_in?: number; speed_out?: number; note?: string }
/** A vehicle's front and rear edges over time, in corrected metres from the barrier face, for the animated
 *  reconstruction: frames are [t, front_m | null, rear_m | null]. */
export type ReconstructionEvent = Base & {
  type: 'reconstruction'; object: string; label: string; length_m: number; height_m: number; mass_kg?: number
  frames: [number, number | null, number | null][]; impact_t: number; speed_ms: number; crush_m: number
  max_crush_t: number | null; rebound_ms: number
}

export type PhysEvent =
  | RunStateEvent | CalibrationEvent | TrackEvent | HypothesisEvent | FitEvent | MemoryVsEyesEvent
  | PredictionEvent | VerdictEvent | ExplainEvent | NarrationEvent | MeasurementEvent | LawEvent | MomentEvent
  | ReconstructionEvent
