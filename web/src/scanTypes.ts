// Mirrors backend/physlens/scan.py's result.

export type ScanInstance = { box: [number, number, number, number]; center: [number, number]; area: number; outline: [number, number][] }
export type ScanMeasure = { label: string; value: number; unit: string; note?: string }
export type ScanPart = {
  id: string; name: string; role: string; many: boolean; color: string
  instances: ScanInstance[]; measures: ScanMeasure[]
  union_box?: [number, number, number, number]; row_lines?: number[]; axis?: [number, number]
  /** A round body's fitted disc (cx, cy, r) in image pixels: the full circle its sunlit edge belongs to. */
  disc?: [number, number, number]
}
export type ScanRelation = { from: string; to: string; kind: string; label: string; contact?: boolean }
export type ScanStep = { text: string; parts: string[] }
export type ScanResult = {
  id?: string
  image: { width: number; height: number; url: string }
  scene_type?: string; title?: string; summary?: string; headline?: string
  parts?: ScanPart[]; relations?: ScanRelation[]; how_it_works?: ScanStep[]
  seen?: { text: string; verified: boolean }[]; explain_source?: string
  measure_ideas?: string[]; video_experiment?: string; video_hint?: string
  evidence?: Record<string, number>
  /** The model's words tested against the measurements (e.g. a Moon's phase name vs its lit share). */
  checks?: { ok: boolean; claim: string; text: string }[]
}
export type ScanStage = 'queued' | 'looking' | 'outlining' | 'explaining' | 'done' | 'error'
export type ScanStatus = { stage: ScanStage; result: ScanResult; error?: string }
export type ScanSample = { name: string; url: string; label?: string; credit?: string }
