import type { PhysEvent } from './types'
import type { ScanSample, ScanStatus } from './scanTypes'

export type Sample = {
  name: string; url: string; label?: string; credit?: string; px_per_m?: number; experiment?: string
  params?: Record<string, number>
}
export type Experiment = {
  id: string; title: string; hook: string; kind: string; params: Record<string, number>; objects: string[]
}
export type Health = { tracker: string; agent: string }

/** The engine's own words for a failed request. FastAPI sends {"detail": "..."} (or a list of field errors);
 *  a crash or the dev proxy sends plain text, and an engine that isn't running gives 502 or 504 through Vite. */
export async function failure(r: Response): Promise<Error> {
  if (r.status === 502 || (r.status === 503 && !r.headers.get('content-type')?.includes('json')) || r.status === 504)
    return new Error('The engine is not reachable. Start the backend on port 8000, then try again.')
  const text = await r.text().catch(() => '')
  try {
    const d = JSON.parse(text).detail
    if (typeof d === 'string') return new Error(d)
    if (Array.isArray(d)) return new Error(d.map((x) => `${(x.loc ?? []).slice(-1)[0] ?? 'input'}: ${x.msg}`).join('; '))
  } catch { /* not JSON */ }
  return new Error(r.status >= 500 ? `The engine hit an error (${r.status}). Its terminal shows the details.` : text || r.statusText)
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw await failure(r)
  return r.json()
}

export function getSamples(): Promise<Sample[]> {
  return getJson('/api/samples')
}

export function getExperiments(): Promise<Experiment[]> {
  return getJson('/api/experiments')
}

export function getHealth(): Promise<Health> {
  return getJson('/api/health')
}

export type RunInput = {
  sample?: string; file?: File; experiment?: string; pxPerM?: number; diameterCm?: number; params?: Record<string, number>
}

export async function createRun(input: RunInput): Promise<string> {
  const form = new FormData()
  form.set('experiment_id', input.experiment ?? 'drop')
  if (input.sample) form.set('sample', input.sample)
  if (input.file) form.set('file', input.file)
  if (input.pxPerM) form.set('px_per_m', String(input.pxPerM))
  if (input.diameterCm) form.set('diameter_cm', String(input.diameterCm))
  if (input.params && Object.keys(input.params).length) form.set('params', JSON.stringify(input.params))
  const r = await fetch('/api/runs', { method: 'POST', body: form })
  if (!r.ok) throw await failure(r)
  return (await r.json()).run_id
}

export type RunEnd = { status: 'done' | 'error'; error?: string | null }

/** Streams a run's events; resolves when the engine says it is finished, and fails if the connection drops first
 *  (the engine stopped or restarted mid-run), so the page never waits forever. */
export function streamRun(runId: string, onEvent: (e: PhysEvent) => void): Promise<RunEnd> {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/stream`)
    let ended = false
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data)
      if (data.type === '_end') { ended = true; resolve({ status: data.status, error: data.error }) }
      else onEvent(data as PhysEvent)
    }
    ws.onclose = () => { if (!ended) reject(new Error('Lost the connection to the engine before the analysis finished. Check that the backend is still running, then try again.')) }
  })
}

// --- Scan mode -----------------------------------------------------------------------------------------------


export function getScanSamples(): Promise<ScanSample[]> {
  return getJson('/api/scan-samples')
}

/** Starts a scan of a sample or an image (a webcam frame or an upload); returns the job id to poll. */
export async function startScan(input: { sample?: string; image?: Blob; reference?: object }): Promise<string> {
  const form = new FormData()
  if (input.sample) form.set('sample', input.sample)
  if (input.image) form.set('file', input.image, 'frame.jpg')
  if (input.reference) form.set('reference', JSON.stringify(input.reference))
  const r = await fetch('/api/scan', { method: 'POST', body: form })
  if (!r.ok) throw await failure(r)
  return (await r.json()).job_id
}

export function scanStatus(jobId: string): Promise<ScanStatus> {
  return getJson(`/api/scan/${jobId}`)
}

// --- Recorder: past runs, replay and what-if -------------------------------------------------------------------

export type RecordRow = {
  id: string; recorded_at: string; experiment: string | null; source_url: string; fps: number; duration: number
  status: string; objects: number; contacts: number; hits: number; verdicts: number
}
export type SimPath = { t: number[]; x_m: number[]; y_m: number[]; peaks: { t: number; height_m: number }[]; params: Record<string, number | null> }
export type WhatIf = {
  object: string; measured: { law: string; restitution: number | null; gravity: number | null; start_t: number; floor_m: number | null }
  replay: SimPath; what_if: SimPath | null; rms_m: number | null; summary: string[]; px_from_m: [number, number, number, number] | null
}

export function getRecords(): Promise<RecordRow[]> {
  return getJson('/api/records')
}

export function getRecordEvents(id: string): Promise<PhysEvent[]> {
  return getJson(`/api/records/${encodeURIComponent(id)}/events`)
}

export async function whatIf(id: string, object: string, changes: { gravity?: number; restitution?: number; speed_factor?: number; drag?: number }): Promise<WhatIf> {
  const form = new FormData()
  form.set('object', object)
  for (const [k, v] of Object.entries(changes)) if (v !== undefined) form.set(k, String(v))
  const r = await fetch(`/api/records/${encodeURIComponent(id)}/whatif`, { method: 'POST', body: form })
  if (!r.ok) throw await failure(r)
  return r.json()
}
