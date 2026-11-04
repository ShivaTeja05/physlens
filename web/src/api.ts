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

export async function getSamples(): Promise<Sample[]> {
  const r = await fetch('/api/samples')
  return r.json()
}

export async function getExperiments(): Promise<Experiment[]> {
  const r = await fetch('/api/experiments')
  return r.json()
}

export async function getHealth(): Promise<Health> {
  const r = await fetch('/api/health')
  return r.json()
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
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
  return (await r.json()).run_id
}

/** Streams a run's events; resolves when the engine says it is finished. */
export function streamRun(runId: string, onEvent: (e: PhysEvent) => void): Promise<'done' | 'error'> {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/api/runs/${runId}/stream`)
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data)
      if (data.type === '_end') resolve(data.status)
      else onEvent(data as PhysEvent)
    }
    ws.onerror = () => reject(new Error('lost connection to the engine'))
  })
}

// --- Scan mode -----------------------------------------------------------------------------------------------


export async function getScanSamples(): Promise<ScanSample[]> {
  const r = await fetch('/api/scan-samples')
  return r.json()
}

/** Starts a scan of a sample or an image (a webcam frame or an upload); returns the job id to poll. */
export async function startScan(input: { sample?: string; image?: Blob; reference?: object }): Promise<string> {
  const form = new FormData()
  if (input.sample) form.set('sample', input.sample)
  if (input.image) form.set('file', input.image, 'frame.jpg')
  if (input.reference) form.set('reference', JSON.stringify(input.reference))
  const r = await fetch('/api/scan', { method: 'POST', body: form })
  if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
  return (await r.json()).job_id
}

export async function scanStatus(jobId: string): Promise<ScanStatus> {
  const r = await fetch(`/api/scan/${jobId}`)
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
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

export async function getRecords(): Promise<RecordRow[]> {
  const r = await fetch('/api/records')
  return r.json()
}

export async function getRecordEvents(id: string): Promise<PhysEvent[]> {
  const r = await fetch(`/api/records/${id}/events`)
  if (!r.ok) throw new Error(r.statusText)
  return r.json()
}

export async function whatIf(id: string, object: string, changes: { gravity?: number; restitution?: number; speed_factor?: number; drag?: number }): Promise<WhatIf> {
  const form = new FormData()
  form.set('object', object)
  for (const [k, v] of Object.entries(changes)) if (v !== undefined) form.set(k, String(v))
  const r = await fetch(`/api/records/${id}/whatif`, { method: 'POST', body: form })
  const body = await r.json()
  if (!r.ok) throw new Error(body.detail ?? r.statusText)
  return body
}
