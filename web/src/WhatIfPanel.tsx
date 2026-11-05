import { useEffect, useState } from 'react'
import { whatIf, type WhatIf } from './api'

export type SimOverlay = { color: string; label: string; points: [number, number, number][] }   // [t, x_px, y_px]

const toOverlay = (w: WhatIf, which: 'replay' | 'what_if', color: string, label: string): SimOverlay | null => {
  const p = w[which]
  if (!p || !w.px_from_m) return null
  const [ax, bx, ay, by] = w.px_from_m
  return { color, label, points: p.t.map((t, i) => [t, ax * p.x_m[i] + bx, ay * p.y_m[i] + by]) }
}

/** Replay the recorded event from its stored state and the law found for it; then change one thing. */
export default function WhatIfPanel({ runId, object, onPaths }: { runId: string; object: string; onPaths: (p: SimOverlay[]) => void }) {
  const [res, setRes] = useState<WhatIf>()
  const [err, setErr] = useState<string>()
  const [busy, setBusy] = useState(false)
  const [g, setG] = useState<number>()
  const [e, setE] = useState<number>()
  const [speed, setSpeed] = useState(1)

  const run = async (changes: Parameters<typeof whatIf>[2]) => {
    setBusy(true); setErr(undefined)
    try {
      const w = await whatIf(runId, object, changes)
      setRes(w)
      if (g === undefined && w.measured.gravity) setG(Number(w.measured.gravity.toFixed(2)))
      if (e === undefined && w.measured.restitution) setE(Number(w.measured.restitution.toFixed(2)))
      onPaths([toOverlay(w, 'replay', '#e8f6ff', 'replay from the record'),
               toOverlay(w, 'what_if', '#ff5ad1', 'what if')].filter(Boolean) as SimOverlay[])
    } catch (x) {
      setErr(String(x)); onPaths([])
    } finally {
      setBusy(false)
    }
  }

  // A fresh run: replay it straight away, and clear the old paths.
  useEffect(() => { setRes(undefined); setG(undefined); setE(undefined); setSpeed(1); onPaths([]); run({}) }, [runId, object])  // eslint-disable-line react-hooks/exhaustive-deps

  const m = res?.measured
  const changes = () => ({
    gravity: m?.gravity && g !== undefined && Math.abs(g - m.gravity) > 0.005 ? g : undefined,
    restitution: m?.restitution && e !== undefined && Math.abs(e - m.restitution) > 0.005 ? e : undefined,
    speed_factor: speed !== 1 ? speed : undefined,
  })
  return (
    <section className="card whatif">
      <div className="label">Replay and what if <span className="knowledge rec">from the record</span></div>
      {err && <div className="sub err">{err.replace('Error: ', '')}</div>}
      {m && (
        <>
          <div className="sub">Law: <code>{m.law}</code>{res?.rms_m != null && <> · replay matches reality to ±{(res.rms_m * 100).toFixed(1)} cm</>}</div>
          {m.gravity != null && (
            <label className="slider">Gravity <b>{g?.toFixed(2)} m/s²</b>
              <input type="range" min={0.5} max={25} step={0.05} value={g ?? m.gravity} onChange={(x) => setG(Number(x.target.value))} />
            </label>
          )}
          {m.restitution != null && (
            <label className="slider">Bounciness <b>{e?.toFixed(2)}</b>
              <input type="range" min={0.1} max={0.98} step={0.01} value={e ?? m.restitution} onChange={(x) => setE(Number(x.target.value))} />
            </label>
          )}
          <label className="slider">Launch speed <b>×{speed.toFixed(2)}</b>
            <input type="range" min={0.3} max={1.8} step={0.05} value={speed} onChange={(x) => setSpeed(Number(x.target.value))} />
          </label>
          <div className="wi-row">
            <button className="chip active" disabled={busy} onClick={() => run(changes())}>Simulate what if</button>
            <button className="chip" disabled={busy} onClick={() => { setG(m.gravity ?? undefined); setE(m.restitution ?? undefined); setSpeed(1); run({}) }}>Reset</button>
          </div>
          <div className="wi-summary">{res?.summary.map((l, i) => <p key={i}>{l}</p>)}</div>
          <div className="wi-legend"><span className="sw replay" /> replay of the record <span className="sw alt" /> what if</div>
        </>
      )}
    </section>
  )
}
