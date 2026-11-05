import { useEffect, useState } from 'react'
import { Check, MagnifyingGlass, Warning } from '@phosphor-icons/react'
import { failure } from './api'

type Source = { source: string; url?: string }
type Agreed = { values_mm: number[]; sources: Source[]; agreed: boolean; value_m: number }
type Step = { tool: string; source?: string; url?: string; error?: string }
type Lookup = {
  object?: string; identified_by?: string; steps: Step[]
  agreed: Record<string, Agreed>
  checks: { name: string; measured_m: number; published_m: number; off_pct: number; published_agreed: boolean }[]
}

const TOOL_LABEL: Record<string, string> = {
  nhtsa_test_vehicles: 'NHTSA test record', decode_vin: 'VIN decoder', wikipedia_search: 'Wikipedia search', wikipedia_infobox: 'Wikipedia specs',
}

/** The Muse Spark specs agent for one world: what the object is and its published dimensions, each number traced to a
 * source, cross-checked between sources, and checked against what the video measured. Only shown when the engine runs. */
export default function SpecsPanel({ world }: { world: string }) {
  const [engine, setEngine] = useState(false)
  const [data, setData] = useState<Lookup>()
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string>()

  useEffect(() => {
    setData(undefined); setErr(undefined); setEngine(false)
    const ctl = new AbortController()
    // A cached lookup is free to read; a 404 from the engine means "none yet", anything else means no engine here.
    fetch(`/api/worlds/${encodeURIComponent(world)}/specs`, { signal: ctl.signal }).then(async (r) => {
      if (r.ok) { setEngine(true); setData(await r.json()) }
      else if (r.status === 404 && (r.headers.get('content-type') ?? '').includes('json')) setEngine(true)
    }).catch(() => {})
    return () => ctl.abort()
  }, [world])

  async function run() {
    setBusy(true); setErr(undefined)
    const fd = new FormData()
    fd.append('refresh', data ? 'true' : 'false')
    try {
      const r = await fetch(`/api/worlds/${encodeURIComponent(world)}/specs`, { method: 'POST', body: fd })
      if (!r.ok) throw await failure(r)
      setData(await r.json())
    } catch (e) {
      setErr(`Couldn’t look up the specs. ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setBusy(false)
    }
  }

  if (!engine) return null
  return (
    <div className="pw-specs">
      <div className="pw-specs-head">
        <div>
          <div className="pw-events-title">Published specs, looked up by a Muse Spark agent</div>
          <div className="pw-note">Every number is copied from a source the agent read, then checked against a second source and the video.</div>
        </div>
        <button type="button" disabled={busy} onClick={run}><MagnifyingGlass weight="bold" aria-hidden /> {busy ? 'Looking up…' : data ? 'Look up again' : 'Look up specs'}</button>
      </div>
      {err && <div className="pw-note pw-warn" role="alert">{err}</div>}
      {data && (
        <>
          <div className="pw-specs-id"><b>{data.object ?? 'Not identified'}</b>{data.identified_by && <span className="pw-note"> · {data.identified_by}</span>}</div>
          <div className="pw-specs-steps pw-note">
            {data.steps.map((s, i) => (
              <span key={i}>{i > 0 && ' → '}{s.url ? <a href={s.url} target="_blank" rel="noreferrer">{TOOL_LABEL[s.tool] ?? s.tool}</a> : TOOL_LABEL[s.tool] ?? s.tool}{s.error ? ' (failed)' : ''}</span>
            ))}
          </div>
          <table className="pw-specs-table">
            <thead><tr><th>Quantity</th><th>Published</th><th>Sources</th><th>Cross-check</th></tr></thead>
            <tbody>
              {Object.entries(data.agreed).map(([name, a]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td className="num">{a.value_m.toFixed(3)} m</td>
                  <td>{a.sources.map((s, i) => <span key={i}>{i > 0 && ', '}{s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.source}</a> : s.source}</span>)}
                    <span className="pw-note"> ({a.values_mm.map((v) => `${v} mm`).join(' / ')})</span></td>
                  <td>{a.agreed ? <span className="pw-ok"><Check weight="bold" aria-hidden /> sources agree</span>
                    : a.sources.length < 2 ? <span className="pw-note">one source only</span>
                      : <span className="pw-warn"><Warning weight="fill" aria-hidden /> sources differ</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.checks.map((c) => (
            <div key={c.name} className={`pw-specs-check ${Math.abs(c.off_pct) <= 1 ? 'pw-ok' : 'pw-warn'}`}>
              Video measured the {c.name} at <b className="num">{c.measured_m.toFixed(3)} m</b>, published {c.published_m.toFixed(3)} m:
              {' '}<b>{c.off_pct > 0 ? '+' : ''}{c.off_pct}%</b>{Math.abs(c.off_pct) <= 1 ? ', consistent.' : ', the measurement or its scale needs a look.'}
            </div>
          ))}
        </>
      )}
    </div>
  )
}
