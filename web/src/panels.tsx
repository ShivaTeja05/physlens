import { Check, LockSimple, X } from '@phosphor-icons/react'
import type { StageState } from './derive'
import type { PredictionEvent, VerdictEvent } from './types'

const MODEL_NAMES: Record<string, string> = {
  constant_velocity: 'Constant speed',
  free_fall: 'Free fall',
  linear_drag: 'Free fall + air drag',
  constant_restitution: 'Same bounciness every bounce',
  speed_dependent_restitution: 'Bounciness depends on impact speed',
}

export function Readout({ s }: { s: StageState }) {
  const g = s.fit?.params.g
  return (
    <section className="card readout">
      <div className="label">Gravity in this video</div>
      {g ? (
        <div className="big">
          {g[0].toFixed(2)}
          <span className="pm"> ± {g[1].toFixed(2)}</span>
          <span className="unit"> m/s²</span>
        </div>
      ) : (
        <div className="big muted">—</div>
      )}
      <div className="sub">{g ? `measured from ${s.fit?.id === 'fit_first' ? 'the first fall' : 'two flights'}` : 'waiting for the first bounce'}</div>
    </section>
  )
}

export function MemoryVsEyes({ s }: { s: StageState }) {
  const m = s.memory
  if (!m) return null
  const max = Math.max(m.memory_value, m.eyes_value) * 1.1
  const factor = Math.sqrt(m.memory_value / m.eyes_value)
  const differs = Math.abs(m.eyes_value - m.memory_value) > 3 * (m.eyes_err ?? 0) && Math.abs(factor - 1) > 0.1
  return (
    <section className={`card mve ${differs ? 'alert' : ''}`}>
      <div className="label">Memory vs eyes</div>
      <div className="row">
        <span className="who">Memory</span>
        <div className="bar"><div className="fill memory" style={{ width: `${(m.memory_value / max) * 100}%` }} /></div>
        <span className="val">{m.memory_value.toFixed(2)}</span>
      </div>
      <div className="row">
        <span className="who">Eyes</span>
        <div className="bar"><div className="fill eyes" style={{ width: `${(m.eyes_value / max) * 100}%` }} /></div>
        <span className="val">{m.eyes_value.toFixed(2)}</span>
      </div>
      <div className="verdict-line">
        {differs
          ? factor > 1
            ? `Time runs ${factor.toFixed(1)}× slower than reality → slow motion, or not real physics`
            : `Time runs ${(1 / factor).toFixed(1)}× faster than reality → sped up`
          : 'Matches real-world gravity'}
      </div>
    </section>
  )
}

const RACE_TITLES: Record<string, string> = { motion: 'Which law explains the motion?', bounce: 'Which bounce law?' }

export function HypothesisRace({ s, topic = 'motion' }: { s: StageState; topic?: string }) {
  const h = s.races[topic]
  if (!h || h.candidates.length < 2) return null
  const best = Math.max(...h.candidates.map((c) => c.score))
  return (
    <section className="card race">
      <div className="label">{h.label ?? RACE_TITLES[topic] ?? topic}</div>
      {h.candidates.map((c) => (
        <div key={c.name} className={`law ${c.score === best ? 'lead' : ''}`}>
          <div className="law-head">
            <span>{MODEL_NAMES[c.name] ?? c.name}</span>
            <span className="pct">{(c.score * 100).toFixed(c.score > 0.001 && c.score < 0.01 ? 1 : 0)}%</span>
          </div>
          <div className="bar"><div className="fill" style={{ width: `${Math.max(c.score * 100, 0.5)}%` }} /></div>
          <code>{c.equation}</code>
        </div>
      ))}
    </section>
  )
}

/** Every hypothesis race in the run, in the order the engine opened them. */
export function HypothesisRaces({ s }: { s: StageState }) {
  return <>{Object.keys(s.races).map((topic) => <HypothesisRace key={topic} s={s} topic={topic} />)}</>
}

/** What the experiment measured with its own eyes, each with its uncertainty. */
export function Measurements({ s }: { s: StageState }) {
  const m = s.measurement
  if (!m) return null
  // Enough decimals for the value's size, and for one significant figure of its uncertainty.
  const digits = (x: number, err: number | null) => {
    const bySize = Math.abs(x) >= 100 ? 0 : Math.abs(x) >= 10 ? 1 : Math.abs(x) >= 1 ? 2 : 3
    const byErr = err ? Math.max(0, -Math.floor(Math.log10(err))) : 0
    return Math.min(Math.max(bySize, byErr), 4)
  }
  return (
    <section className="card measures">
      <div className="label">Measured from the video</div>
      {m.items.map((it) => (
        <div key={it.label} className="measure">
          <span className="m-label">{it.label}</span>
          <span className="m-val">
            {it.value.toFixed(digits(it.value, it.err))}
            {it.err !== null && (
              <span className="pm">
                {' '}± {it.err >= 0.5 * 10 ** -digits(it.value, it.err) ? it.err.toFixed(digits(it.value, it.err)) : `<${(10 ** -digits(it.value, it.err)).toFixed(digits(it.value, it.err))}`}
              </span>
            )}
            {it.unit && <span className="unit"> {it.unit}</span>}
          </span>
        </div>
      ))}
    </section>
  )
}

/** Equations the engine found on its own, by searching combinations of force terms against the track. */
export function DiscoveredLaws({ s }: { s: StageState }) {
  const laws = Object.values(s.laws)
  if (!laws.length) return null
  return (
    <section className="card laws">
      <div className="label">Law found from the motion</div>
      {laws.map((l) => {
        const runnerUp = l.alternatives[1]
        return (
          <div key={l.id} className="law-found">
            <div className="lf-label">{l.label}</div>
            <div className="lf-eq">{l.equation}</div>
            {l.meaning.map((m) => <div key={m} className="lf-meaning">{m}</div>)}
            <div className="lf-foot">
              searched {l.searched} equations · fits to ±{(l.rmse * 1000).toFixed(l.rmse < 0.001 ? 2 : 1)} mm
              {runnerUp && ` · next best: ${runnerUp.name} (${(runnerUp.score * 100).toFixed(0)}%)`}
            </div>
          </div>
        )
      })}
    </section>
  )
}

function lockedTime(iso: string) {
  const d = new Date(iso)
  return `${d.toLocaleTimeString([], { hour12: false })}.${String(d.getMilliseconds()).padStart(3, '0')}`
}

function PredictionCard({ p, v }: { p: PredictionEvent; v?: VerdictEvent }) {
  const fmt = (x: number) => (p.unit === 's' ? x.toFixed(3) : x.toFixed(1))
  return (
    <div className={`pred ${v ? (v.hit ? 'hit' : 'miss') : 'locked'}`}>
      <div className="pred-top">
        <span className="lock"><LockSimple weight="fill" aria-hidden /> LOCKED · {lockedTime(p.locked_at)}</span>
        <span className="hash" title={p.hash}>#{p.hash.slice(0, 10)}</span>
      </div>
      <div className="pred-text">{p.text}</div>
      <div className="pred-ci">90% band: {fmt(p.ci90[0])} – {fmt(p.ci90[1])} {p.unit}</div>
      {v && (
        <div className="stamp">
          {v.hit ? <><Check weight="bold" aria-hidden /> HIT</> : <><X weight="bold" aria-hidden /> MISS</>} <span>actual {fmt(v.actual)} {p.unit}</span>
        </div>
      )}
    </div>
  )
}

export function Predictions({ s }: { s: StageState }) {
  if (!s.predictions.length) return null
  return (
    <section className="card preds">
      <div className="label">Predictions</div>
      {s.predictions.map((p) => <PredictionCard key={p.id} p={p} v={s.verdicts[p.id]} />)}
    </section>
  )
}

export function Scoreboard({ hits, total }: { hits: number; total: number }) {
  return (
    <section className="card score">
      <div className="label">Scoreboard</div>
      <div className="score-num">{hits}<span>/{total}</span></div>
      <div className="sub">predictions inside their 90% band this session</div>
    </section>
  )
}

export function Explanation({ s }: { s: StageState }) {
  const e = s.explain
  if (!e) return null
  return (
    <section className="card explain">
      <div className="label">Why</div>
      <p>{e.text}</p>
      <div className={`badge ${e.verified ? 'ok' : ''}`}>
        {e.verified ? <><Check weight="bold" aria-hidden /> every number traced to a measurement</> : 'unverified'} · {e.source === 'muse_spark' ? 'Muse Spark' : 'template'}
      </div>
    </section>
  )
}
