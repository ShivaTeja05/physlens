import { useCallback, useEffect, useRef, useState } from 'react'
import { getScanSamples, scanStatus, startScan } from './api'
import AskBar from './AskBar'
import { drawScan, edgePoints } from './scanOverlay'
import type { ScanResult, ScanSample, ScanStage } from './scanTypes'

const STAGE_TEXT: Record<ScanStage, string> = {
  queued: 'Starting the scan', looking: 'Muse Spark is looking: what is this?', outlining: 'SAM 3.1 is outlining the parts',
  explaining: 'Measuring, and checking every number', done: '', error: 'Scan failed',
}
const SCENE_LABEL: Record<string, string> = {
  structure: 'Structure', astronomy: 'Astronomy', motion: 'Motion', activity: 'Activity', device: 'Machine',
  nature: 'Nature', other: 'Scene',
}
const EXPERIMENT_TITLE: Record<string, string> = {
  drop: 'Drop & bounce', cup: 'Catch the cup', cans: 'Mystery Cans', weigh: 'Weigh it by watching', free_motion: 'Free motion',
}

function speak(text: string, onEnd: () => void) {
  if (!('speechSynthesis' in window)) { setTimeout(onEnd, 1500 + text.length * 30); return }
  const u = new SpeechSynthesisUtterance(text)
  const v = speechSynthesis.getVoices().find((x) => /Samantha|Daniel|Google UK English Female|Karen/.test(x.name))
  if (v) u.voice = v
  u.rate = 1.03
  let done = false
  const end = () => { if (!done) { done = true; onEnd() } }
  u.onend = end
  u.onerror = end
  speechSynthesis.speak(u)
  setTimeout(end, 4000 + text.length * 90)   // some browsers never fire onend
}

export default function ScanView({ voice, onFilm }: { voice: boolean; onFilm: (experiment: string) => void }) {
  const [samples, setSamples] = useState<ScanSample[]>([])
  const [result, setResult] = useState<ScanResult>()
  const [stage, setStage] = useState<ScanStage | 'idle'>('idle')
  const [error, setError] = useState<string>()
  const [credit, setCredit] = useState<string>()
  const [live, setLive] = useState(false)
  const [step, setStep] = useState<number | null>(null)
  const [refPart, setRefPart] = useState('')
  const [refSize, setRefSize] = useState('')

  const stageRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const anim = useRef({ stage: 'queued' as ScanStage, stageAt: 0, partsAt: null as number | null,
    edges: null as Float32Array | null, highlight: null as Set<string> | null })
  const resultRef = useRef<ScanResult>()
  resultRef.current = result
  const lastSource = useRef<{ sample?: string; image?: Blob }>({})

  useEffect(() => { getScanSamples().then(setSamples).catch(() => {}) }, [])

  // Draw loop.
  useEffect(() => {
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const c = canvasRef.current, st = stageRef.current, r = resultRef.current
      if (!c || !st) return
      const dpr = window.devicePixelRatio || 1
      const w = st.clientWidth, h = st.clientHeight
      if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; c.style.width = `${w}px`; c.style.height = `${h}px` }
      const ctx = c.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      if (!r || live) { ctx.clearRect(0, 0, w, h); return }
      const scale = Math.min(w / r.image.width, h / r.image.height)
      const view = { scale, ox: (w - r.image.width * scale) / 2, oy: (h - r.image.height * scale) / 2, w, h }
      drawScan(ctx, r, view, { ...anim.current, now: performance.now() })
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [live])

  const narrate = useCallback((r: ScanResult) => {
    speechSynthesis.cancel()
    const steps = r.how_it_works ?? []
    const lines = [r.headline || r.summary || '', ...steps.map((s) => s.text)]
    let i = 0
    const next = () => {
      if (i >= lines.length) { setStep(null); anim.current.highlight = null; return }
      const k = i - 1
      setStep(k >= 0 ? k : null)
      anim.current.highlight = k >= 0 && steps[k].parts?.length ? new Set(steps[k].parts) : null
      const line = lines[i++]
      if (voice && line) speak(line, next)
      else setTimeout(next, 1800 + line.length * 35)
    }
    next()
  }, [voice])

  const run = useCallback(async (input: { sample?: string; image?: Blob; reference?: object }, sampleCredit?: string) => {
    speechSynthesis.cancel()
    setError(undefined)
    setStep(null)
    setCredit(sampleCredit)
    lastSource.current = { sample: input.sample, image: input.image }
    anim.current = { stage: 'queued', stageAt: performance.now(), partsAt: null, edges: null, highlight: null }
    setStage('queued')
    try {
      const id = await startScan(input)
      let prev: ScanStage | null = null
      for (;;) {
        const s = await scanStatus(id)
        if (s.stage !== prev) {
          prev = s.stage
          anim.current.stage = s.stage
          anim.current.stageAt = performance.now()
          setStage(s.stage)
        }
        if (anim.current.partsAt === null && s.result.parts?.some((p) => p.instances.length)) anim.current.partsAt = performance.now()
        setResult(s.result)
        if (s.stage === 'done') { narrate(s.result); return }
        if (s.stage === 'error') { setError(s.error ?? 'scan failed'); return }
        await new Promise((r) => setTimeout(r, 600))
      }
    } catch (e) {
      setStage('error')
      setError(String(e))
    }
  }, [narrate])

  // Webcam: live view, then freeze a frame and scan it.
  const toggleLive = async () => {
    if (live) {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setLive(false)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 1280 } })
      streamRef.current = stream
      setLive(true)
      setResult(undefined)
      setStage('idle')
      requestAnimationFrame(() => { if (videoRef.current) videoRef.current.srcObject = stream })
    } catch (e) {
      setError(`Camera unavailable: ${String(e)}`)
    }
  }
  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), [])

  const captureAndScan = () => {
    const v = videoRef.current
    if (!v || !v.videoWidth) return
    const c = document.createElement('canvas')
    c.width = v.videoWidth; c.height = v.videoHeight
    c.getContext('2d')!.drawImage(v, 0, 0)
    c.toBlob((blob) => {
      if (!blob) return
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setLive(false)
      run({ image: blob })
    }, 'image/jpeg', 0.9)
  }

  const rescanWithScale = () => {
    const size = Number(refSize)
    if (!refPart || !(size > 0)) return
    run({ ...lastSource.current, reference: { part: refPart, size_m: size, axis: 'height' } }, credit)
  }

  const busy = stage === 'queued' || stage === 'looking' || stage === 'outlining' || stage === 'explaining'
  const parts = result?.parts ?? []
  const found = parts.filter((p) => p.instances.length)

  return (
    <main>
      <div className="stage-col">
        <div className="scan-bar">
          {samples.map((s) => (
            <button key={s.name} className="chip" disabled={busy} onClick={() => run({ sample: s.name }, s.credit)}>{s.label ?? s.name}</button>
          ))}
          <label className={`chip upload ${busy ? 'disabled' : ''}`}>
            Upload photo
            <input type="file" accept="image/*" disabled={busy}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) run({ image: f }); e.target.value = '' }} />
          </label>
          <button className={`chip ${live ? 'active' : ''}`} disabled={busy} onClick={toggleLive}>{live ? 'Stop camera' : 'Camera'}</button>
          {live && <button className="chip scan-go" onClick={captureAndScan}>◎ Scan</button>}
        </div>
        <div className="stage scan-stage" ref={stageRef}>
          {live && <video ref={videoRef} autoPlay muted playsInline />}
          {!live && result && <img ref={imgRef} src={result.image.url} alt=""
            onLoad={(e) => { anim.current.edges = edgePoints(e.currentTarget) }} />}
          <canvas ref={canvasRef} />
          {live && <div className="live-hud"><span className="rec">● LIVE</span> point at anything, then Scan</div>}
          {!live && stage === 'idle' && (
            <div className="veil">
              <div className="veil-title">Point it at anything.</div>
              <div className="veil-msg">A building, the sky, a machine, someone doing something. Pick a sample, upload a photo, or use the camera.</div>
            </div>
          )}
          {!live && result?.title && (
            <div className="scan-hud">
              <div className="lt-kicker">{SCENE_LABEL[result.scene_type ?? 'other']} · scan</div>
              <div className="scan-title">{result.title}</div>
            </div>
          )}
          {!live && busy && <div className="scan-status"><span className="dot" />{STAGE_TEXT[stage as ScanStage]}</div>}
          {!live && stage === 'done' && result?.headline && <div className="caption" key={result.id}>{result.headline}</div>}
          {credit && !live && <div className="credit">{credit}</div>}
        </div>
        <AskBar context={stage === 'done' && result?.id ? `scan:${result.id}` : null} voice={voice}
          onHighlight={(ids) => {
            speechSynthesis.cancel()
            anim.current.highlight = ids.length ? new Set(ids) : null
            setStep(null)
          }} />
      </div>

      <aside>
        {error && <section className="card"><div className="label">Error</div><p>{error}</p></section>}
        {result?.summary && (
          <section className="card">
            <div className="label">What it is</div>
            <div className="scan-what">{result.summary}</div>
          </section>
        )}
        {found.length > 0 && (
          <section className="card measures">
            <div className="label">Measured from the image</div>
            {found.map((p) => (
              <div key={p.id} className="scan-part">
                <div className="sp-head"><span className="sp-dot" style={{ background: p.color }} />{p.name}<span className="sp-role">{p.role}</span></div>
                {p.measures.map((m) => (
                  <div key={m.label} className="measure">
                    <span className="m-label">{m.label}</span>
                    <span className="m-val">{m.value}<span className="unit"> {m.unit}</span></span>
                  </div>
                ))}
              </div>
            ))}
            {(result?.relations ?? []).filter((r) => r.contact !== undefined).map((r, i) => (
              <div key={i} className="measure">
                <span className="m-label">{parts.find((p) => p.id === r.from)?.name} {r.label} {parts.find((p) => p.id === r.to)?.name}</span>
                <span className={`m-val ${r.contact ? 'ok' : 'warn'}`}>{r.contact ? '✓ touching' : 'not touching'}</span>
              </div>
            ))}
          </section>
        )}
        {(result?.checks?.length ?? 0) > 0 && (
          <section className={`card mve ${result!.checks!.some((c) => !c.ok) ? 'alert' : ''}`}>
            <div className="label">Words vs eyes</div>
            {result!.checks!.map((c, i) => (
              <div key={i} className="check">
                <div className="sub">{c.claim}</div>
                <div className={`check-verdict ${c.ok ? 'ok' : 'bad'}`}>{c.ok ? '✓ ' : '✗ '}{c.text}</div>
              </div>
            ))}
          </section>
        )}
        {(result?.how_it_works?.length ?? 0) > 0 && (
          <section className="card how">
            <div className="label">How it works <span className="knowledge">background knowledge, not measured</span></div>
            <ol>
              {result!.how_it_works!.map((s, i) => (
                <li key={i} className={step === i ? 'active' : ''}
                  onMouseEnter={() => { anim.current.highlight = s.parts?.length ? new Set(s.parts) : null }}
                  onMouseLeave={() => { if (step === null) anim.current.highlight = null }}>{s.text}</li>
              ))}
            </ol>
          </section>
        )}
        {(result?.seen?.length ?? 0) > 0 && (
          <section className="card explain">
            <div className="label">What the camera saw</div>
            {result!.seen!.map((s, i) => <p key={i} className="seen-line">{s.text}</p>)}
            <div className={`badge ${result!.seen!.every((s) => s.verified) ? 'ok' : ''}`}>
              {result!.seen!.every((s) => s.verified) ? '✓ every number traced to a measurement' : 'some lines unverified'} · {result!.explain_source === 'muse_spark' ? 'Muse Spark' : 'template'}
            </div>
          </section>
        )}
        {stage === 'done' && found.length > 0 && (
          <section className="card">
            <div className="label">Real sizes</div>
            <div className="sub">One known size turns pixels into metres for things at about the same distance.</div>
            <div className="ref-row">
              <select value={refPart} onChange={(e) => setRefPart(e.target.value)}>
                <option value="">pick a part…</option>
                {found.filter((p) => !p.many).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <input className="scale" inputMode="decimal" placeholder="height m" value={refSize} onChange={(e) => setRefSize(e.target.value)} />
              <button className="chip" onClick={rescanWithScale} disabled={!refPart || !(Number(refSize) > 0)}>Measure</button>
            </div>
          </section>
        )}
        {stage === 'done' && ((result?.measure_ideas?.length ?? 0) > 0 || (result?.video_experiment && result.video_experiment !== 'none')) && (
          <section className="card">
            <div className="label">Measure it for real</div>
            {result!.measure_ideas!.map((m, i) => <div key={i} className="idea">◇ {m}</div>)}
            {result!.video_experiment && result!.video_experiment !== 'none' && EXPERIMENT_TITLE[result!.video_experiment] && (
              <div className="film">
                <div className="sub">{result!.video_hint}</div>
                {result!.video_experiment !== 'free_motion' && (
                  <button className="chip active" onClick={() => onFilm(result!.video_experiment!)}>
                    Film it → {EXPERIMENT_TITLE[result!.video_experiment!]}
                  </button>
                )}
              </div>
            )}
          </section>
        )}
      </aside>
    </main>
  )
}
