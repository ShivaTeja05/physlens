import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRun, getExperiments, getHealth, getRecordEvents, getRecords, getSamples, streamRun, type RecordRow, type Experiment, type Health, type RunInput, type Sample } from './api'
import { cursor, deriveState, objectPaths, timeline } from './derive'
import { drawOverlay, type PathMode, type SimOverlay } from './overlay'
import { DiscoveredLaws, Explanation, HypothesisRaces, Measurements, MemoryVsEyes, Predictions, Readout, Scoreboard } from './panels'
import type { NarrationEvent, PhysEvent, RunStateEvent, VerdictEvent, VideoInfo } from './types'
import AskBar from './AskBar'
import ScanView from './ScanView'
import './App.css'

type Status = 'idle' | 'analysing' | 'ready' | 'error'

/** Tape-measure inputs an upload needs, in words a person holding the tape understands. */
const PARAM_LABELS: Record<string, string> = {
  ball_diameter_m: 'Ball ⌀ m', table_height_m: 'Table height m', can_diameter_m: 'Can ⌀ m',
  finish_distance_m: 'Course length m', weight_diameter_m: 'Weight ⌀ m', reference_mass_g: 'Known weight g',
  true_mass_g: 'Scale reading g (optional)',
}

function pickVoice(): SpeechSynthesisVoice | undefined {
  const voices = speechSynthesis.getVoices()
  const preferred = ['Samantha', 'Daniel', 'Google UK English Female', 'Karen', 'Moira', 'Serena']
  return voices.find((v) => preferred.some((p) => v.name.includes(p))) ?? voices.find((v) => v.lang.startsWith('en'))
}

export default function App() {
  const [samples, setSamples] = useState<Sample[]>([])
  const [health, setHealth] = useState<Health>()
  const [events, setEvents] = useState<PhysEvent[]>([])
  const [status, setStatus] = useState<Status>('idle')
  const [statusMsg, setStatusMsg] = useState('Pick a clip to analyse')
  const [video, setVideo] = useState<VideoInfo>()
  const [idx, setIdx] = useState(-1)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [voice, setVoice] = useState(true)
  const [beats, setBeats] = useState(true)
  const [rate, setRate] = useState(0.5)
  const [session, setSession] = useState<Record<string, { hits: number; total: number }>>({})
  const [runId, setRunId] = useState<string>()
  const [credit, setCredit] = useState<string>()
  const [scaleMode, setScaleMode] = useState<'ball' | 'ruler'>('ball')
  const [scaleValue, setScaleValue] = useState('6.5')
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [uploadExp, setUploadExp] = useState('drop')
  const [uploadParams, setUploadParams] = useState<Record<string, string>>({})
  const [runExp, setRunExp] = useState('drop')
  const [mode, setMode] = useState<'lab' | 'scan'>('lab')
  const [activeSample, setActiveSample] = useState<string>()

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const lastIdx = useRef(-1)
  const inBeat = useRef(false)
  const frameTime = useRef<number | null>(null)
  const opts = useRef({ voice, beats })
  opts.current = { voice, beats }

  const tl = useMemo(() => timeline(events), [events])
  const paths = useMemo(() => objectPaths(tl), [tl])
  const pathsRef = useRef(paths)
  pathsRef.current = paths
  const [pathMode, setPathMode] = useState<PathMode>('sofar')
  const [sims] = useState<SimOverlay[]>([])
  const simsRef = useRef(sims)
  simsRef.current = sims
  const [records, setRecords] = useState<RecordRow[]>([])
  const pathModeRef = useRef(pathMode)
  pathModeRef.current = pathMode
  const tlRef = useRef(tl)
  tlRef.current = tl
  const state = useMemo(() => deriveState(tl, idx), [tl, idx])
  const pxPerM = state.calibration?.px_per_m ?? 480

  useEffect(() => {
    getSamples().then(setSamples).catch(() => setStatusMsg('Engine not reachable: start the backend on port 8000'))
    getHealth().then(setHealth).catch(() => {})
    getExperiments().then(setExperiments).catch(() => {})
    getRecords().then(setRecords).catch(() => {})
  }, [])

  // Keep the session scoreboard in step with verdicts as they appear.
  useEffect(() => {
    if (!runId) return
    const verdicts = Object.values(state.verdicts)
    setSession((prev) => {
      const cur = prev[runId]
      const next = { hits: verdicts.filter((v) => v.hit).length, total: verdicts.length }
      if (cur && cur.hits >= next.hits && cur.total >= next.total) return prev
      return { ...prev, [runId]: next }
    })
  }, [state.verdicts, runId])

  const start = useCallback(async (input: RunInput, sampleCredit?: string) => {
    speechSynthesis.cancel()
    setCredit(sampleCredit)
    setRunExp(input.experiment ?? 'drop')
    setEvents([])
    setVideo(undefined)
    setIdx(-1)
    lastIdx.current = -1
    inBeat.current = false
    setStatus('analysing')
    setStatusMsg(input.file ? `Uploading ${input.file.name}…` : 'Starting…')
    try {
      const id = await createRun({ experiment: 'drop', ...input })
      setRunId(id)
      const collected: PhysEvent[] = []
      const result = await streamRun(id, (e) => {
        collected.push(e)
        if (e.type === 'run_state') {
          const rs = e as RunStateEvent
          if (rs.message) setStatusMsg(rs.message)
          if (rs.experiment) setRunExp(rs.experiment)
          if (rs.video) setVideo(rs.video)
        }
      })
      setEvents(collected)
      if (result === 'error') {
        setStatus('error')
        return
      }
      setStatus('ready')
      setStatusMsg('Ready')
      getRecords().then(setRecords).catch(() => {})
    } catch (err) {
      setStatus('error')
      setStatusMsg(String(err))
    }
  }, [])

  // Open a past record: its saved event stream plays back exactly like a fresh run.
  const openRecord = useCallback(async (id: string) => {
    speechSynthesis.cancel()
    try {
      const evs = await getRecordEvents(id)
      const created = evs.find((e) => e.type === 'run_state' && (e as RunStateEvent).video) as RunStateEvent | undefined
      const exp = (evs.find((e) => e.type === 'run_state' && (e as RunStateEvent).experiment) as RunStateEvent | undefined)?.experiment
      setEvents(evs); setVideo(created?.video); setRunId(id); setIdx(-1); lastIdx.current = -1
      setRunExp(exp ?? records.find((r) => r.id === id)?.experiment ?? 'drop')
      setCredit(undefined); setActiveSample(undefined)
      setStatus('ready'); setStatusMsg('Ready')
    } catch (e) {
      setStatus('error'); setStatusMsg(`couldn't open ${id}: ${String(e)}`)
    }
  }, [records])

  // A "beat": pause on a narration line, say it, then carry on, like a presenter.
  const beat = useCallback((n: NarrationEvent) => {
    const v = videoRef.current
    if (!v) return
    inBeat.current = true
    v.pause()
    let done = false
    const resume = () => {
      if (done || !inBeat.current) return
      done = true
      inBeat.current = false
      if (videoRef.current && videoRef.current.currentTime < videoRef.current.duration - 0.01) videoRef.current.play()
    }
    const readingTime = 1400 + n.text.length * 28
    if (opts.current.voice && 'speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(n.text)
      const chosen = pickVoice()
      if (chosen) u.voice = chosen
      u.rate = 1.03
      u.onend = resume
      u.onerror = resume
      speechSynthesis.speak(u)
      // Some browsers never fire onend; never leave the show stuck on a pause.
      setTimeout(resume, readingTime * 2.5 + 2000)
    } else {
      setTimeout(resume, readingTime)
    }
  }, [])

  // Render loop: draw overlays every frame, update panels only when the event cursor moves.
  useEffect(() => {
    let raf = 0
    const loop = () => {
      raf = requestAnimationFrame(loop)
      const v = videoRef.current, c = canvasRef.current, stage = stageRef.current
      if (!v || !c || !stage) return
      const events = tlRef.current
      // While playing, use the timestamp of the frame actually on screen; currentTime runs ahead of it.
      const t = !v.paused && frameTime.current !== null ? frameTime.current : v.currentTime
      const i = cursor(events, t)

      if (i < lastIdx.current) lastIdx.current = i // seeked backwards
      if (!inBeat.current && i > lastIdx.current) {
        let next: NarrationEvent | undefined
        let at = i
        if (opts.current.beats && !v.paused) {
          for (let j = lastIdx.current + 1; j <= i; j++) {
            const e = events[j]
            if (e.type === 'narration') { next = e; at = j; break }
          }
        }
        lastIdx.current = at
        if (next) beat(next)
      }

      const dpr = window.devicePixelRatio || 1
      const w = stage.clientWidth, h = stage.clientHeight
      if (c.width !== w * dpr || c.height !== h * dpr) {
        c.width = w * dpr
        c.height = h * dpr
        c.style.width = `${w}px`
        c.style.height = `${h}px`
      }
      const ctx = c.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const vw = v.videoWidth || 1280, vh = v.videoHeight || 720
      const scale = Math.min(w / vw, h / vh)
      const view = { scale, ox: (w - vw * scale) / 2, oy: (h - vh * scale) / 2, pxPerM, vw, vh }
      drawOverlay(ctx, deriveState(events, i, t), view, t, pathsRef.current, pathModeRef.current, simsRef.current)

      setIdx((prev) => (prev === i ? prev : i))
      setTime(t)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [beat, pxPerM])

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate
  }, [rate, video])

  // Timestamp of each presented video frame, so overlays line up with the picture exactly.
  useEffect(() => {
    const v = videoRef.current
    frameTime.current = null
    if (!v || !('requestVideoFrameCallback' in v)) return
    let id = 0
    const onFrame = (_now: number, meta: VideoFrameCallbackMetadata) => {
      frameTime.current = meta.mediaTime
      id = v.requestVideoFrameCallback(onFrame)
    }
    id = v.requestVideoFrameCallback(onFrame)
    return () => v.cancelVideoFrameCallback(id)
  }, [video])

  // Panels follow the video even when animation frames are throttled (paused video, background tab).
  const syncPanels = () => {
    const v = videoRef.current
    if (!v) return
    const i = cursor(tlRef.current, v.currentTime)
    setIdx((prev) => (prev === i ? prev : i))
    setTime(v.currentTime)
  }

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      if (v.ended || v.currentTime >= v.duration - 0.01) {
        v.currentTime = 0
        lastIdx.current = -1
      }
      v.play()
    } else {
      v.pause()
      speechSynthesis.cancel()
      inBeat.current = false
    }
  }

  const seek = (t: number) => {
    const v = videoRef.current
    if (!v) return
    speechSynthesis.cancel()
    inBeat.current = false
    v.currentTime = t
    lastIdx.current = cursor(tlRef.current, t)
  }

  const chooseUploadExp = (id: string) => {
    setUploadExp(id)
    const card = experiments.find((x) => x.id === id)
    setUploadParams(Object.fromEntries(Object.entries(card?.params ?? {})
      .filter(([k]) => k in PARAM_LABELS).map(([k, x]) => [k, String(x)])))
  }

  const totals = Object.values(session).reduce((a, b) => ({ hits: a.hits + b.hits, total: a.total + b.total }), { hits: 0, total: 0 })
  const duration = video?.duration ?? 0
  const markers = tl.filter((e) => e.type === 'prediction' || e.type === 'verdict')

  // Keynote moments, tied to video time so they replay exactly when scrubbing: a title card as the clip opens,
  // a flash when the predictions lock, and a stamp when reality settles each one.
  const card = experiments.find((x) => x.id === runExp)
  const t0 = tl[0]?.t ?? 0
  const firstPred = state.predictions[0]
  const showTitle = status === 'ready' && card && time - t0 < 1.2 && !firstPred
  const lastVerdict = Object.values(state.verdicts).reduce<VerdictEvent | undefined>((a, b) => (!a || b.t > a.t ? b : a), undefined)
  const stampPred = lastVerdict && state.predictions.find((p) => p.id === lastVerdict.id)
  const showStamp = status === 'ready' && lastVerdict && stampPred && time - lastVerdict.t < 1.3
  const showLock = status === 'ready' && firstPred && time - firstPred.t < 0.9 && !lastVerdict

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">◉</span>
          <div>
            <h1>PhysLens Lab</h1>
            <p>Every camera, a physics instrument</p>
          </div>
        </div>
        <div className="modes">
          <button className={mode === 'lab' ? 'on' : ''} onClick={() => setMode('lab')}>Lab</button>
          <button className={mode === 'scan' ? 'on' : ''} onClick={() => { speechSynthesis.cancel(); videoRef.current?.pause(); setMode('scan') }}>Scan</button>
        </div>
        <div className="sources" style={mode !== 'lab' ? { display: 'none' } : undefined}>
          {[['Drop & bounce', samples.filter((s) => (s.experiment ?? 'drop') === 'drop')],
            ['Stage experiments', samples.filter((s) => (s.experiment ?? 'drop') !== 'drop')]].map(([title, group]) => (
            (group as Sample[]).length > 0 && (
              <div key={title as string} className="chip-group">
                <span className="group-label">{title as string}</span>
                {(group as Sample[]).map((s) => (
                  <button key={s.name} className={`chip ${activeSample === s.name ? 'active' : ''}`}
                    disabled={status === 'analysing'}
                    onClick={() => { setActiveSample(s.name); start({ sample: s.name, experiment: s.experiment }, s.credit) }}>
                    {s.label ?? s.name}
                  </button>
                ))}
              </div>
            )
          ))}
          <div className="upload-group">
            <label className={`chip upload ${status === 'analysing' ? 'disabled' : ''}`}>
              Upload clip
              <input type="file" accept="video/*" disabled={status === 'analysing'}
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) setActiveSample(undefined)
                  const v = Number(scaleValue)
                  if (file && (uploadExp === 'drop' || uploadExp === 'auto')) {
                    const scale = scaleMode === 'ruler' ? { pxPerM: v || undefined } : { diameterCm: v || undefined }
                    start({ file, experiment: uploadExp, ...scale })
                  } else if (file) {
                    const params = Object.fromEntries(Object.entries(uploadParams)
                      .map(([k, x]) => [k, Number(x)] as const).filter(([, x]) => x > 0))
                    start({ file, experiment: uploadExp, params })
                  }
                  e.target.value = ''
                }} />
            </label>
            <select value={uploadExp} title="Which experiment is in your clip"
              onChange={(e) => chooseUploadExp(e.target.value)}>
              <option value="auto">Auto-detect</option>
              {(experiments.length ? experiments : [{ id: 'drop', title: 'Drop & bounce' }]).map((x) => (
                <option key={x.id} value={x.id}>{x.title}</option>
              ))}
            </select>
            {uploadExp === 'drop' || uploadExp === 'auto' ? (
              <>
                <select value={scaleMode} title="How to get the scale"
                  onChange={(e) => { const m = e.target.value as 'ball' | 'ruler'; setScaleMode(m); setScaleValue(m === 'ball' ? '6.5' : '') }}>
                  <option value="ball">Ball ⌀ cm</option>
                  <option value="ruler">Ruler px/m</option>
                </select>
                <input className="scale" inputMode="decimal" value={scaleValue} placeholder={scaleMode === 'ball' ? '6.5' : 'e.g. 161'}
                  onChange={(e) => setScaleValue(e.target.value)} />
              </>
            ) : (
              [...Object.keys(uploadParams), ...(uploadExp === 'weigh' && !('true_mass_g' in uploadParams) ? ['true_mass_g'] : [])]
                .map((k) => (
                  <label key={k} className="param-field">
                    <span>{PARAM_LABELS[k]}</span>
                    <input className="scale param" inputMode="decimal" placeholder="—"
                      value={uploadParams[k] ?? ''} onChange={(e) => setUploadParams({ ...uploadParams, [k]: e.target.value })} />
                  </label>
                ))
            )}
          </div>
        </div>
        <div className="toggles">
          <button className={`toggle ${voice ? 'on' : ''}`} onClick={() => setVoice(!voice)}>Voice</button>
          {mode === 'lab' && <>
            <button className={`toggle ${beats ? 'on' : ''}`} onClick={() => setBeats(!beats)}>Presenter beats</button>
            <select value={rate} onChange={(e) => setRate(Number(e.target.value))}>
              {[0.25, 0.5, 1].map((r) => <option key={r} value={r}>{r}×</option>)}
            </select>
          </>}
        </div>
      </header>

      {mode === 'scan' ? <ScanView voice={voice} onFilm={(x) => { chooseUploadExp(x); setMode('lab') }} /> : (
      <main>
        <div className="stage-col">
          <div className="stage" ref={stageRef}>
            {video && (
              <video ref={videoRef} src={video.url} muted playsInline preload="auto"
                onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
                onSeeked={syncPanels} onTimeUpdate={syncPanels}
                onLoadedData={() => { if (videoRef.current) videoRef.current.playbackRate = rate }} />
            )}
            <canvas ref={canvasRef} />
            {status !== 'ready' && (
              <div className="veil">
                {status === 'analysing' && <div className="spinner" />}
                <div className="veil-title">{status === 'idle' ? 'Point a camera at physics.' : status === 'error' ? 'Something went wrong' : 'Analysing'}</div>
                <div className="veil-msg">{statusMsg}</div>
              </div>
            )}
            {state.narration && status === 'ready' && <div className="caption" key={state.narration.t}>{state.narration.text}</div>}
            {showTitle && (
              <div className="lower-third">
                <div className="lt-kicker">Experiment</div>
                <div className="lt-title">{card.title}</div>
                <div className="lt-hook">{card.hook}</div>
              </div>
            )}
            {showLock && (
              <div className="lock-flash" key={`lock-${firstPred.t}`}>
                <span>🔒 Prediction locked</span>
                <code>#{firstPred.hash.slice(0, 10)} · from frames up to {firstPred.based_on_until_t.toFixed(2)} s</code>
              </div>
            )}
            {showStamp && (
              <div className={`verdict-stamp ${lastVerdict.hit ? 'hit' : 'miss'}`} key={`v-${lastVerdict.id}`}>
                <div className="vs-mark">{lastVerdict.hit ? '✓ HIT' : '✗ MISS'}</div>
                <div className="vs-text">{stampPred.text}</div>
                <div className="vs-actual">
                  actual {stampPred.unit === 's' ? lastVerdict.actual.toFixed(3) : lastVerdict.actual.toFixed(1)} {stampPred.unit}
                </div>
              </div>
            )}
            {credit && <div className="credit">{credit}</div>}
          </div>
          <div className="transport">
            <button className="play" disabled={status !== 'ready'} onClick={togglePlay}>{playing ? '❚❚' : '▶'}</button>
            <div className="scrub">
              <input type="range" min={0} max={duration || 1} step={0.001} value={time} disabled={status !== 'ready'}
                onChange={(e) => seek(Number(e.target.value))} />
              {duration > 0 && markers.map((m, i) => (
                <span key={i} className={`marker ${m.type}`} style={{ left: `${(m.t / duration) * 100}%` }} />
              ))}
            </div>
            <span className="clock">{time.toFixed(3)} s</span>
            <button className="toggle" title="Trajectory lines: what happened"
              onClick={() => setPathMode(pathMode === 'sofar' ? 'whole' : pathMode === 'whole' ? 'off' : 'sofar')}>
              Path: {pathMode === 'sofar' ? 'so far' : pathMode === 'whole' ? 'whole run' : 'off'}
            </button>
          </div>
          <AskBar context={status === 'ready' && runId ? `run:${runId}` : null} voice={voice} />
        </div>

        <aside>
          {runExp === 'drop' ? <Readout s={state} /> : <Measurements s={state} />}
          <DiscoveredLaws s={state} />
          <MemoryVsEyes s={state} />
          <Predictions s={state} />
          <HypothesisRaces s={state} />
          <Explanation s={state} />
          <Scoreboard hits={totals.hits} total={totals.total} />
          {records.length > 0 && (
            <section className="card records">
              <div className="label">Records <span className="knowledge rec">records/physlens.db</span></div>
              {records.slice(0, 30).map((r) => (
                <button key={r.id} className={`rec-row ${r.id === runId ? 'on' : ''}`} disabled={status === 'analysing'} onClick={() => openRecord(r.id)}>
                  <span className="rec-id">{r.id}</span>
                  <span className="rec-meta">{r.experiment ?? '?'} · {r.contacts} contacts · {r.hits}/{r.verdicts} hits</span>
                </button>
              ))}
            </section>
          )}
          <div className="engine">engine · {health ? `${health.tracker} · ${health.agent}` : 'offline'}</div>
        </aside>
      </main>
      )}
    </div>
  )
}
