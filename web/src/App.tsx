import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRun, getExperiments, getHealth, getRecordEvents, getRecords, getSamples, streamRun, type RecordRow, type Experiment, type Health, type RunInput, type Sample } from './api'
import { cursor, deriveState, objectPaths, timeline } from './derive'
import { drawOverlay, type PathMode, type SimOverlay } from './overlay'
import { DiscoveredLaws, Explanation, HypothesisRaces, Measurements, MemoryVsEyes, Predictions, Readout, Scoreboard } from './panels'
import type { NarrationEvent, PhysEvent, ReconstructionEvent, RunStateEvent, VerdictEvent, VideoInfo } from './types'
import AskBar from './AskBar'
import ReconstructionPanel from './ReconstructionPanel'
import World3D from './World3D'
import WhatIfPanel from './WhatIfPanel'
import ScanView from './ScanView'
import PointWorld from './PointWorld'
import { speakLine, stopVoice } from './voice'
import Logo from './design/Logo'
import { Check, Cube, Flask, LockSimple, Pause, Path, Play, ScanSmiley, SpeakerHigh, SpeakerSlash, UploadSimple, X } from '@phosphor-icons/react'
import './App.css'

type Status = 'idle' | 'analysing' | 'ready' | 'error'

/** Tape-measure inputs an upload needs, in words a person holding the tape understands. */
const PARAM_LABELS: Record<string, string> = {
  ball_diameter_m: 'Ball ⌀ m', table_height_m: 'Table height m', can_diameter_m: 'Can ⌀ m',
  finish_distance_m: 'Course length m', weight_diameter_m: 'Weight ⌀ m', reference_mass_g: 'Known weight g',
  true_mass_g: 'Scale reading g (optional)',
}

const MAX_UPLOAD_MB = 300   // the engine's limit (backend/physlens/api.py MAX_VIDEO_BYTES)
// The engine (and its API keys) only ever runs on the user's machine; a hosted copy of this page has no engine.
const LOCAL_HOST = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)

/** What's wrong with an upload before it is sent, in words that say what to do; undefined when it's fine. */
function uploadProblem(file: File, experiment: string, scale: string, params: Record<string, string>): string | undefined {
  if (file.type && !file.type.startsWith('video/')) return `${file.name} is not a video. Pick an MP4, MOV or WebM clip.`
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `${file.name} is over ${MAX_UPLOAD_MB} MB. Trim the clip to the moment that matters.`
  const positive = (x: string) => x.trim() === '' || Number(x) > 0
  if ((experiment === 'drop' || experiment === 'auto') && !positive(scale))
    return 'The scale must be a positive number (the ball’s width in cm, or pixels per metre), or left empty.'
  const wrong = Object.entries(params).find(([, x]) => !positive(x))
  if (wrong) return `${PARAM_LABELS[wrong[0]] ?? wrong[0]} must be a positive number, or left empty.`
  return undefined
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
  const [mode, setMode] = useState<'lab' | 'scan' | 'points'>('lab')
  const [activeSample, setActiveSample] = useState<string>()

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const lastIdx = useRef(-1)
  // The narration last spoken, so a frame timestamp that steps back after a beat's pause never says it twice.
  const lastBeat = useRef(-1)
  const inBeat = useRef(false)
  const frameTime = useRef<number | null>(null)
  const opts = useRef({ voice, beats })
  opts.current = { voice, beats }

  const tl = useMemo(() => timeline(events), [events])
  const paths = useMemo(() => objectPaths(tl), [tl])
  const recon = useMemo(() => tl.find((e) => e.type === 'reconstruction') as ReconstructionEvent | undefined, [tl])
  const pathsRef = useRef(paths)
  pathsRef.current = paths
  const [pathMode, setPathMode] = useState<PathMode>('sofar')
  const [sims, setSims] = useState<SimOverlay[]>([])
  const simsRef = useRef(sims)
  simsRef.current = sims
  const [records, setRecords] = useState<RecordRow[]>([])
  const [show3d, setShow3d] = useState(false)
  const pathModeRef = useRef(pathMode)
  pathModeRef.current = pathMode
  const tlRef = useRef(tl)
  tlRef.current = tl
  const state = useMemo(() => deriveState(tl, idx), [tl, idx])
  const pxPerM = state.calibration?.px_per_m ?? 480

  useEffect(() => {
    getSamples().then(setSamples).catch(() => setStatusMsg(LOCAL_HOST
      ? 'Engine not reachable: start the backend on port 8000'
      : 'The Lab runs on your own machine, next to the engine that holds the API keys. Here, open the 3D simulator to explore the measured crash tests.'))
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
    stopVoice()
    setCredit(sampleCredit)
    setRunExp(input.experiment ?? 'drop')
    setEvents([])
    setVideo(undefined)
    setIdx(-1)
    lastIdx.current = -1; lastBeat.current = -1
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
      if (result.status === 'error') {
        setStatus('error')
        setStatusMsg(result.error || 'The analysis stopped without a result. Try another clip.')
        return
      }
      setStatus('ready')
      setStatusMsg('Ready')
      getRecords().then(setRecords).catch(() => {})
    } catch (err) {
      setStatus('error')
      setStatusMsg(err instanceof Error ? err.message : String(err))
    }
  }, [])

  // Open a past record: its saved event stream plays back exactly like a fresh run.
  const openRecord = useCallback(async (id: string) => {
    stopVoice()
    try {
      const evs = await getRecordEvents(id)
      const created = evs.find((e) => e.type === 'run_state' && (e as RunStateEvent).video) as RunStateEvent | undefined
      const exp = (evs.find((e) => e.type === 'run_state' && (e as RunStateEvent).experiment) as RunStateEvent | undefined)?.experiment
      setEvents(evs); setVideo(created?.video); setRunId(id); setIdx(-1); lastIdx.current = -1; lastBeat.current = -1
      setRunExp(exp ?? records.find((r) => r.id === id)?.experiment ?? 'drop')
      setCredit(undefined); setActiveSample(undefined)
      setStatus('ready'); setStatusMsg('Ready')
    } catch (e) {
      setStatus('error'); setStatusMsg(`Couldn't open record ${id}. ${e instanceof Error ? e.message : String(e)}`)
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
    // Stopping the voice (pause, seek, another mode, a hidden tab) ends the beat without resuming the video.
    if (opts.current.voice) speakLine(n.text, { onEnd: resume })
    else setTimeout(resume, readingTime)
  }, [])

  // A hidden tab pauses the show; voice.ts already stops the speech. Play picks it up again.
  useEffect(() => {
    const onHide = () => { if (document.hidden) { inBeat.current = false; videoRef.current?.pause() } }
    document.addEventListener('visibilitychange', onHide)
    return () => document.removeEventListener('visibilitychange', onHide)
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
        if (next && at !== lastBeat.current) { lastBeat.current = at; beat(next) }
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
        lastIdx.current = -1; lastBeat.current = -1
      }
      v.play()
    } else {
      v.pause()
      stopVoice()
      inBeat.current = false
    }
  }

  const seek = (t: number) => {
    const v = videoRef.current
    if (!v) return
    stopVoice()
    inBeat.current = false
    v.currentTime = t
    lastIdx.current = cursor(tlRef.current, t)
    lastBeat.current = -1
  }

  // Leaving a mode ends what it was saying and showing.
  const switchMode = (m: typeof mode) => {
    stopVoice()
    inBeat.current = false
    videoRef.current?.pause()
    setMode(m)
  }

  // Voice off mid-line: stop talking and let the show carry on silently.
  const toggleVoice = () => {
    if (voice) {
      stopVoice()
      if (inBeat.current) { inBeat.current = false; videoRef.current?.play() }
    }
    setVoice(!voice)
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
      <header className="app-bar">
        <a className="brand" href="./" title="PhysLens home"><Logo /> PhysLens <span>Lab</span></a>
        <nav className="modes" aria-label="Mode">
          <button className={mode === 'lab' ? 'on' : ''} aria-current={mode === 'lab' ? 'page' : undefined} onClick={() => switchMode('lab')}><Flask weight="bold" aria-hidden /> Lab</button>
          <button className={mode === 'scan' ? 'on' : ''} aria-current={mode === 'scan' ? 'page' : undefined} onClick={() => switchMode('scan')}><ScanSmiley weight="bold" aria-hidden /> Scan</button>
          <button className={mode === 'points' ? 'on' : ''} aria-current={mode === 'points' ? 'page' : undefined} onClick={() => switchMode('points')}><Cube weight="bold" aria-hidden /> 3D points</button>
        </nav>
        <div className="toggles">
          <button className={`toggle ${voice ? 'on' : ''}`} aria-pressed={voice} onClick={toggleVoice}>
            {voice ? <SpeakerHigh weight="bold" aria-hidden /> : <SpeakerSlash weight="bold" aria-hidden />} Voice
          </button>
          {mode === 'lab' && <>
            <button className={`toggle ${beats ? 'on' : ''}`} aria-pressed={beats} title="Pause on each narration line, like a presenter" onClick={() => setBeats(!beats)}>Presenter beats</button>
            <select value={rate} aria-label="Playback speed" onChange={(e) => setRate(Number(e.target.value))}>
              {[0.25, 0.5, 1].map((r) => <option key={r} value={r}>{r}×</option>)}
            </select>
          </>}
        </div>
      </header>
        <div className="sources" hidden={mode !== 'lab'}>
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
              <UploadSimple weight="bold" aria-hidden /> Upload clip
              <input type="file" accept="video/*" disabled={status === 'analysing'}
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  e.target.value = ''
                  if (!file) return
                  const bad = uploadProblem(file, uploadExp, scaleValue, uploadParams)
                  if (bad) { setStatus('error'); setStatusMsg(bad); return }
                  setActiveSample(undefined)
                  const v = Number(scaleValue)
                  if (file && (uploadExp === 'drop' || uploadExp === 'auto')) {
                    const scale = scaleMode === 'ruler' ? { pxPerM: v || undefined } : { diameterCm: v || undefined }
                    start({ file, experiment: uploadExp, ...scale })
                  } else if (file) {
                    const params = Object.fromEntries(Object.entries(uploadParams)
                      .map(([k, x]) => [k, Number(x)] as const).filter(([, x]) => x > 0))
                    start({ file, experiment: uploadExp, params })
                  }
                }} />
            </label>
            <select value={uploadExp} title="Which experiment is in your clip"
              onChange={(e) => chooseUploadExp(e.target.value)}>
              <option value="auto">Detect automatically</option>
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


      {mode === 'points' ? <PointWorld /> : mode === 'scan' ? <ScanView voice={voice} onFilm={(x) => { chooseUploadExp(x); setMode('lab') }} /> : (
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
              <div className="veil" role={status === 'error' ? 'alert' : undefined}>
                {status === 'analysing' && <div className="skeleton" aria-label="Analysing"><span /><span /><span /></div>}
                <div className="veil-title">{status === 'idle' ? 'Point a camera at physics.' : status === 'error' ? 'Something went wrong' : 'Analysing'}</div>
                <div className="veil-msg">{statusMsg}</div>
                {status === 'idle' && (
                  <div className="steps-hint caption-5">
                    <span>1 · pick a sample or upload a clip</span><span>2 · watch predictions lock</span><span>3 · see the verdicts</span>
                  </div>
                )}
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
                <span><LockSimple weight="fill" aria-hidden /> Prediction locked</span>
                <code>#{firstPred.hash.slice(0, 10)} · from frames up to {firstPred.based_on_until_t.toFixed(2)} s</code>
              </div>
            )}
            {showStamp && (
              <div className={`verdict-stamp ${lastVerdict.hit ? 'hit' : 'miss'}`} key={`v-${lastVerdict.id}`}>
                <div className="vs-mark">{lastVerdict.hit ? <><Check weight="bold" aria-hidden /> HIT</> : <><X weight="bold" aria-hidden /> MISS</>}</div>
                <div className="vs-text">{stampPred.text}</div>
                <div className="vs-actual">
                  actual {stampPred.unit === 's' ? lastVerdict.actual.toFixed(3) : lastVerdict.actual.toFixed(1)} {stampPred.unit}
                </div>
              </div>
            )}
            {credit && <div className="credit">{credit}</div>}
          </div>
          <div className="transport">
            <button className="play" disabled={status !== 'ready'} onClick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>{playing ? <Pause weight="fill" aria-hidden /> : <Play weight="fill" aria-hidden />}</button>
            <div className="scrub">
              <input type="range" min={0} max={duration || 1} step={0.001} value={time} disabled={status !== 'ready'}
                onChange={(e) => seek(Number(e.target.value))} />
              {duration > 0 && markers.map((m, i) => (
                <span key={i} className={`marker ${m.type}`} style={{ left: `${(m.t / duration) * 100}%` }} />
              ))}
            </div>
            <span className="clock">{time.toFixed(3)} s</span>
            <button className={`toggle ${show3d ? 'on' : ''}`} title="Simulate the recorded atoms again in a 3D world" onClick={() => setShow3d(!show3d)}><Cube weight="bold" aria-hidden /> 3D world</button>
            <button className="toggle" title="Trajectory lines: what happened"
              onClick={() => setPathMode(pathMode === 'sofar' ? 'whole' : pathMode === 'whole' ? 'off' : 'sofar')}>
              <Path weight="bold" aria-hidden /> Path: {pathMode === 'sofar' ? 'so far' : pathMode === 'whole' ? 'whole run' : 'off'}
            </button>
          </div>
          {recon && status === 'ready' && <ReconstructionPanel recon={recon} time={time} />}
          {status === 'ready' && runId && show3d && <World3D runId={runId} time={time} />}
          <AskBar context={status === 'ready' && runId ? `run:${runId}` : null} voice={voice} />
        </div>

        <aside>
          {status === 'idle' && !events.length && (
            <section className="card getting-started">
              <div className="label">Getting started</div>
              <ol>
                <li>Pick a sample above, or upload your own drop, roll or crash clip.</li>
                <li>Measurements and the law of motion appear here as the video plays.</li>
                <li>Predictions lock before the frames that settle them, then turn into hits or misses.</li>
              </ol>
            </section>
          )}
          {runExp === 'drop' ? <Readout s={state} /> : <Measurements s={state} />}
          <DiscoveredLaws s={state} />
          <MemoryVsEyes s={state} />
          <Predictions s={state} />
          <HypothesisRaces s={state} />
          <Explanation s={state} />
          {status === 'ready' && runId && paths[0] && runExp !== 'crash' && <WhatIfPanel runId={runId} object={paths[0].id} onPaths={setSims} />}
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
          <div className="engine"><span className={`dot ${health ? 'on' : ''}`} />engine · {health ? `${health.tracker} · ${health.agent}` : LOCAL_HOST ? 'offline: start the backend on port 8000' : <>runs locally · <a href="./sim.html">open the simulator</a></>}</div>
        </aside>
      </main>
      )}
    </div>
  )
}
