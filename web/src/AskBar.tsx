import { useCallback, useEffect, useRef, useState } from 'react'
import { Microphone, Record } from '@phosphor-icons/react'
import { failure } from './api'
import { speakLine, stopVoice } from './voice'

type Turn = { q: string; a?: string; verified?: boolean; error?: string }
type Props = {
  /** "scan:<id>" or "run:<id>": what the questions are about. Null while there is nothing to ask about yet. */
  context: string | null
  voice: boolean
  onHighlight?: (parts: string[]) => void
}

// Captures the microphone as 16 kHz mono floats; the main thread turns them into 16-bit PCM for Muse Voice.
const WORKLET = `class Mic extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true }
}
registerProcessor('mic', Mic)`

function toPcm16(f: Float32Array, fromRate: number): ArrayBuffer {
  const step = fromRate / 16000
  const n = Math.floor(f.length / step)
  const out = new Int16Array(n)
  for (let i = 0; i < n; i++) {
    const v = Math.max(-1, Math.min(1, f[Math.floor(i * step)]))
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff
  }
  return out.buffer
}

function say(text: string) {
  stopVoice()
  speakLine(text, { rate: 1.04 })
}

export default function AskBar({ context, voice, onHighlight }: Props) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [text, setText] = useState('')
  const [listening, setListening] = useState(false)
  const [heard, setHeard] = useState('')
  const [busy, setBusy] = useState(false)
  const fresh = useRef(true)
  const mic = useRef<{ ws: WebSocket; ctx: AudioContext; stream: MediaStream; queue: ArrayBuffer[]; ready: boolean } | null>(null)
  const logRef = useRef<HTMLDivElement>(null)

  // A new scan or run starts a new conversation.
  useEffect(() => { setTurns([]); fresh.current = true }, [context])
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' }) }, [turns])

  const ask = useCallback(async (question: string) => {
    if (!context || !question.trim()) return
    setBusy(true)
    setTurns((t) => [...t, { q: question }])
    const form = new FormData()
    form.set('context', context)
    form.set('question', question)
    if (fresh.current) form.set('reset', 'true')
    fresh.current = false
    try {
      const r = await fetch('/api/ask', { method: 'POST', body: form })
      if (!r.ok) throw await failure(r)
      const body = await r.json()
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, a: body.answer, verified: body.verified } : x)))
      if (voice) say(body.answer)
      onHighlight?.(body.parts ?? [])
    } catch (e) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: e instanceof Error ? e.message : String(e) } : x)))
    } finally {
      setBusy(false)
    }
  }, [context, voice, onHighlight])

  // Push-to-talk: stream the mic to the engine (→ Muse Voice) while held.
  const startListening = useCallback(async () => {
    if (!context || mic.current || busy) return
    stopVoice()
    setHeard('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } })
      const ctx = new AudioContext({ sampleRate: 16000 })
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })))
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/api/voice`)
      ws.binaryType = 'arraybuffer'
      const m = { ws, ctx, stream, queue: [] as ArrayBuffer[], ready: false }
      mic.current = m
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data)
        if (msg.type === 'ready') { m.ready = true; m.queue.forEach((b) => ws.send(b)); m.queue = [] }
        else if (msg.type === 'partial') setHeard(msg.text)
        else if (msg.type === 'final') { setHeard(''); if (msg.text) ask(msg.text) }
        else if (msg.type === 'error') setTurns((t) => [...t, { q: '(voice)', error: msg.message }])
      }
      const node = new AudioWorkletNode(ctx, 'mic')
      node.port.onmessage = (ev) => {
        const buf = toPcm16(ev.data as Float32Array, ctx.sampleRate)
        if (m.ready && ws.readyState === WebSocket.OPEN) ws.send(buf); else m.queue.push(buf)
      }
      ctx.createMediaStreamSource(stream).connect(node)
      setListening(true)
    } catch (e) {
      mic.current = null
      setTurns((t) => [...t, { q: '(microphone)', error: `Microphone unavailable: ${String(e)}. Type your question instead.` }])
    }
  }, [context, busy, ask])

  const stopListening = useCallback(() => {
    const m = mic.current
    if (!m) return
    mic.current = null
    setListening(false)
    m.stream.getTracks().forEach((t) => t.stop())
    m.ctx.close()
    const stop = () => m.ws.send(JSON.stringify({ type: 'stop' }))
    if (m.ws.readyState === WebSocket.OPEN) stop(); else m.ws.addEventListener('open', stop, { once: true })
  }, [])

  // Hold the space bar to talk (when not typing in a field).
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat || (e.target as HTMLElement)?.closest('input, textarea, select')) return
      e.preventDefault(); startListening()
    }
    const up = (e: KeyboardEvent) => { if (e.code === 'Space' && mic.current) { e.preventDefault(); stopListening() } }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up) }
  }, [startListening, stopListening])

  const disabled = !context
  return (
    <div className={`askbar ${disabled ? 'disabled' : ''}`}>
      {turns.length > 0 && (
        <div className="ask-log" ref={logRef}>
          {turns.map((t, i) => (
            <div key={i} className="ask-turn">
              <div className="ask-q">{t.q}</div>
              {t.a && <div className="ask-a">{t.a}{t.verified === false && <span className="ask-flag"> · contains numbers not measured</span>}</div>}
              {t.error && <div className="ask-a err">{t.error}</div>}
              {!t.a && !t.error && <div className="ask-a thinking" aria-label="Thinking"><span /><span /></div>}
            </div>
          ))}
        </div>
      )}
      <div className="ask-row">
        <button className={`mic ${listening ? 'on' : ''}`} disabled={disabled || busy}
          onPointerDown={(e) => { e.preventDefault(); startListening() }} onPointerUp={stopListening} onPointerLeave={() => listening && stopListening()}
          title="Hold to talk (or hold the space bar)">
          {listening ? <><Record weight="fill" aria-hidden /><span className="mic-text">Listening</span></> : <><Microphone weight="bold" aria-hidden /><span className="mic-text">Hold to ask</span></>}
        </button>
        <input className="ask-input" value={listening ? heard : text} disabled={disabled || listening}
          placeholder={disabled ? 'Run an experiment or a scan, then ask about it' : listening ? 'Listening…' : 'Ask about what you see…'}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && text.trim() && !busy) { ask(text); setText('') } }} />
        <button className="chip" disabled={disabled || busy || !text.trim()} onClick={() => { ask(text); setText('') }}>Ask</button>
      </div>
    </div>
  )
}
