import { Canvas, useThree } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

// Mirrors backend/physlens/pointworld.py (format physlens-pointworld/1) and the notes benchmark/world_events.py and
// benchmark/crash3d.py add (events, the rigid body at each moment, structures, checks).
type Group = { tag: number; start: number; count: number; frame?: number; t?: number }
type Box = { frame: number; t: number; points: number; centre: [number, number, number]; size: [number, number, number]; yaw: number }
type Rigid = { frame: number; t: number; centre: [number, number, number]; size: [number, number, number]; yaw: number }
type WorldEvent = { t: number; kind: string; label: string; detail?: string | null; window?: [number, number]; frame?: number; truth?: string }
type Structure = { name: string; label: string; from: [number, number, number]; to: [number, number, number] }
type Header = {
  name: string; title: string; video: string; fps: number; points: number; static_points: number; object_points: number
  upsample: number; density_note: string; origin: [number, number, number]; step: [number, number, number]
  groups: Group[]; object: string; boxes: Box[]; object_size_median_m: { length: number; height: number; width: number }
  camera: { position: [number, number, number]; forward: [number, number, number]; fov_deg: number; moving: boolean }
  scale_from: string; notes: Record<string, unknown>; decimated_from?: number; bin_parts?: string[]; bin_bytes?: number
}
type Loaded = { header: Header; positions: Float32Array; colours: Uint8Array }

async function load(base: string, name: string): Promise<Loaded> {
  // The header is small and gains notes (events, checks) after export: always revalidated.
  const header: Header = await (await fetch(`${base}${name}.json`, { cache: 'no-cache' })).json()
  // The points: one binary file from the engine, or base64 text parts in a static shared copy
  // (benchmark/web_copy.py; artifact hosting serves text, not raw binary).
  let buf: ArrayBuffer
  if (header.bin_parts) {
    const bytes = new Uint8Array(header.bin_bytes ?? 0)
    let at = 0, carry = ''
    for (const p of header.bin_parts) {
      const text = carry + (await (await fetch(`${base}${p}`)).text())
      const whole = text.length - (text.length % 4)
      const bin = atob(text.slice(0, whole))
      carry = text.slice(whole)
      for (let i = 0; i < bin.length; i++) bytes[at++] = bin.charCodeAt(i)
    }
    buf = bytes.buffer
  } else buf = await (await fetch(`${base}${name}.bin`)).arrayBuffer()
  const n = header.points
  const q = new Int16Array(buf, 0, n * 3)
  const colours = new Uint8Array(buf, n * 6, n * 3)
  const positions = new Float32Array(n * 3)
  const [ox, oy, oz] = header.origin, [sx, sy, sz] = header.step
  for (let i = 0; i < n; i++) {
    positions[3 * i] = ox + q[3 * i] * sx
    positions[3 * i + 1] = oy + q[3 * i + 1] * sy
    positions[3 * i + 2] = oz + q[3 * i + 2] * sz
  }
  return { header, positions, colours }
}

/** One contiguous range of the point buffer as its own small-point cloud (no copies: subarray views). */
function Cloud({ data, group, size, opacity = 1 }: { data: Loaded; group: Group; size: number; opacity?: number }) {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(data.positions.subarray(group.start * 3, (group.start + group.count) * 3), 3))
    g.setAttribute('color', new THREE.BufferAttribute(data.colours.subarray(group.start * 3, (group.start + group.count) * 3), 3, true))
    g.computeBoundingSphere()
    return g
  }, [data, group])
  useEffect(() => () => geom.dispose(), [geom])
  return (
    <points geometry={geom} frustumCulled={false}>
      <pointsMaterial size={size} sizeAttenuation={false} vertexColors transparent={opacity < 1} opacity={opacity} depthWrite={opacity >= 1} />
    </points>
  )
}

function SizeBox({ box, label }: { box: Box; label: string }) {
  const [L, H, W] = box.size
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(L, H, W)), [L, H, W])
  useEffect(() => () => edges.dispose(), [edges])
  return (
    <group position={box.centre} rotation={[0, -box.yaw, 0]}>
      <lineSegments geometry={edges}><lineBasicMaterial color="#38e1ff" /></lineSegments>
      <Html position={[0, H / 2 + 0.35, 0]} center className="pw-label">
        <b>{label}</b><br />{L.toFixed(2)} × {W.toFixed(2)} × {H.toFixed(2)} m
      </Html>
    </group>
  )
}

/** The whole body as a rigid box, placed from the face the camera sees: its hidden parts included. */
function RigidBox({ r, hot }: { r: Rigid; hot: boolean }) {
  const [L, H, W] = r.size
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(L, H, W)), [L, H, W])
  useEffect(() => () => edges.dispose(), [edges])
  return (
    <group position={r.centre} rotation={[0, -r.yaw, 0]}>
      <lineSegments geometry={edges}><lineBasicMaterial color={hot ? '#ff4d4d' : '#ffb347'} /></lineSegments>
      <mesh position={[L / 2, 0, 0]}>
        <boxGeometry args={[0.04, H, W]} />
        <meshBasicMaterial color={hot ? '#ff4d4d' : '#ffb347'} transparent opacity={0.35} />
      </mesh>
    </group>
  )
}

function StructureLine({ s, hot }: { s: Structure; hot: boolean }) {
  const mid: [number, number, number] = [(s.from[0] + s.to[0]) / 2, (s.from[1] + s.to[1]) / 2 + 0.3, (s.from[2] + s.to[2]) / 2]
  return (
    <>
      <Line points={[s.from, s.to]} color={hot ? '#ff4d4d' : '#ffe14d'} lineWidth={hot ? 6 : 3} />
      <Html position={mid} center className={`pw-label ${hot ? 'pw-hot' : ''}`}>{hot ? 'IMPACT · ' : ''}{s.label}</Html>
    </>
  )
}

/** A camera that filmed this world (several when cameras are merged into one world), as a small pyramid. */
type Cam = { name: string; position: [number, number, number]; forward: [number, number, number]; fov_deg?: number }
function CameraMarker({ c }: { c: Cam }) {
  const p = new THREE.Vector3(...c.position), f = new THREE.Vector3(...c.forward).normalize()
  const up = new THREE.Vector3(0, 1, 0), r = new THREE.Vector3().crossVectors(f, up).normalize(), u = new THREE.Vector3().crossVectors(r, f)
  const tip = (sx: number, sy: number) => p.clone().addScaledVector(f, 0.8).addScaledVector(r, sx * 0.45).addScaledVector(u, sy * 0.28).toArray() as [number, number, number]
  const P = p.toArray() as [number, number, number]
  const a = tip(-1, -1), b = tip(1, -1), cc = tip(1, 1), d = tip(-1, 1)
  return (
    <>
      <Line points={[P, a, b, P, cc, d, P, a, d, a, b, cc]} color="#9b8cff" lineWidth={1.5} />
      <Html position={[P[0], P[1] + 0.4, P[2]]} center className="pw-label">{c.name}</Html>
    </>
  )
}

type View = 'camera' | 'top' | 'side' | 'event'

/** The middle of the object's path: what orbiting turns around. */
function pathCentre(h: Header): THREE.Vector3 {
  if (!h.boxes.length) {
    const [px, py, pz] = h.camera.position, [fx, fy, fz] = h.camera.forward
    return new THREE.Vector3(px + fx * 8, py + fy * 8, pz + fz * 8)
  }
  const c = new THREE.Vector3()
  for (const b of h.boxes) c.add(new THREE.Vector3(...b.centre))
  return c.divideScalar(h.boxes.length)
}

/** Where the key event happened: the structure it hit, else the whole body at that moment. */
function eventFocus(h: Header): { target: THREE.Vector3; along: THREE.Vector3; across: THREE.Vector3 } | null {
  const n = h.notes as Record<string, any>
  const ev = ((n.events ?? []) as WorldEvent[]).find((e) => e.kind === 'contact' || e.kind === 'impact')
  if (!ev) return null
  const rig = ((n.rigid ?? []) as Rigid[]).reduce<Rigid | null>((b, r) => (!b || Math.abs(r.t - ev.t) < Math.abs(b.t - ev.t) ? r : b), null)
  const s = ((n.structures ?? []) as Structure[])[0]
  let target: THREE.Vector3
  if (s) target = new THREE.Vector3(...s.from).add(new THREE.Vector3(...s.to)).multiplyScalar(0.5)
  else if (rig) target = new THREE.Vector3(...rig.centre)
  else return null
  // The direction of travel (the body's yaw) and across it.
  const yaw = rig ? rig.yaw : 0
  const along = new THREE.Vector3(Math.cos(yaw), 0, Math.sin(yaw))
  const across = new THREE.Vector3(-Math.sin(yaw), 0, Math.cos(yaw))
  return { target, along, across }
}

/** Camera view: where the real camera stood, looking the way it looked. Top and side: around the object's path.
 *  Event: a close-up across the place the key event happened. */
function ViewSetter({ header, controls, view }: { header: Header; controls: React.RefObject<any>; view: View }) {
  const { camera } = useThree()
  useEffect(() => {
    const focus = view === 'event' ? eventFocus(header) : null
    const target = focus ? focus.target.clone() : pathCentre(header)
    const [px, py, pz] = header.camera.position
    const cam = camera as THREE.PerspectiveCamera
    cam.fov = view === 'camera' ? header.camera.fov_deg : 50
    if (view === 'camera') cam.position.set(px, py, pz)
    else if (focus) {
      // A chase view: behind the body and above it, a little to the real camera's side, so the gap between its
      // front and the structure shows over the roof.
      const across = focus.across.clone()
      if (across.dot(new THREE.Vector3(px - target.x, 0, pz - target.z)) < 0) across.negate()
      cam.position.copy(target).addScaledVector(focus.along, -11).addScaledVector(across, 5).add(new THREE.Vector3(0, 3.5, 0))
      target.y -= 1.0
    } else {
      // Distance that fits the whole path plus some of the world around it.
      const ends = header.boxes.length ? [header.boxes[0].centre, header.boxes[header.boxes.length - 1].centre] : [[0, 0, 0], [0, 0, 0]]
      const span = Math.max(new THREE.Vector3(...ends[0]).distanceTo(new THREE.Vector3(...ends[1])), 6)
      const d = span * 1.4
      if (view === 'top') cam.position.set(target.x, target.y + d, target.z + 0.01)
      else {
        // Side: at right angles to the direction of travel, on the camera's side.
        const dir = new THREE.Vector3(...ends[1]).sub(new THREE.Vector3(...ends[0])).setY(0).normalize()
        const side = new THREE.Vector3(-dir.z, 0, dir.x)
        if (side.dot(new THREE.Vector3(px - target.x, 0, pz - target.z)) < 0) side.negate()
        cam.position.copy(target).addScaledVector(side, d).add(new THREE.Vector3(0, d * 0.25, 0))
      }
    }
    cam.updateProjectionMatrix()
    cam.lookAt(target)
    controls.current?.target.copy(target)
    controls.current?.update()
  }, [header, camera, controls, view])
  return null
}

const KIND_ICON: Record<string, string> = { cruise: '→', braking: '⏷', contact: '✸', impact: '✸', crush: '▣', stop: '■', rebound: '↩', peak_speed: '→' }

type Props = { base?: string; initial?: string; standalone?: boolean }

export default function PointWorld({ base = '/media/pointworld/', initial, standalone = false }: Props) {
  const [worlds, setWorlds] = useState<{ name: string; title: string; points: number }[]>([])
  const [name, setName] = useState<string | undefined>(initial)
  const [data, setData] = useState<Loaded>()
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string>()
  const [k, setK] = useState(0)
  const [trail, setTrail] = useState(true)
  const [rigid, setRigid] = useState(true)
  const [size, setSize] = useState(1.2)
  const [playing, setPlaying] = useState(false)
  const [view, setView] = useState<View>('camera')
  const [answer, setAnswer] = useState<string>()
  const [asking, setAsking] = useState(false)
  const [question, setQuestion] = useState('')
  const controls = useRef<any>(null)

  useEffect(() => {
    fetch(`${base}index.json`).then((r) => r.json()).then((w) => {
      setWorlds(w)
      if (!name && w.length) setName(w[0].name)
    }).catch(() => setWorlds([]))
  }, [base])
  useEffect(() => {
    if (!name) return
    setLoading(true); setErr(undefined); setData(undefined); setK(0); setPlaying(false); setView('camera'); setAnswer(undefined)
    if (standalone) history.replaceState(null, '', `#${name}`)
    load(base, name).then((d) => { setData(d); if (d.header.decimated_from) setSize(1.8) }).catch((e) => setErr(String(e))).finally(() => setLoading(false))
  }, [name, base, standalone])
  const objectGroups = data ? data.header.groups.filter((g) => g.tag !== 255) : []
  useEffect(() => {
    if (!playing || !objectGroups.length) return
    const id = setInterval(() => setK((x) => (x + 1) % objectGroups.length), 180)
    return () => clearInterval(id)
  }, [playing, objectGroups.length])

  const h = data?.header
  const box = h?.boxes[k]
  const notes = (h?.notes ?? {}) as Record<string, any>
  const events = (notes.events ?? []) as WorldEvent[]
  const rigids = (notes.rigid ?? []) as Rigid[]
  const structures = (notes.structures ?? []) as Structure[]
  const tNow = objectGroups[k]?.t
  const frameNow = objectGroups[k]?.frame
  const rigidNow = rigids.find((r) => r.frame === frameNow)
  const contactEv = events.find((e) => e.kind === 'contact' || e.kind === 'impact')
  const hot = !!(contactEv && tNow !== undefined && tNow >= (contactEv.window?.[0] ?? contactEv.t) - 0.02)
  const tFirst = objectGroups[0]?.t ?? 0, tLast = objectGroups[objectGroups.length - 1]?.t ?? 1
  const indexAt = (t: number) => {
    let best = 0
    objectGroups.forEach((g, i) => { if (Math.abs((g.t ?? 0) - t) < Math.abs((objectGroups[best].t ?? 0) - t)) best = i })
    return best
  }
  const reached = (e: WorldEvent) => tNow !== undefined && tNow + 1e-3 >= (e.window?.[0] ?? e.t)
  const lastEvent = [...events].reverse().find(reached)

  // The fitted speed curve (one physical fit over every frame) when there is one; else frame-to-frame speeds.
  const curve = notes.speed_curve as { t: number[]; kmh: number[]; err_kmh: number[] } | undefined
  let speed: string | undefined
  if (curve && tNow !== undefined && tNow >= curve.t[0] && tNow <= curve.t[curve.t.length - 1]) {
    const i = curve.t.reduce((best, x, j) => (Math.abs(x - tNow) < Math.abs(curve.t[best] - tNow) ? j : best), 0)
    speed = `${Math.max(curve.kmh[i], 0).toFixed(1)} ± ${curve.err_kmh[i].toFixed(1)}`
  } else if (notes.speeds_kmh && k > 0 && notes.speeds_kmh[k - 1] !== undefined) speed = `${notes.speeds_kmh[k - 1]}`

  async function ask(q: string) {
    if (!name) return
    setAsking(true); setAnswer(undefined)
    const fd = new FormData()
    fd.append('context', `world:${name}`)
    fd.append('question', q)
    try {
      const r = await fetch('/api/ask', { method: 'POST', body: fd })
      const j = await r.json()
      setAnswer(r.ok ? j.answer : `Couldn't ask: ${j.detail ?? r.status}`)
    } catch (e) {
      setAnswer(`Couldn't ask: ${e}`)
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className={`pw ${standalone ? 'pw-standalone' : ''}`}>
      {standalone && <div className="pw-brand"><b>PhysLens</b> · video → measured 3D world → physical events</div>}
      <div className="pw-bar">
        {worlds.map((w) => (
          <button key={w.name} className={w.name === name ? 'on' : ''} onClick={() => setName(w.name)}>{w.title}</button>
        ))}
        {!worlds.length && <span>No point worlds yet: run benchmark/export_pointworlds.py</span>}
      </div>
      <div className="pw-stage">
        {loading && <div className="pw-loading">Loading points…</div>}
        {err && <div className="pw-loading">Couldn't load this world: {err}</div>}
        {data && (
          <div className="pw-views">
            {(['camera', 'top', 'side', ...(contactEv ? ['event'] : [])] as View[]).map((v) => (
              <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ camera: 'Camera view', top: "Bird's-eye", side: 'Side', event: `${contactEv?.kind === 'impact' ? 'Impact' : 'Contact'} close-up` }[v]}</button>
            ))}
          </div>
        )}
        {lastEvent && <div className={`pw-event-now ${lastEvent === contactEv ? 'hot' : ''}`}>{KIND_ICON[lastEvent.kind] ?? '•'} {lastEvent.label}</div>}
        {data && h && (
          <Canvas camera={{ near: 0.05, far: 500 }} gl={{ antialias: false }} style={{ background: '#05070c' }}>
            <ViewSetter header={h} controls={controls} view={view} />
            <OrbitControls ref={controls} makeDefault enableDamping />
            <Cloud data={data} group={h.groups[0]} size={view === 'event' ? Math.max(size - 0.4, 0.5) : size} opacity={view === 'event' ? 0.35 : 1} />
            {trail && objectGroups.map((g, i) => i !== k && <Cloud key={g.tag} data={data} group={g} size={size} opacity={0.18} />)}
            {objectGroups[k] && <Cloud data={data} group={objectGroups[k]} size={size + 0.4} />}
            {box && !(rigid && rigidNow) && <SizeBox box={box} label={h.object} />}
            {rigid && rigidNow && <RigidBox r={rigidNow} hot={hot} />}
            {structures.map((s) => <StructureLine key={s.name} s={s} hot={hot} />)}
            {((notes.cameras ?? []) as Cam[]).map((c) => <CameraMarker key={c.name} c={c} />)}
            <gridHelper args={[60, 60, '#1d3b4f', '#0f1d28']} />
          </Canvas>
        )}
      </div>
      {data && h && (
        <div className="pw-panel">
          <div className="pw-controls">
            <button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
            <div className="pw-timeline">
              <input type="range" min={0} max={Math.max(objectGroups.length - 1, 0)} value={k} onChange={(e) => { setPlaying(false); setK(+e.target.value) }} />
              {events.map((e, i) => {
                const x = (e.t - tFirst) / Math.max(tLast - tFirst, 1e-6)
                if (x < -0.02 || x > 1.02) return null
                return <button key={i} className={`pw-tick ${e.kind}`} style={{ left: `${Math.min(Math.max(x, 0), 1) * 100}%` }} title={e.label}
                  onClick={() => { setPlaying(false); setK(indexAt(e.t)) }}>{KIND_ICON[e.kind] ?? '•'}</button>
              })}
            </div>
            <span>t = {tNow?.toFixed(2)} s (frame {frameNow})</span>
            <label><input type="checkbox" checked={trail} onChange={(e) => setTrail(e.target.checked)} /> trail</label>
            {rigids.length > 0 && <label><input type="checkbox" checked={rigid} onChange={(e) => setRigid(e.target.checked)} /> whole body</label>}
            <label>point size <input type="range" min={0.5} max={3} step={0.1} value={size} onChange={(e) => setSize(+e.target.value)} /></label>
          </div>
          {events.length > 0 && (
            <div className="pw-events">
              <div className="pw-events-title">What happened (found from the 3D points)</div>
              {events.map((e, i) => (
                <button key={i} className={`pw-ev ${reached(e) ? 'past' : ''} ${e === contactEv ? 'key' : ''}`}
                  onClick={() => { setPlaying(false); setK(indexAt(e.t)) }}>
                  <span className="pw-ev-t">{e.t.toFixed(2)} s</span>
                  <span className="pw-ev-l">{KIND_ICON[e.kind] ?? '•'} {e.label}</span>
                  {e.detail && <span className="pw-ev-d">{e.detail}</span>}
                  {e.truth && <span className="pw-ev-truth">{e.truth}</span>}
                </button>
              ))}
            </div>
          )}
          {(notes.explanation || !standalone) && (
            <div className="pw-explain">
              {notes.explanation && <div><b>Muse Spark, from these measurements:</b> {notes.explanation}</div>}
              {!standalone && (
                <form onSubmit={(e) => { e.preventDefault(); if (question.trim()) ask(question.trim()) }}>
                  <button type="button" disabled={asking} onClick={() => ask('What happened in this video, physically? Explain the event and why it happened.')}>Explain what happened</button>
                  <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about this world… (e.g. how fast was it going when it hit?)" />
                  <button disabled={asking || !question.trim()}>Ask</button>
                </form>
              )}
              {asking && <div className="pw-note">Asking Muse Spark…</div>}
              {answer && <div className="pw-answer">{answer}</div>}
            </div>
          )}
          <div className="pw-facts">
            <div><b>{h.points.toLocaleString()}</b> points{h.decimated_from ? ` (web copy of ${h.decimated_from.toLocaleString()})` : ''}: {h.static_points.toLocaleString()} world, {h.object_points.toLocaleString()} {h.object} ({objectGroups.length} moments)</div>
            {box && <div>{h.object} now (visible points): <b>{box.size[0].toFixed(2)} m</b> long × <b>{box.size[2].toFixed(2)} m</b> wide × <b>{box.size[1].toFixed(2)} m</b> high, from {box.points.toLocaleString()} points{speed !== undefined && <> · <b>{speed} km/h</b></>}</div>}
            {notes.rigid_body && <div>Whole body (rigid, from the frames where all of it is in view): <b>{notes.rigid_body.length.toFixed(2)} m</b> long; the orange box places it from the face the camera sees, hidden parts included</div>}
            {notes.fused_size_m
              ? <div>Size, all moments merged ({notes.fused_from}): <b>{notes.fused_size_m.length.toFixed(2)} × {notes.fused_size_m.width.toFixed(2)} × {notes.fused_size_m.height.toFixed(2)} m</b>
                  {notes.true_size_m && <> · published {notes.true_size_m.length} × {notes.true_size_m.width} × {notes.true_size_m.height} m</>}
                  {notes.beam_clearance_m && <> · bridge clearance measured {notes.beam_clearance_m.toFixed(2)} m (signed 3.56 m)</>}</div>
              : <div>Size over all moments (median): {h.object_size_median_m.length} × {h.object_size_median_m.width} × {h.object_size_median_m.height} m</div>}
            {notes.height_check && <div>Height check: measured {notes.height_check.measured_height_m.toFixed(2)} m vs clearance {notes.height_check.clearance_m.toFixed(2)} m · {notes.height_check.verdict === 'height reads low'
              ? <>it hit anyway, so it is at least {notes.height_check.true_height_at_least_m} m tall: <b className="pw-warn">our height reads ≥{Math.round(notes.height_check.measured_low_by_at_least_pct)}% low here</b> (roof seen edge-on)</>
              : notes.height_check.verdict}</div>}
            {notes.truth_table && (
              <table className="pw-truth">
                <thead><tr><th></th><th>3D world</th><th>Ground truth</th></tr></thead>
                <tbody>{(notes.truth_table as [string, string, string][]).map(([a, b, c]) => <tr key={a}><td>{a}</td><td><b>{b}</b></td><td>{c}</td></tr>)}</tbody>
              </table>
            )}
            <div>Scale: {h.scale_from}{notes.check ? ` · check: ${notes.check}` : ''}</div>
            {notes.speed_kmh_median && <div>Speed {notes.speed_kmh_median} km/h (range {notes.speed_kmh_range?.join('–')})</div>}
            <div className="pw-note">{h.density_note}. Camera {h.camera.moving ? 'moving (its motion removed)' : 'fixed'}. Drag to orbit, scroll to zoom.{notes.credit ? ` ${notes.credit}` : ''}</div>
          </div>
        </div>
      )}
    </div>
  )
}
