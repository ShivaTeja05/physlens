import { Canvas, useThree } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'

// Mirrors backend/physlens/pointworld.py (format physlens-pointworld/1).
type Group = { tag: number; start: number; count: number; frame?: number; t?: number }
type Box = { frame: number; t: number; points: number; centre: [number, number, number]; size: [number, number, number]; yaw: number }
type Header = {
  name: string; title: string; video: string; fps: number; points: number; static_points: number; object_points: number
  upsample: number; density_note: string; origin: [number, number, number]; step: [number, number, number]
  groups: Group[]; object: string; boxes: Box[]; object_size_median_m: { length: number; height: number; width: number }
  camera: { position: [number, number, number]; forward: [number, number, number]; fov_deg: number; moving: boolean }
  scale_from: string; notes: Record<string, unknown>
}
type Loaded = { header: Header; positions: Float32Array; colours: Uint8Array }

async function load(name: string): Promise<Loaded> {
  const header: Header = await (await fetch(`/media/pointworld/${name}.json`)).json()
  const buf = await (await fetch(`/media/pointworld/${name}.bin`)).arrayBuffer()
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

type View = 'camera' | 'top' | 'side'

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

/** Camera view: where the real camera stood, looking the way it looked. Top and side: around the object's path. */
function ViewSetter({ header, controls, view }: { header: Header; controls: React.RefObject<any>; view: View }) {
  const { camera } = useThree()
  useEffect(() => {
    const target = pathCentre(header)
    const [px, py, pz] = header.camera.position
    const cam = camera as THREE.PerspectiveCamera
    cam.fov = view === 'camera' ? header.camera.fov_deg : 50
    if (view === 'camera') cam.position.set(px, py, pz)
    else {
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

export default function PointWorld() {
  const [worlds, setWorlds] = useState<{ name: string; title: string; points: number }[]>([])
  const [name, setName] = useState<string>()
  const [data, setData] = useState<Loaded>()
  const [loading, setLoading] = useState(false)
  const [k, setK] = useState(0)
  const [trail, setTrail] = useState(true)
  const [size, setSize] = useState(1.2)
  const [playing, setPlaying] = useState(false)
  const [view, setView] = useState<View>('camera')
  const controls = useRef<any>(null)

  useEffect(() => {
    fetch('/media/pointworld/index.json').then((r) => r.json()).then((w) => { setWorlds(w); if (w.length) setName(w[0].name) }).catch(() => setWorlds([]))
  }, [])
  useEffect(() => {
    if (!name) return
    setLoading(true); setData(undefined); setK(0); setPlaying(false); setView('camera')
    load(name).then((d) => setData(d)).finally(() => setLoading(false))
  }, [name])
  const objectGroups = data ? data.header.groups.filter((g) => g.tag !== 255) : []
  useEffect(() => {
    if (!playing || !objectGroups.length) return
    const id = setInterval(() => setK((x) => (x + 1) % objectGroups.length), 180)
    return () => clearInterval(id)
  }, [playing, objectGroups.length])

  const h = data?.header
  const box = h?.boxes[k]
  const notes = (h?.notes ?? {}) as Record<string, any>
  // The fitted speed curve (one physical fit over every frame) when there is one; else frame-to-frame speeds.
  const curve = notes.speed_curve as { t: number[]; kmh: number[]; err_kmh: number[] } | undefined
  const tNow = objectGroups[k]?.t
  let speed: string | undefined
  if (curve && tNow !== undefined && tNow >= curve.t[0] && tNow <= curve.t[curve.t.length - 1]) {
    const i = curve.t.reduce((best, x, j) => (Math.abs(x - tNow) < Math.abs(curve.t[best] - tNow) ? j : best), 0)
    speed = `${Math.max(curve.kmh[i], 0).toFixed(1)} ± ${curve.err_kmh[i].toFixed(1)}`
  } else if (notes.speeds_kmh && k > 0) speed = `${notes.speeds_kmh[k - 1]}`

  return (
    <div className="pw">
      <div className="pw-bar">
        {worlds.map((w) => (
          <button key={w.name} className={w.name === name ? 'on' : ''} onClick={() => setName(w.name)}>{w.title}</button>
        ))}
        {!worlds.length && <span>No point worlds yet: run benchmark/export_pointworlds.py</span>}
      </div>
      <div className="pw-stage">
        {loading && <div className="pw-loading">Loading points…</div>}
        {data && (
          <div className="pw-views">
            {(['camera', 'top', 'side'] as View[]).map((v) => (
              <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ camera: 'Camera view', top: "Bird's-eye", side: 'Side' }[v]}</button>
            ))}
          </div>
        )}
        {data && h && (
          <Canvas camera={{ near: 0.05, far: 500 }} gl={{ antialias: false }} style={{ background: '#05070c' }}>
            <ViewSetter header={h} controls={controls} view={view} />
            <OrbitControls ref={controls} makeDefault enableDamping />
            <Cloud data={data} group={h.groups[0]} size={size} />
            {trail && objectGroups.map((g, i) => i !== k && <Cloud key={g.tag} data={data} group={g} size={size} opacity={0.18} />)}
            {objectGroups[k] && <Cloud data={data} group={objectGroups[k]} size={size + 0.4} />}
            {box && <SizeBox box={box} label={h.object} />}
            <gridHelper args={[60, 60, '#1d3b4f', '#0f1d28']} />
          </Canvas>
        )}
      </div>
      {data && h && (
        <div className="pw-panel">
          <div className="pw-controls">
            <button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
            <input type="range" min={0} max={Math.max(objectGroups.length - 1, 0)} value={k} onChange={(e) => { setPlaying(false); setK(+e.target.value) }} />
            <span>t = {objectGroups[k]?.t?.toFixed(2)} s (frame {objectGroups[k]?.frame})</span>
            <label><input type="checkbox" checked={trail} onChange={(e) => setTrail(e.target.checked)} /> trail</label>
            <label>point size <input type="range" min={0.5} max={3} step={0.1} value={size} onChange={(e) => setSize(+e.target.value)} /></label>
          </div>
          <div className="pw-facts">
            <div><b>{h.points.toLocaleString()}</b> points: {h.static_points.toLocaleString()} world, {h.object_points.toLocaleString()} {h.object} ({objectGroups.length} moments)</div>
            {box && <div>{h.object} now: <b>{box.size[0].toFixed(2)} m</b> long × <b>{box.size[2].toFixed(2)} m</b> wide × <b>{box.size[1].toFixed(2)} m</b> high, from {box.points.toLocaleString()} points{speed !== undefined && <> · <b>{speed} km/h</b></>}</div>}
            {notes.fused_size_m
              ? <div>Size, all moments merged ({notes.fused_from}): <b>{notes.fused_size_m.length.toFixed(2)} × {notes.fused_size_m.width.toFixed(2)} × {notes.fused_size_m.height.toFixed(2)} m</b>
                  {notes.true_size_m && <> · published {notes.true_size_m.length} × {notes.true_size_m.width} × {notes.true_size_m.height} m</>}
                  {notes.beam_clearance_m && <> · bridge clearance measured {notes.beam_clearance_m.toFixed(2)} m (signed 3.56 m)</>}</div>
              : <div>Size over all moments (median): {h.object_size_median_m.length} × {h.object_size_median_m.width} × {h.object_size_median_m.height} m</div>}
            <div>Scale: {h.scale_from}{notes.check ? ` · check: ${notes.check}` : ''}</div>
            {notes.speed_at_contact_kmh && <div>Cruising {notes.cruise_kmh} km/h, braking at {notes.braking_decel_ms2} m/s²; hits the beam at <b>{notes.speed_at_contact_kmh} ± {notes.speed_at_contact_err_kmh} km/h</b> (frames {notes.contact_frames?.join('–')}), then moves {notes.distance_after_contact_m} m more while the roof is peeled</div>}
            {notes.speed_kmh_median && <div>Speed {notes.speed_kmh_median} km/h (range {notes.speed_kmh_range?.join('–')})</div>}
            <div className="pw-note">{h.density_note}. Camera {h.camera.moving ? 'moving (its motion removed)' : 'fixed'}. Drag to orbit, scroll to zoom.</div>
          </div>
        </div>
      )}
    </div>
  )
}
