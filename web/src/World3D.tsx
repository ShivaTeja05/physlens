import { Canvas } from '@react-three/fiber'
import { Grid, Html, OrbitControls } from '@react-three/drei'
import RAPIER from '@dimforge/rapier3d-compat'
import { useEffect, useMemo, useState } from 'react'

// Mirrors backend/physlens/scene.py (format physlens-scene/1).
type Src<T> = { value: T; source: string }
type Atom = { t: number; p: [number, number, number]; v: [number, number, number]; length?: number; measured: string[]; assumed: string[] }
type SceneObject = {
  id: string; label: string; shape: 'sphere' | 'box'; size: Src<number[]>; restitution: Src<number>; friction: Src<number>
  mass_kg: Src<number | null>; post_impact_friction?: Src<number>
}
type Scene = {
  capture: string; world: { gravity: [number, number, number]; gravity_source: string; obstacles: { id: string; size: number[]; position: number[]; source: string }[] }
  objects: SceneObject[]; atoms: Record<string, Atom[]>; notes: string
  moments: { object: string; t: number; kind: string; ratio?: number }[]
}
type Track = [number, number, number, number][]          // [t, x, y, z]

let rapierReady: Promise<void> | null = null
const initRapier = () => (rapierReady ??= RAPIER.init())

/** Re-simulate one object with Rapier from its first atom, deterministically, for the whole clip. */
async function simulate(scene: Scene, obj: SceneObject, w: { g: number; e: number; speed: number; perContact: boolean }): Promise<Track> {
  await initRapier()
  const atoms = scene.atoms[obj.id]
  const a0 = atoms.find((a) => Math.hypot(...a.v) > 0.1) ?? atoms[0]
  const world = new RAPIER.World({ x: 0, y: -w.g, z: 0 })
  const ground = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(0, -0.05, 0))
  world.createCollider(RAPIER.ColliderDesc.cuboid(200, 0.05, 200).setRestitution(0).setFriction(obj.shape === 'box' ? 0.02 : 0.5), ground)
  for (const o of scene.world.obstacles) {
    const b = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(o.position[0], o.position[1], o.position[2]))
    world.createCollider(RAPIER.ColliderDesc.cuboid(o.size[0] / 2, o.size[1] / 2, o.size[2] / 2).setRestitution(0), b)
  }
  const desc = RAPIER.RigidBodyDesc.dynamic().setTranslation(a0.p[0], a0.p[1], a0.p[2])
    .setLinvel(a0.v[0] * w.speed, a0.v[1] * w.speed, a0.v[2] * w.speed).setCcdEnabled(true)
  if (obj.shape === 'box') desc.enabledRotations(false, false, false)        // a car on its wheels doesn't tumble here
  const body = world.createRigidBody(desc)
  const s = obj.size.value
  const col = obj.shape === 'sphere' ? RAPIER.ColliderDesc.ball(s[0]) : RAPIER.ColliderDesc.cuboid(s[0] / 2, s[1] / 2, s[2] / 2)
  // Bounciness is the object's own (the ground and barrier have none, and Max picks the object's).
  col.setRestitution(w.e).setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Max).setFriction(obj.shape === 'box' ? 0.02 : 0.5)
  const collider = world.createCollider(col, body)
  // A real ball's bounciness changes from bounce to bounce (it depends on impact speed). With the measured value
  // left alone, each bounce gets its own measured bounciness, in order; a what-if uses one value for all.
  const ratios = scene.moments.filter((m) => m.object === obj.id && m.kind === 'contact' && m.ratio && m.ratio < 1 && m.t > a0.t)
    .sort((x, y) => x.t - y.t).map((m) => m.ratio as number)
  let bounces = 0, falling = true, rebounding = false
  if (w.perContact && ratios.length) collider.setRestitution(ratios[0])
  const dt = 1 / 240
  world.timestep = dt
  const t1 = atoms[atoms.length - 1].t + 0.5
  const out: Track = []
  for (let t = a0.t; t <= t1; t += dt) {
    const p = body.translation()
    out.push([t, p.x, p.y, p.z])
    world.step()
    if (obj.shape === 'box' && obj.post_impact_friction && !rebounding && body.linvel().x * a0.v[0] < 0) {
      rebounding = true      // moving back off the barrier: from here the measured post-impact drag applies
      collider.setFriction(obj.post_impact_friction.value)
      collider.setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max)
    }
    const vy = body.linvel().y
    if (falling && vy > 0.05) {          // just bounced: the next contact gets the next measured bounciness
      bounces++
      if (w.perContact && ratios.length) collider.setRestitution(ratios[Math.min(bounces, ratios.length - 1)])
    }
    falling = vy < 0
  }
  world.free()
  return out
}

function at<T extends { t: number }>(rows: T[], t: number): [T, T, number] | null {
  if (!rows.length || t < rows[0].t) return null
  let i = 0
  while (i < rows.length - 2 && rows[i + 1].t <= t) i++
  const a = rows[i], b = rows[Math.min(i + 1, rows.length - 1)]
  return [a, b, b.t > a.t ? Math.min(Math.max((t - a.t) / (b.t - a.t), 0), 1) : 0]
}

function trackAt(track: Track, t: number): [number, number, number] | null {
  if (!track.length || t < track[0][0]) return null
  const i = Math.min(Math.max(Math.round((t - track[0][0]) * 240), 0), track.length - 1)
  return [track[i][1], track[i][2], track[i][3]]
}

function Body({ obj, pos, length, ghost, tint }: { obj: SceneObject; pos: [number, number, number]; length?: number; ghost?: boolean; tint?: string }) {
  const s = obj.size.value
  const color = tint ?? (ghost ? '#ff5ad1' : '#3ee6ff')
  const mat = <meshStandardMaterial color={color} transparent opacity={ghost ? 0.35 : 0.9} emissive={color} emissiveIntensity={ghost ? 0.4 : 0.15} />
  if (obj.shape === 'sphere') return <mesh position={pos}><sphereGeometry args={[s[0], 32, 16]} />{mat}</mesh>
  const L = length ?? s[0]
  return (
    <group position={pos}>
      <mesh><boxGeometry args={[L, s[1] * 0.55, s[2]]} />{mat}</mesh>
      <mesh position={[0.08 * L, s[1] * 0.45, 0]}><boxGeometry args={[L * 0.5, s[1] * 0.38, s[2] * 0.9]} />{mat}</mesh>
      {!ghost && length !== undefined && length < s[0] - 0.03 && (
        <mesh position={[-L / 2 - (s[0] - L) / 2, 0, 0]}><boxGeometry args={[s[0] - L, s[1] * 0.5, s[2] * 1.02]} />
          <meshStandardMaterial color="#ff5d73" transparent opacity={0.35} /></mesh>
      )}
    </group>
  )
}

export default function World3D({ runId, time }: { runId: string; time: number }) {
  const [scene, setScene] = useState<Scene>()
  const [err, setErr] = useState<string>()
  const [track, setTrack] = useState<Track>()
  const [g, setG] = useState<number>(9.81)
  const [e, setE] = useState<number>(0.7)
  const [speed, setSpeed] = useState(1)
  const [busy, setBusy] = useState(false)
  const [mj, setMj] = useState<{ track: Track; params: Record<string, number>; meaning: Record<string, unknown>; rms_m: number; rms_unfitted_m: number }>()
  const [mjBusy, setMjBusy] = useState(false)

  useEffect(() => {
    setScene(undefined); setTrack(undefined); setErr(undefined); setMj(undefined)
    fetch(`/api/records/${runId}/scene`).then(async (r) => {
      if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText)
      const sc: Scene = await r.json()
      setScene(sc)
      setG(Math.abs(sc.world.gravity[1]))
      setE(sc.objects[0]?.restitution.value ?? 0.7)
      setSpeed(1)
    }).catch((x) => setErr(String(x)))
  }, [runId])

  const obj = scene?.objects[0]
  const run = async () => {
    if (!scene || !obj) return
    setBusy(true)
    const measuredE = obj.restitution.value
    // Per-bounce bounciness is for bouncing balls; a car hitting a barrier has one measured restitution.
    try { setTrack(await simulate(scene, obj, { g, e, speed, perContact: obj.shape === 'sphere' && Math.abs(e - measuredE) < 0.005 })) } catch (x) { setErr(String(x)) } finally { setBusy(false) }
  }
  // Re-simulate straight away with the measured values, so the ghost shows where physics alone would go.
  useEffect(() => { if (scene && obj) run() }, [scene])   // eslint-disable-line react-hooks/exhaustive-deps

  const atoms = useMemo(() => (obj ? scene!.atoms[obj.id] : []), [obj, scene])
  const now = at(atoms, time)
  const pos: [number, number, number] | null = now
    ? [0, 1, 2].map((k) => now[0].p[k] + (now[1].p[k] - now[0].p[k]) * now[2]) as [number, number, number] : null
  const length = now && now[0].length !== undefined ? now[0].length + ((now[1].length ?? now[0].length) - now[0].length) * now[2] : undefined
  const ghost = track ? trackAt(track, time) : null
  const fitMujoco = async () => {
    setMjBusy(true)
    try {
      const r = await fetch(`/api/records/${runId}/mujoco`, { method: 'POST' })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? r.statusText)
      setMj(body)
    } catch (x) { setErr(String(x)) } finally { setMjBusy(false) }
  }
  // The fitted MuJoCo track is sampled every few ms; nearest sample is plenty for drawing.
  const mjAt = (t: number): [number, number, number] | null => {
    if (!mj || !mj.track.length || t < mj.track[0][0]) return null
    let i = 0
    while (i < mj.track.length - 1 && mj.track[i + 1][0] <= t) i++
    return [mj.track[i][1], mj.track[i][2], mj.track[i][3]]
  }
  const mjPos = mjAt(time)
  // How far the re-simulation drifts from the measured atoms (RMS over the clip).
  const drift = useMemo(() => {
    if (!track || !atoms.length) return null
    const d = atoms.filter((a) => a.t >= track[0][0]).map((a) => { const q = trackAt(track, a.t); return q ? Math.hypot(q[0] - a.p[0], q[1] - a.p[1]) : 0 })
    return d.length ? Math.sqrt(d.reduce((s, x) => s + x * x, 0) / d.length) : null
  }, [track, atoms])

  if (err) return <div className="world3d"><div className="sub err">3D world: {err.replace('Error: ', '')}</div></div>
  if (!scene || !obj) return <div className="world3d"><div className="sub">Loading the atomic scene…</div></div>
  const crash = obj.shape === 'box'
  const top = Math.max(...atoms.map((a) => a.p[1]), 0.5)
  const target: [number, number, number] = crash ? [3, 0.8, 0] : [atoms[0]?.p[0] ?? 0, top * 0.45, 0]
  const cam: [number, number, number] = crash ? [5, 3.5, 10] : [target[0] + 0.9, top * 0.55, top * 1.25]
  const path = atoms.map((a) => a.p)
  return (
    <div className="world3d">
      <div className="recon-head">
        <span className="label">3D world <span className="knowledge rec">{atoms.length} atoms · replay + Rapier re-simulation</span></span>
        <div className="w3-controls">
          <label>g <input type="range" min={0.5} max={25} step={0.05} value={g} onChange={(x) => setG(Number(x.target.value))} /><b>{g.toFixed(2)}</b></label>
          <label>bounce <input type="range" min={0} max={0.98} step={0.01} value={e} onChange={(x) => setE(Number(x.target.value))} /><b>{e.toFixed(2)}</b></label>
          <label>speed <input type="range" min={0.3} max={1.8} step={0.05} value={speed} onChange={(x) => setSpeed(Number(x.target.value))} /><b>×{speed.toFixed(2)}</b></label>
          <button className="chip active" disabled={busy} onClick={run}>Re-simulate</button>
          <button className="chip mj" disabled={mjBusy} onClick={fitMujoco}>{mjBusy ? 'Fitting MuJoCo…' : 'Fit MuJoCo'}</button>
        </div>
      </div>
      <div className="w3-canvas">
        <Canvas camera={{ position: cam, fov: 45 }} shadows>
          <color attach="background" args={['#05070b']} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 8, 5]} intensity={1.2} />
          <Grid args={[40, 40]} cellSize={0.25} sectionSize={1} cellColor="#16323a" sectionColor="#2b6f7d" fadeDistance={30} infiniteGrid />
          {scene.world.obstacles.map((o) => (
            <mesh key={o.id} position={o.position as [number, number, number]}>
              <boxGeometry args={o.size as [number, number, number]} />
              <meshStandardMaterial color="#8a94a6" transparent opacity={0.55} />
            </mesh>
          ))}
          {path.length > 1 && (
            <line>
              <bufferGeometry><bufferAttribute attach="attributes-position" args={[new Float32Array(path.flat()), 3]} /></bufferGeometry>
              <lineBasicMaterial color="#3ee6ff" transparent opacity={0.4} />
            </line>
          )}
          {pos && <Body obj={obj} pos={pos} length={length} />}
          {ghost && <Body obj={obj} pos={ghost} ghost />}
          {mjPos && <Body obj={obj} pos={mjPos} ghost tint="#ffb020" />}
          {pos && <Html position={[pos[0], pos[1] + (crash ? 1.6 : 0.25), pos[2]]} center><div className="w3-tag">measured</div></Html>}
          <OrbitControls target={target} />
        </Canvas>
      </div>
      <div className="recon-note">
        Solid = the measured atoms. Magenta = Rapier re-simulating from the first atom with g {g.toFixed(2)} m/s², bounciness {obj.shape === 'sphere' && Math.abs(e - obj.restitution.value) < 0.005 ? 'as measured at each bounce' : e.toFixed(2)}
        {speed !== 1 ? `, speed ×${speed.toFixed(2)}` : ''}{drift !== null ? ` · drifts ${(drift * 100).toFixed(1)} cm from reality (RMS)` : ''}.
        {crash && ' Rigid bodies can’t crush: the ghost bounces where the real car crumples.'}
        {mj && (
          <span className="mj-note">
            {' '}Orange = MuJoCo fitted to the atoms: {(mj.rms_unfitted_m * 100).toFixed(1)} → {(mj.rms_m * 100).toFixed(1)} cm RMS
            {crash ? ` · crumple zone: crush ${String(mj.meaning.crush_m)} m, restitution ${String(mj.meaning.restitution)}, impact ${String(mj.meaning.impact_speed_kmh)} km/h`
              : ` · bounciness ${mj.params.e0} − ${mj.params.e1}·v`}
            {' · '}<a href={`/api/records/${runId}/mujoco.xml`}>download MJCF model</a>
          </span>
        )}
        <span className="knowledge"> {scene.notes}</span>
      </div>
    </div>
  )
}
