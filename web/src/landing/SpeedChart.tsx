import { jeepSensor, jeepVideo, truckSensor, truckVideo } from './data8555'

const W = 640, H = 340, L = 44, R = 16, T = 20, B = 40
const T0 = -80, T1 = 300, V1 = 80
const X = (ms: number) => L + ((ms - T0) / (T1 - T0)) * (W - L - R)
const Y = (kmh: number) => T + (1 - kmh / V1) * (H - T - B)

/** A polyline that breaks where the samples have a gap (frames where tracking had no answer). */
function path(pts: [number, number][], gap = 12) {
  return pts.map(([t, v], i) => `${i && t - pts[i - 1][0] <= gap ? 'L' : 'M'}${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join('')
}

/** NHTSA 8555: both vehicles' speed through the crash, from the overhead video and from the accelerometers. */
export default function SpeedChart() {
  return (
    <svg className="speed-chart" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label="Speed of the F-150 and the Jeep through the crash: PhysLens from video follows the accelerometers">
      {[0, 20, 40, 60, 80].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} className="grid" />
          <text x={L - 8} y={Y(v) + 4} textAnchor="end">{v}</text>
        </g>
      ))}
      {[-50, 0, 100, 200, 300].map((t) => (
        <text key={t} x={X(t)} y={H - B + 20} textAnchor="middle">{t}</text>
      ))}
      <text x={W - R} y={H - 4} textAnchor="end">ms from first contact</text>
      <text x={L - 8} y={T - 8} textAnchor="end">km/h</text>
      <line x1={X(0)} x2={X(0)} y1={T} y2={H - B} className="contact" />
      <text x={X(0) + 6} y={T + 12} className="contact-label">contact</text>

      <path d={path(truckSensor)} className="sensor truck" />
      <path d={path(jeepSensor)} className="sensor jeep" />
      {/* The first two samples are the measured approach speed, a straight line; tracking resumes at 40 ms. */}
      <path d={path(truckVideo.slice(0, 2), 100) + path(truckVideo.slice(2))} className="video truck" />
      <path d={path(jeepVideo)} className="video jeep" />
    </svg>
  )
}
