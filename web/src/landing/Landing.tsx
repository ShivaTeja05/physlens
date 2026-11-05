import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ArrowRight, Camera, CarProfile, Check, Crosshair, Cube, Eye, LockSimple, Ruler, Warning } from '@phosphor-icons/react'
import SiteNav from '../design/SiteNav'
import Logo from '../design/Logo'
import SpeedChart from './SpeedChart'
import { result } from './data8555'

// Every number on this page comes from benchmark/EVENTS.md and the benchmark/*.json results; keep them in step.

const PROOF = [
  { test: '9070', what: 'Lexus NX into a rigid barrier', measured: 56.81, truth: 56.55 },
  { test: '15756', what: 'Nissan Kicks sideways into a pole', measured: 31.29, truth: 31.09 },
  { test: '8555', what: 'Ford F-150 into a standing Jeep', measured: 69.09, truth: 69.1 },
  { test: '10072', what: 'Moving barrier into a Nissan Altima', measured: 90.72, truth: 90.33 },
]
const pct = (m: number, t: number) => (Math.abs(m - t) / t) * 100

const BENEFITS: { icon: ReactNode; title: string; body: string; proof: string }[] = [
  { icon: <CarProfile weight="regular" />, title: 'Crash speed from one camera',
    body: 'Impact speed, the moment of contact, how far the front crushed and how each car moved after, read from a fixed camera.',
    proof: 'Approach speed within 0.7% of the speed trap on four NHTSA tests' },
  { icon: <LockSimple weight="regular" />, title: 'Predictions locked before they happen',
    body: 'The lab fingerprints each forecast before the frames that settle it, then marks it hit or miss against what the video shows.',
    proof: 'Classroom drop: g measured at 9.73 ± 0.21 m/s², next bounce called within 1 mm' },
  { icon: <Eye weight="regular" />, title: 'Catches slowed or generated clips',
    body: 'Gravity is the same in every honest video. A clip that was slowed, sped up or generated shows the wrong g.',
    proof: '28 of 28 trick clips caught, against 19 of 28 for Muse Spark alone' },
  { icon: <Ruler weight="regular" />, title: 'Clearance checked before contact',
    body: 'Measures a structure in the 3D world, checks the vehicle against it, and flags the strike before it happens.',
    proof: '11foot8 bridge: 3.63 m measured against a 3.556 m sign' },
  { icon: <Camera weight="regular" />, title: 'Measure what is in a photo',
    body: 'Scan names the scene, outlines its parts with SAM 3.1, and measures counts, lean and contact from one known size.',
    proof: 'Flags when the words disagree with the pixels, like “gibbous” against 46.5% lit' },
]

const STEPS = [
  { n: '01', title: 'Film or upload', body: 'A camera that holds still during the event, and one length you know: a car’s width, a wheelbase, a marked target.' },
  { n: '02', title: 'Measure', body: 'MoGe 3 depth finds the floor, SAM 3.1 outlines each object, and tracked image features give every body its slide and turn, frame by frame.' },
  { n: '03', title: 'Explain', body: 'Contact, crush and rebound are found from the points. Muse Spark explains them, and every number it uses is traced back to a measurement.' },
]

const RESULTS: { test: string; what: string; measured: string; truth: string; status: 'ok' | 'close' | 'open' }[] = [
  { test: '9070', what: 'Approach speed', measured: '56.81 km/h', truth: '56.55 km/h', status: 'ok' },
  { test: '15756', what: 'Approach speed', measured: '31.29 km/h', truth: '31.09 km/h', status: 'ok' },
  { test: '8555', what: 'F-150 approach speed', measured: `${result.approach} km/h`, truth: `${result.approachTrue} km/h`, status: 'ok' },
  { test: '8555', what: 'Jeep speed change', measured: `+${result.dvJeep} km/h`, truth: `+${result.dvJeepTrue} km/h`, status: 'ok' },
  { test: '8555', what: 'F-150 speed change', measured: `${result.dvTruck} km/h`, truth: `${result.dvTruckTrue} km/h`, status: 'close' },
  { test: '8555', what: 'Momentum balance', measured: '0.954', truth: '1.013', status: 'close' },
  { test: '10072', what: 'Approach speed', measured: '90.72 km/h', truth: '90.33 km/h', status: 'ok' },
  { test: '10072', what: 'Barrier speed change, first 40 ms', measured: 'within 1.6 km/h', truth: 'accelerometers', status: 'ok' },
  { test: '10072', what: 'Barrier speed change after 50 ms', measured: '−27 to −48 km/h', truth: '−26.5 km/h', status: 'open' },
  { test: '7371', what: 'Yaris into a Yaris, 113 km/h', measured: 'not reported', truth: 'lens and scale unknown', status: 'open' },
]
const STATUS = { ok: 'Matches', close: 'Close', open: 'Unresolved' }

const FAQ = [
  { q: 'Do I need a GPU or API keys?', a: 'Not for the lab. The physics runs offline on a laptop. A Meta Model API key adds SAM 3.1 outlines, Muse Spark explanations and Muse Voice. A RunPod serverless GPU is only needed to build new 3D worlds, and costs nothing while idle.' },
  { q: 'What footage works best?', a: 'A camera that does not move during the event, an object that fills a good part of the picture, and one length you know. Crash test cameras at 1,000 frames per second are ideal; a phone at 30 frames per second is fine for a drop or a roll.' },
  { q: 'What does not work yet?', a: 'Fast objects close to a 30 fps camera blur too much. Objects a few pixels across, like the Apollo 15 hammer and feather, are too small. A lens that cannot be calibrated from straight lines in the scene, like the overhead camera in NHTSA 7371, is left unreported rather than guessed.' },
  { q: 'How do you know the numbers are right?', a: 'Every benchmark uses footage that comes with official numbers: speed traps, accelerometers, published sizes. The method is fixed before looking at the answer, and the official numbers are used only in the final comparison. Misses are recorded next to the hits.' },
  { q: 'Where does my clip go?', a: 'To the engine running on your own machine. With keys set, frames are sent to api.meta.ai for outlines and explanations, and to your own RunPod endpoint for depth. Nothing is sent anywhere else, and keys never leave your .env file.', id: 'privacy' },
  { q: 'Can it recognise people?', a: 'No. SAM 3.1 can outline a person or a number plate, which is useful for blurring before sharing. Telling who someone is stays out of scope on purpose.' },
  { q: 'Where do the test videos come from?', a: 'NHTSA’s Vehicle Crash Test Database (public, credited, never edited or hosted here), Wikimedia Commons under CC BY with each author credited, and NASA’s public domain archive.', id: 'credits' },
  { q: 'What does the AI do, and what does the geometry do?', a: 'Measurements come from geometry: rays from the camera meeting the floor, and features tracked across frames. Muse Spark names things, gives published sizes and writes the explanation, and its numbers are checked against the measurements before you see them.' },
]

function useReveal<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { el.classList.add('in'); io.disconnect() } }, { threshold: 0.15 })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return ref
}

function Section({ id, className = '', children, label }: { id?: string; className?: string; children: ReactNode; label?: string }) {
  const ref = useReveal<HTMLElement>()
  return <section id={id} ref={ref} className={`section reveal ${className}`} aria-label={label}>{children}</section>
}

/** The tagline: each word turns from muted to full colour in reading order as the section scrolls past a line. */
function TaglineReveal({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLElement>(null)
  const words = lines.map((l) => l.split(' '))
  const total = words.flat().length
  const [lit, setLit] = useState(() => (matchMedia('(prefers-reduced-motion: reduce)').matches ? total : 0))
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let raf = 0
    const update = () => {
      raf = 0
      const el = ref.current
      if (!el) return
      const r = el.getBoundingClientRect(), vh = innerHeight
      // 0 when the block's top reaches 85% of the screen, 1 when its bottom reaches 45%.
      const p = (vh * 0.85 - r.top) / (r.height + vh * 0.4)
      setLit((prev) => Math.max(prev, Math.round(Math.min(Math.max(p, 0), 1) * total)))
    }
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(update) }
    addEventListener('scroll', onScroll, { passive: true })
    update()
    return () => { removeEventListener('scroll', onScroll); cancelAnimationFrame(raf) }
  }, [total])
  let i = 0
  return (
    <section ref={ref} className="tagline" aria-label={lines.join(' ')}>
      <p className="heading-2" aria-hidden>
        {words.map((ws, li) => (
          <span key={li} className="tagline-line">
            {ws.map((w) => { const k = i++; return <span key={k} className={`tw ${k < lit ? 'on' : ''}`}>{w} </span> })}
          </span>
        ))}
      </p>
    </section>
  )
}

export default function Landing() {
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <SiteNav current="home" />
      <main id="main">
        <header className="hero">
          <div className="hero-copy">
            <p className="overline hero-kicker"><span className="dot" /> Meta Global AI Developer Hackathon · Indie track</p>
            <h1 className="heading-1 hero-title"><span>Measure real physics</span> <span>from ordinary video</span></h1>
            <p className="body-3 hero-sub">
              PhysLens turns a camera clip into measured 3D points, finds the moment of impact,
              and checks every number it reports against the official record.
            </p>
            <div className="hero-actions">
              <a className="btn primary" href="./app.html">Open the lab <ArrowRight weight="bold" aria-hidden /></a>
              <span className="caption-2 muted">Runs on your laptop. No account needed.</span>
            </div>
          </div>

          <figure className="hero-visual">
            <div className="hv-head">
              <div>
                <div className="overline">NHTSA test 8555 · overhead camera, 1,000 fps</div>
                <div className="title-3">Ford F-150 into a standing Jeep Liberty</div>
              </div>
              <span className="tag"><Crosshair weight="bold" aria-hidden /> Measured from video</span>
            </div>
            <SpeedChart />
            <figcaption className="hv-legend caption-5">
              <span><i className="key video truck" /> F-150, PhysLens</span>
              <span><i className="key video jeep" /> Jeep, PhysLens</span>
              <span><i className="key sensor" /> NHTSA accelerometers</span>
            </figcaption>
            <p className="caption-5 hv-note">The F-150 line reads low from 80 to 100 ms after contact, so the Results table lists its speed change as close, not matched.</p>
            <div className="hv-readouts">
              <div><div className="caption-4 muted">F-150 approach</div><div className="readout num">{result.approach}<small> km/h</small></div><div className="caption-5 muted">speed trap {result.approachTrue}</div></div>
              <div><div className="caption-4 muted">Jeep pushed to</div><div className="readout num">+{result.dvJeep}<small> km/h</small></div><div className="caption-5 muted">accelerometers +{result.dvJeepTrue}</div></div>
            </div>
          </figure>
        </header>

        <Section className="proof" label="Approach speed on four crash tests">
          <p className="overline">Approach speed, PhysLens against the NHTSA speed trap</p>
          <ul className="proof-grid">
            {PROOF.map((p) => (
              <li key={p.test} className="proof-cell">
                <div className="num heading-5">{p.measured.toFixed(2)}<small className="caption-4 muted"> km/h</small></div>
                <div className="caption-4">{p.what}</div>
                <div className="caption-5 muted">Test {p.test} · trap {p.truth} · off by {pct(p.measured, p.truth).toFixed(1)}%</div>
              </li>
            ))}
          </ul>
        </Section>

        <Section className="problem" label="The problem">
          <div className="split">
            <h2 className="heading-4">A clip shows what happened.{' '}<br />It rarely says how fast.</h2>
            <div className="stack">
              <p className="body-3 muted">Dashcams, phones and test rigs record more events than anyone measures. The numbers are in the pixels, but reading them by hand takes a frame by frame session with a ruler.</p>
              <p className="body-3">PhysLens reads speed, timing and crush straight from the frames, states the assumptions it made, and tells you where it cannot be sure.</p>
            </div>
          </div>
        </Section>

        <TaglineReveal lines={['Every camera is already', 'a physics instrument.', 'PhysLens shows the reading,', 'and how sure it is.']} />

        <Section id="benefits" className="two-col" label="What it does">
          <div className="col-head">
            <p className="overline">What it does</p>
            <h2 className="heading-3">Five ways to read a video</h2>
            <p className="body-6 muted">Each one has been checked on footage that comes with official numbers. The proof sits under the claim.</p>
          </div>
          <ol className="benefits">
            {BENEFITS.map((b, i) => (
              <li key={b.title} className="benefit">
                <span className="num caption-4 benefit-n">{String(i + 1).padStart(2, '0')}</span>
                <div className="benefit-body">
                  <h3 className="title-2"><span className="icon" aria-hidden>{b.icon}</span>{b.title}</h3>
                  <p className="body-6 muted">{b.body}</p>
                  <p className="caption-4 benefit-proof"><Check weight="bold" aria-hidden /> {b.proof}</p>
                </div>
              </li>
            ))}
          </ol>
        </Section>

        <Section id="how" label="How it works">
          <p className="overline">How it works</p>
          <h2 className="heading-3 section-title">Video in, measured world out</h2>
          <ol className="steps">
            {STEPS.map((s) => (
              <li key={s.n} className="step">
                <span className="num caption-3 step-n">{s.n}</span>
                <h3 className="title-2">{s.title}</h3>
                <p className="body-6 muted">{s.body}</p>
              </li>
            ))}
          </ol>
          <a className="world-link" href="./sim.html">
            <Cube weight="regular" aria-hidden />
            <span><span className="title-4">Walk around a measured 3D world</span><br /><span className="caption-5 muted">The simulator: every point the camera saw, each object as one body moving through time</span></span>
            <ArrowRight weight="bold" aria-hidden />
          </a>
        </Section>

        <Section id="results" label="Results">
          <p className="overline">Results</p>
          <h2 className="heading-3 section-title">Every test, including the misses</h2>
          <p className="body-3 muted section-lede">The method is fixed before looking at the official numbers. When the footage cannot pin down the lens or the scale, the result is left unreported instead of tuned.</p>
          <div className="table-wrap">
            <table className="results">
              <thead><tr><th>Test</th><th>Quantity</th><th>PhysLens</th><th>Official</th><th>Status</th></tr></thead>
              <tbody>
                {RESULTS.map((r, i) => (
                  <tr key={i}>
                    <td className="num">{r.test}</td><td>{r.what}</td><td className="num">{r.measured}</td><td className="num muted">{r.truth}</td>
                    <td><span className={`status ${r.status}`}>{r.status === 'open' && <Warning weight="fill" aria-hidden />}{STATUS[r.status]}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="faq" label="Questions">
          <p className="overline">Questions</p>
          <h2 className="heading-3 section-title">Before you try it</h2>
          <dl className="faq">
            {FAQ.map((f) => (
              <div key={f.q} id={f.id} className="faq-row">
                <dt className="title-3">{f.q}</dt>
                <dd className="body-6 muted">{f.a}</dd>
              </div>
            ))}
          </dl>
        </Section>

        <Section className="final" label="Open the lab">
          <h2 className="heading-3">Point a camera at physics</h2>
          <p className="body-3 muted">Start with a sample clip, or bring your own drop, roll or crash.</p>
          <a className="btn primary" href="./app.html">Open the lab <ArrowRight weight="bold" aria-hidden /></a>
          <span className="caption-2 muted">Runs on your laptop. No account, and the physics works offline.</span>
        </Section>
      </main>

      <footer className="site-footer">
        <div className="footer-brand"><Logo /> <span className="title-4">PhysLens</span></div>
        <nav aria-label="Footer" className="footer-links caption-2">
          <a href="./app.html">Lab</a>
          <a href="./sim.html">Simulator</a>
          <a href="#privacy">Data and privacy</a>
          <a href="#credits">Credits</a>
          <a href="https://github.com/ShivaTeja05/physlens" target="_blank" rel="noreferrer">GitHub</a>
        </nav>
        <p className="caption-5 muted footer-note">Crash footage and sensor data: NHTSA Vehicle Crash Test Database. Built with SAM 3.1, Muse Spark, MoGe 3 and MapAnything.</p>
      </footer>
    </>
  )
}
