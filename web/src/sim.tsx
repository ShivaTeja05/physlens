import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './App.css'
import PointWorld from './PointWorld'

// The simulator on its own page: one measured 3D world per link (sim.html#truck-sharp). Under the dev server it
// reads the engine's worlds; built, it reads ./data/ beside it (benchmark/web_copy.py: a static copy for sharing,
// no engine needed). `?data=` overrides either.
const params = new URLSearchParams(location.search)
const base = params.get('data') ?? (import.meta.env.DEV ? '/media/pointworld/' : './data/')
const initial = location.hash.slice(1) || undefined
const standalone = true

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <PointWorld base={base} initial={initial} standalone={standalone} />
  </StrictMode>,
)
