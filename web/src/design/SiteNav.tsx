import { useEffect, useState } from 'react'
import { ArrowRight } from '@phosphor-icons/react'
import Logo from './Logo'

const LINKS = [
  { href: '#how', label: 'How it works' },
  { href: '#results', label: 'Results' },
  { href: '#faq', label: 'FAQ' },
  { href: 'sim.html', label: 'Simulator' },
]

/** The floating site bar for the landing and 404 pages; on phones the links open in a full height panel. `base` is
 *  where the site's pages are: './' beside them, '/' from the 404 page, which a host serves at any path. */
export default function SiteNav({ current, base = './' }: { current?: string; base?: string }) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : ''
    const close = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [open])

  return (
    <>
      <nav className="site-nav" aria-label="Main">
        <a className="site-brand" href={base} aria-current={current === 'home' ? 'page' : undefined}>
          <Logo /> PhysLens
        </a>
        <ul className="site-links">
          {LINKS.map((l) => (
            <li key={l.href}><a href={base + l.href} aria-current={current === l.label ? 'page' : undefined}>{l.label}</a></li>
          ))}
        </ul>
        <a className="btn primary sm site-cta" href={base + 'app.html'}>Open the lab <ArrowRight weight="bold" aria-hidden /></a>
        <button className={`menu-toggle ${open ? 'open' : ''}`} aria-expanded={open} aria-controls="site-menu"
          aria-label={open ? 'Close menu' : 'Open menu'} onClick={() => setOpen(!open)}>
          <span /><span />
        </button>
      </nav>
      {/* Closed, the panel is visibility: hidden (so its links can't be focused) rather than unmounted, so it can animate. */}
      <div id="site-menu" className={`site-menu ${open ? 'open' : ''}`} aria-hidden={!open}>
        <ul>
          {[...LINKS, { href: 'app.html', label: 'Open the lab' }].map((l, i) => (
            <li key={l.href} style={{ transitionDelay: `${Math.min(i * 60, 400)}ms` }}>
              <a href={base + l.href} onClick={() => setOpen(false)}>{l.label}</a>
            </li>
          ))}
        </ul>
      </div>
    </>
  )
}
