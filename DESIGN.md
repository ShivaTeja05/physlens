---
version: alpha
name: PhysLens
description: A measurement instrument for video. Warm paper pages, dark instrument stages for footage and 3D points, one muted crash-test orange for what PhysLens measured, Geist for words and Geist Mono for measured figures. Calm, exact, honest about misses.

colors:
  primary: "#1c1a17"
  on-primary: "#f6f4ef"
  bg: "#f6f4ef"
  surface: "#f6f4ef"
  panel: "#fdfcf9"
  raised: "#eeebe4"
  hover: "#e6e2d9"
  line: "#e3dfd6"
  line-strong: "#d3cec2"
  text: "#1c1a17"
  muted: "#6b665d"
  faint: "#9a9488"
  btn: "#1c1a17"
  btn-hover: "#36322b"
  btn-text: "#f6f4ef"
  accent: "#b9512a"
  accent-soft: "rgba(185, 81, 42, 0.09)"
  accent-line: "rgba(185, 81, 42, 0.32)"
  series-2: "#2f5d8a"
  sensor: "#4f7a5a"
  hit: "#3d7a4f"
  miss: "#b3261e"
  amber: "#9a6a12"
  whatif: "#8a4f9e"
  predicted: "#5b5fa8"
  stage: "#000000"
  stage-panel: "rgba(255, 255, 255, 0.06)"
  stage-line: "rgba(255, 255, 255, 0.08)"
  stage-text: "#f4f1ea"
  stage-muted: "#a8a196"
  stage-faint: "#6f695f"
  stage-btn: "#f2f4fa"
  stage-btn-text: "#000000"
  stage-accent: "#f08a5d"
  stage-sensor: "#9cc9a6"
  stage-hit: "#4ade80"
  stage-miss: "#ff5d73"
  stage-amber: "#ffc857"
  stage-whatif: "#ff5ad1"
  stage-predicted: "#b49bff"

typography:
  heading-1: { fontFamily: Geist, fontSize: 60px, lineHeight: 60px, fontWeight: 600, letterSpacing: -0.025em }
  heading-2: { fontFamily: Geist, fontSize: 48px, lineHeight: 48px, fontWeight: 600, letterSpacing: -0.025em }
  heading-3: { fontFamily: Geist, fontSize: 36px, lineHeight: 40px, fontWeight: 600, letterSpacing: -0.02em }
  heading-4: { fontFamily: Geist, fontSize: 30px, lineHeight: 36px, fontWeight: 600, letterSpacing: -0.02em }
  heading-5: { fontFamily: Geist, fontSize: 24px, lineHeight: 32px, fontWeight: 600, letterSpacing: -0.01em }
  heading-6: { fontFamily: Geist, fontSize: 20px, lineHeight: 28px, fontWeight: 600 }
  title-1: { fontFamily: Geist, fontSize: 20px, lineHeight: 28px, fontWeight: 500 }
  title-2: { fontFamily: Geist, fontSize: 18px, lineHeight: 28px, fontWeight: 500 }
  title-3: { fontFamily: Geist, fontSize: 16px, lineHeight: 24px, fontWeight: 500 }
  title-4: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 500 }
  title-5: { fontFamily: Geist, fontSize: 12px, lineHeight: 16px, fontWeight: 500 }
  body-1: { fontFamily: Geist, fontSize: 16px, lineHeight: 24px, fontWeight: 600 }
  body-2: { fontFamily: Geist, fontSize: 16px, lineHeight: 24px, fontWeight: 500 }
  body-3: { fontFamily: Geist, fontSize: 16px, lineHeight: 24px, fontWeight: 400 }
  body-4: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 600 }
  body-5: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 500 }
  body-6: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 400 }
  button-lg: { fontFamily: Geist, fontSize: 18px, lineHeight: 28px, fontWeight: 600 }
  button-md: { fontFamily: Geist, fontSize: 16px, lineHeight: 24px, fontWeight: 600 }
  button-sm: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 600 }
  button-xsm: { fontFamily: Geist, fontSize: 12px, lineHeight: 16px, fontWeight: 600 }
  caption-1: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 500 }
  caption-2: { fontFamily: Geist, fontSize: 14px, lineHeight: 20px, fontWeight: 400 }
  caption-3: { fontFamily: Geist, fontSize: 12px, lineHeight: 16px, fontWeight: 600 }
  caption-4: { fontFamily: Geist, fontSize: 12px, lineHeight: 16px, fontWeight: 500 }
  caption-5: { fontFamily: Geist, fontSize: 12px, lineHeight: 16px, fontWeight: 400 }
  figure: { fontFamily: Geist Mono, fontSize: 14px, lineHeight: 20px, fontWeight: 500, fontFeature: '"tnum"' }

rounded:
  sm: 2px
  DEFAULT: 4px
  md: 6px
  lg: 8px
  xl: 12px
  2xl: 16px
  3xl: 24px
  full: 9999px

spacing:
  s-25: 2px
  s-50: 4px
  s-75: 8px
  s-100: 12px
  s-200: 16px
  s-300: 24px
  s-400: 32px
  s-500: 40px
  s-600: 48px
  s-700: 64px
  s-800: 80px
  s-900: 96px

components:
  button-primary:
    backgroundColor: "{colors.btn}"
    textColor: "{colors.btn-text}"
    typography: "{typography.button-md}"
    rounded: "{rounded.lg}"
    padding: "{spacing.s-75} {spacing.s-100}"
  button-primary-hover:
    backgroundColor: "{colors.btn-hover}"
    textColor: "{colors.btn-text}"
  button-secondary:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.text}"
    typography: "{typography.button-md}"
    rounded: "{rounded.lg}"
    padding: "{spacing.s-75} {spacing.s-100}"
  button-secondary-hover:
    backgroundColor: "{colors.hover}"
    textColor: "{colors.text}"
  button-ghost:
    textColor: "{colors.muted}"
    typography: "{typography.button-md}"
    rounded: "{rounded.lg}"
  button-sm:
    typography: "{typography.button-sm}"
    padding: "{spacing.s-50} {spacing.s-100}"
  chip:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.text}"
    typography: "{typography.body-5}"
    rounded: "{rounded.lg}"
    padding: "{spacing.s-50} {spacing.s-100}"
  mode-tab-on:
    backgroundColor: "{colors.btn}"
    textColor: "{colors.btn-text}"
    typography: "{typography.button-sm}"
    rounded: "{rounded.lg}"
  card:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.text}"
    rounded: "{rounded.2xl}"
    padding: "{spacing.s-200}"
  readout:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.text}"
    typography: "{typography.figure}"
    rounded: "{rounded.xl}"
    padding: "{spacing.s-200}"
  stage:
    backgroundColor: "{colors.stage}"
    textColor: "{colors.stage-text}"
    rounded: "{rounded.2xl}"
  stage-caption:
    textColor: "{colors.stage-text}"
    typography: "{typography.body-5}"
    rounded: "{rounded.lg}"
    padding: "{spacing.s-75} {spacing.s-200}"
  stage-button:
    backgroundColor: "{colors.stage-btn}"
    textColor: "{colors.stage-btn-text}"
    typography: "{typography.button-sm}"
    rounded: "{rounded.lg}"
  page:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.text}"
    typography: "{typography.body-3}"
  measured-figure:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.accent}"
    typography: "{typography.heading-6}"
  verdict-hit:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.hit}"
    typography: "{typography.body-4}"
  verdict-miss:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.miss}"
    typography: "{typography.body-4}"
  verdict-close:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.amber}"
    typography: "{typography.body-4}"
  stage-measured:
    backgroundColor: "{colors.stage}"
    textColor: "{colors.stage-accent}"
    typography: "{typography.figure}"
  stage-note:
    backgroundColor: "{colors.stage}"
    textColor: "{colors.stage-muted}"
    typography: "{typography.caption-4}"
  skip-link:
    backgroundColor: "{colors.btn}"
    textColor: "{colors.btn-text}"
    typography: "{typography.button-sm}"
    rounded: "{rounded.lg}"
    padding: "{spacing.s-75} {spacing.s-100}"
---

# PhysLens design system

Read this before any change to `web/`. It is the source of truth for every teammate and every AI agent, and it
overrides installed design skills where they disagree (see "Rule precedence"). The CSS lives in
`web/src/design/tokens.css` under the same names (`colors.accent` is `--accent`, `typography.body-6` is `.body-6`,
`spacing.s-200` is `--s-200`, `rounded.lg` is `--r-lg`). Change both in the same commit.

## Overview

PhysLens turns ordinary video into measured 3D points, finds the physical events in them (contact, braking,
impact) and checks every number against an official record, such as an NHTSA crash test. It is read by crash
investigators, physics students and hackathon judges. The interface should feel like a good instrument: quiet
chrome, exact figures, the evidence large, and misses shown as plainly as hits.

Two worlds, one system:
- **Pages** (landing `index.html`, `404.html`): warm paper, read and scrolled. Few elements, large type, real
  footage and real charts.
- **Instrument screens** (`app.html` Lab, Scan and 3D points; `sim.html` simulator): worked in. Dense but calm
  panels around a dark stage that shows the footage or the point cloud.

The stages stay dark like an instrument display so video and point clouds keep their colours. Everything drawn on a
stage (captions, labels, stamps, HUD) uses the `stage-*` colours.

## Colors

The page is one warm grey family. Colour carries meaning, never decoration.

- **Ground and surfaces:** `bg` (page), `panel` (cards, a step lighter), `raised` (buttons, chips, readouts, a step
  darker), `hover`. Surfaces are told apart by these tone steps, not by lines or shadows.
- **Ink:** `text` for all running text and headings. `muted` for secondary notes (5.2:1 on `bg`). `faint` is for
  disabled states and chart axes only, never for text people must read (2.7:1).
- **Primary action:** `btn` charcoal with `btn-text`. One primary action per screen.
- **Accent `#b9512a`:** what PhysLens measured, and the current item. Not for chrome, not for decoration. As text it is
  4.47:1 on `bg` and 4.13:1 on `raised`, so use it as text at 18px and up, or as a line, dot or fill (see Pending decisions).
- **Data colours (fixed meanings, same in every chart and on every page):**
  - `accent`: PhysLens' measurement of the main object. `series-2` (blue): the second object.
  - `sensor`: the official record (accelerometers, speed trap). Dashed in charts.
  - `hit` / `miss`: verdicts. `amber`: close, not matched.
  - `whatif`: a what-if replay. `predicted`: a prediction before the frames that settle it.
- **On stages:** the `stage-*` values; `stage-accent` `#f08a5d` is the accent on black (7.3:1).

## Typography

Geist for every word, Geist Mono only for measured figures (speeds, times, gravity, frame numbers, record ids) with
tabular numerals. No italics, nothing above weight 600 in the interface (700 only if a display size needs it), no
uppercase tracked labels, sentence case everywhere, no hyphens inside sentences.

Every piece of text uses one named style (`heading-1` to `caption-5`, `figure`), all on Tailwind's type scale. Use
`text-wrap: balance` on headings and `pretty` on paragraphs. Headlines stay at two lines or fewer at every width,
including 390px: cut words first, then widen the measure, then size down. Running text on pages is 16px
(`body-3`); 14px is for instrument panels.

On the landing page, figures inside sentences use Geist with `font-variant-numeric: tabular-nums`; Geist Mono is
kept for instrument readouts and chart axes.

## Layout

- Spacing uses only `s-25` to `s-900`. Main buttons: 8px by 12px.
- Pages: one pair of equal outer margins at every width; hero heading and subheading at most 680px wide; sections
  clearly further apart than anything inside them; no section that leaves half the width empty without a reason.
- Instrument screens: stage on the left, panels on the right on desktop; on phones the stage comes first and the
  controls fold below it (long lists become a select).
- Check every page at 1440, 768 and 390px: no sideways scroll, buttons keep their text inside, the primary action
  visible on the first screen.

## Elevation & Depth

Depth comes from tone steps. Shadows only on what truly floats: the site nav and app bar, menus, popovers, dialogs.
Nothing at rest has a shadow (not buttons, not cards). Shadows are tinted toward the page (`rgba(66, 52, 28, …)`),
one light from above.

Glass (translucent with blur) only where something moves behind it: the floating site nav, the app bar, and labels
over video and 3D (dark glass). Nowhere else. A faint paper grain sits on the page ground; cards and stages cover it.

## Shapes

Tailwind radii only. Buttons, chips and inputs `lg` (8px); cards and stages `2xl` (16px); pills `full` only for status
dots and toggles. Nested shapes under a 32px gap use inner = outer − gap when the result is above 2px.

## Components

Every control has hover (tone step), active (`scale(0.98)`), visible focus (2px `accent` outline, 2px offset),
disabled (40% opacity), and, where it loads, a skeleton shaped like the result (no spinners). Errors are inline,
specific, say what to do next, use `role="alert"`, and never `window.alert()`.

- **Site nav:** brand, three to five links, one primary action. On phones a full-height menu; Escape closes it and
  focus stays inside while it is open.
- **Mode tabs (Lab, Scan, 3D points):** the current one is `mode-tab-on` with `aria-current="page"`.
- **Chips:** samples and options. A chip that starts a paid model call says so in its tooltip.
- **Readouts:** a label in `caption-4` over a value in `figure`, unit in `muted`.
- **Charts:** data colours above; dashed `sensor` for the official record; axes in `faint`; one finding stated in a
  sentence under the chart.
- **Stage captions and stamps:** dark glass, `stage-text`, never covering the object being measured.
- **404:** branded, says what happened, one way back home.

## Do's and Don'ts

Do
- Show the evidence large: real footage, the measured 3D world, the chart against the official record.
- State misses next to hits, with the size of the miss.
- Keep one primary action per screen and name it for what it does ("Open the lab").
- Use real numbers from records; never invent statistics, quotes or logos.
- Credit every outside picture or clip (NHTSA test number, source) in a caption or the footer.

Don't
- No gradients (not on backgrounds, buttons or text, including the hero heading).
- No eyebrow labels above headings on pages; no decorative 01 / 02 / 03 numbering.
- No hairline dividers or 1px outlined boxes on pages; on instrument screens a 1px border or a shadow, never both.
- No shadows on things at rest; no glass outside the places listed above.
- No icon-topped feature card rows; no emoji; no Inter, Roboto or Arial; icons only from Phosphor.
- No monospace for prose on pages.
- No browser-default easing; never animate layout properties.

## Motion

One curve: `cubic-bezier(0.22, 1, 0.36, 1)`; 500ms for movement, 300ms for colour and opacity, never under 200ms.
Transform and opacity only. Sections rise 16 to 32px and fade in once as they enter (IntersectionObserver), parts
staggered 60 to 90ms. The stage playback, the point-cloud camera and the chart drawing are the signature motion.
Under `prefers-reduced-motion` everything appears at once with no travel; no content may stay hidden.

## Rule precedence

When rules conflict: this file, then taste-skill (`design-taste-frontend`) house rules, then elayadesign
`landing-page-design`, then `redesign-existing-projects`. Where this file allows something those skills ban
(glass on the two bars, Geist Mono for figures, 1px borders on instrument panels, the warm palette), it is a
deliberate choice; pass the matching `--allow` to the taste inspector and do not "fix" it.

## Testing

Agents and tests never trigger paid Meta calls. The paid endpoints are `POST /api/runs`, `POST /api/scan`,
`POST /api/ask` and `POST /api/worlds/{name}/specs` (the specs agent); block them (402) in every browser session and
test. `GET /api/worlds/{name}/specs` only reads a cached lookup and is free. Use saved records (`GET /api/records`, open a
row in the Lab) instead of running samples, and compare `calls` in `GET /api/budget` before and after. For a
dedicated test engine, start it with `PHYSLENS_BUDGET_USD=0`.

## Pending decisions

- **Palette:** the cream ground with a terracotta accent is flagged by taste-skill as the "Claude palette". Options:
  keep it (pass `--allow palette-claude,palette-terracotta,palette-cream`), or try a lower-chroma stone ground
  (`#f4f3f0`) and a subject colour (crash-test target yellow and black) behind a screenshot comparison.
- **Accent contrast:** `#a84926` keeps the hue and passes AA as small text (5.24:1 on `bg`, 4.84:1 on `raised`).
- **Data violet and indigo** (`whatif`, `predicted`): flagged as "AI purple"; a teal and a slate would avoid it.
- **Fonts:** Geist loads from Google Fonts; self-host it so tests and offline demos never fall back.
