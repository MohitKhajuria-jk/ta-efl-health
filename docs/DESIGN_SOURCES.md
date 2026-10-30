# Design references and asset provenance

Verified during implementation on 29 October 2026. The interface is an original design for a local healthcare research workstation. Decorative art never represents a live experiment, hospital deployment, or measured result.

## Motion Primitives

The requested [Motion Primitives documentation](https://motion-primitives.com/docs/animated-group) returned HTTP 403 through the research browser. The current official repository and raw sources were accessible and inspected instead:

- [AnimatedGroup source](https://github.com/ibelick/motion-primitives/blob/main/components/core/animated-group.tsx): copied into `web/src/components/motion-primitives/animated-group.tsx`. Supports `children`, `className`, `variants.container`, `variants.item`, `preset`, `as`, and `asChild`. It imports `motion` from `motion/react` and uses parent/child variants for staggered entrances.
- [TransitionPanel source](https://github.com/ibelick/motion-primitives/blob/main/components/core/transition-panel.tsx): reviewed as a transition reference; supports a child array, `activeIndex`, `transition`, and enter/center/exit variants. Reviewing it does not imply it is installed or used.
- [MIT license](https://github.com/ibelick/motion-primitives/blob/main/LICENCE.md), copyright 2024 ibelick, retained alongside the vendored component. The repository spells the filename `LICENCE.md`.

The component is vendored source, not a runtime import from an invented `motion-primitives` package. The local copy removes the Next-specific client directive, formats the source, and makes React JSX/type imports explicit for current TypeScript/React. Its API and animation behavior are otherwise retained. The application's `Enter` wrapper applies reduced-motion handling and a small translation with full text opacity; the upstream source does not itself disable all effects under reduced motion.

## Haikei and locally authored assets

The [Haikei site](https://haikei.app/) and [generator app](https://app.haikei.app/) were accessed. A headless-browser session reached the actual Layered Waves generator and inspected its direction, background, wave colors, shape controls, and SVG download button. A subsequent customized export could not complete: connection timeouts and unavailable generator controls persisted across retries. No direct Haikei export is claimed.

All delivered vector files below were authored specifically for this project, using the requested fallback permission:

| Asset | Origin and intended placement |
|---|---|
| `web/public/assets/research-orbit.svg` | Original concentric geometric research motif, transparent background, 1000 × 900. Designed for a navy hero/art panel; static and decorative. |
| `web/public/assets/layered-waves.svg` | Original layered-wave fallback informed by Haikei's generator concept, 1440 × 460. Restrained section-edge decoration; not placed behind analytical charts. |
| `web/public/assets/empty-experiment.svg` | Original record-and-chart illustration, 240 × 180. Used for a missing selection or empty state. |
| `web/public/assets/workspace-preview.png` | Actual browser capture of the working dashboard using a disposable copy of the 576 original experiment/model records. Labelled as a historical product preview, not live metrics or a clinical claim. No real account credentials were used. |

The SVGs contain no scripts, remote resources, text metrics, embedded patient data, or perpetual animation. They include internal title/description elements. When embedded purely decoratively, use empty `alt` text and `aria-hidden` as appropriate. The interactive simulation network is implemented separately from these assets.

## Realtime Colors and central palette

The [Realtime Colors homepage](https://www.realtimecolors.com/) was accessed, including its five-role color setup, contrast-checking description, and export options. The palette below was defined locally using those roles; it is not falsely described as a downloaded Realtime Colors export. A [palette preview link](https://www.realtimecolors.com/?colors=0b1725-f5f7f8-0f766e-dceeea-7ce8d3&fonts=Inter-Inter) records the chosen values for manual review.

| Role | Value | Usage |
|---|---|---|
| Text / deep navy | `#0b1725` | Main text, dark navigation, hero background |
| Background / paper | `#f5f7f8` | Main light canvas and light text on strong fills |
| Primary / deep teal | `#0f766e` | Primary actions, readable emphasis on paper |
| Secondary / pale teal | `#dceeea` | Quiet selected surfaces and supporting fills |
| Accent / cyan mint | `#7ce8d3` | Dark-surface accents and decorative strokes |

The frontend's central CSS tokens extend this base with surface, border, semantic, and chart roles. Avoid introducing arbitrary per-page colors. Use icon/text labels as well as color for status, and distinguish chart series with labels, marks, or stroke patterns.

Contrast was calculated locally using the WCAG sRGB relative-luminance formula:

| Foreground / background | Ratio |
|---|---:|
| Navy / paper | 16.80:1 |
| Paper / teal | 5.09:1 |
| Teal / paper | 5.09:1 |
| Cyan mint / navy | 12.33:1 |
| Cyan mint / paper | 1.36:1 — unsuitable for text or essential chart marks |

These values verify the base color pairs, not every rendered combination. Page-level accessibility and responsive checks are documented with the implementation's verification results.

## Light and dark appearance

`src/theme.tsx` manages Light, Dark and System preferences. The header and login controls switch themes immediately; Settings → Appearance exposes all three options. The browser stores the choice as `ta-efl-theme`, synchronizes it across tabs, and follows OS changes only in System mode. Storage failures fall back gracefully to an in-memory choice. A small external `public/theme-init.js` applies the saved appearance before React paints, retaining the existing Content Security Policy.

Both palettes live in `src/styles.css`. Dark mode uses a `#0b141f` canvas, `#13212e` surfaces, `#e6edf3` text, `#a4b7c7` supporting text and a `#7ce8d3` teal accent. Semantic and chart colors have dedicated dark values. Separate `--on-brand` and `--on-dark` tokens keep button labels and permanent navy brand panels readable when surfaces change. Native inputs, tables, dialogs, network diagrams and chart tooltips use the same theme tokens. The saved product screenshot remains a labelled historical light-mode preview.

Verified on 29 October 2026: TypeScript/Vite production build succeeded; the complete browser suite passed **14 tests, 0 failed, 0 skipped** in 134.97 seconds. Theme checks cover OS preference changes, explicit overrides, reload/navigation persistence, cross-tab synchronization, unavailable storage, settings selection, mobile navigation and axe accessibility checks across the research pages. Dark text/surface contrast is 13.83:1, supporting text 7.92:1, and primary-button labels 12.33:1. Browser evidence is stored in ignored `web/test-results/`.

## Godly / Recent art direction

The requested [Godly URL](https://godly.website/) redirected to [Recent](https://recent.design/). The current [website gallery](https://recent.design/websites) was accessed; it exposes SaaS, product, technology, health, and editorial categories. It serves as a discovery and art-direction reference, not an asset source. No individual product's branding, layout, logo, customer claims, or screenshots were copied.

The resulting direction combines editorial hero typography, generous whitespace, restrained teal highlights, a dark research identity, compact navigation, and grouped analytical panels. Live measurements use quieter light surfaces. The main visual motif is an abstract orbit of collaborating research components, distinct from the factual in-process network diagram.

## Motion and honesty rules

- Use brief entrance and navigation transitions rather than continuous decorative movement.
- Honor `prefers-reduced-motion` for both Motion and CSS effects.
- Animate network connections only in response to authoritative engine events.
- Show polling/disconnection/staleness explicitly; elapsed frontend time is never progress.
- Label historical metrics, newly executed runs, validation metrics, and held-out test results distinctly.
