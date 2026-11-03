# Memory vs eyes benchmark

Plain Muse Spark (`muse-spark-1.3`, video input) vs PhysLens (SAM 3.1 tracking + physics fits + Muse Spark explanation) on clips built so that answering from memory is wrong. Same scale info for both. Gravity counts as correct within 10%.

| Clip | Truth | Muse Spark: gravity | PhysLens: gravity | Muse Spark: trick spotted | PhysLens: trick spotted |
|---|---|---|---|---|---|
| real | g = 9.81 | 9.80 ✓ | 9.73 ± 0.21 ✓ | real-time ✓ | real-time ✓ |
| slow-mo 2× | g = 2.45, slow-motion | 0.30 ✗ | 2.48 ± 0.05 ✓ | slow-motion ✓ | slow-motion ✓ |
| sped up 2× | g = 39.24, sped-up | 9.80 ✗ | 39.50 ± 0.82 ✓ | real-time ✗ | sped-up ✓ |
| reversed | g = 9.81, reversed | 0.59 ✗ | 9.68 ± 0.23 ✓ | slow-motion ✗ | reversed ✓ |
| upside down | g = 9.81, upside down | 9.80 ✓ | 9.81 ± 0.19 ✓ | real-time ✓, upside down ✓ | real-time ✓, upside down ✓ |
| 12 fps | g = 9.81 | 9.80 ✓ | 10.33 ± 0.19 ✓ | real-time ✓ | real-time ✓ |
| Mars gravity | g = 3.71 | 0.60 ✗ | 3.70 ± 0.04 ✓ | — | — |
| bouncy twin | g = 9.81, e = 0.85 | 0.40 ✗ | 9.79 ± 0.10 ✓ | slow-motion ✗, e = 0.80 ✓ | real-time ✓, e = 0.85 ✓ |
| dead twin | g = 9.81, e = 0.55 | 9.80 ✓ | 9.79 ± 0.10 ✓ | real-time ✓, e = 0.30 ✗ | real-time ✓, e = 0.55 ✓ |

**Checks passed:** Muse Spark 19/28 · PhysLens 28/28
