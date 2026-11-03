"""Scientist agent: turns measured evidence into a short explanation.

The agent never produces numbers of its own. It receives the measurements, writes one or two sentences,
and the verifier rejects any sentence containing a number that isn't in the evidence. Without a Muse Spark
key it uses deterministic templates, so the pipeline always works offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import verifier

DROP_GLOSSARY = """- g, g_err: gravity measured in this video, m/s², with its uncertainty. memory_g: textbook gravity, m/s².
- gravity_gap_sigma: how many uncertainties g is from memory_g (above 3 means a real difference).
- slowmo_factor: how many times slower than reality the clip runs (1 means real time).
- restitution_pct: percent of its SPEED the ball keeps in a bounce (not energy).
- contact1_/contact2_restitution_pct and _impact_speed (m/s): speed kept at two earlier bounces and how
  fast the ball hit. speed_dependent = 1 means bounciness changed with impact speed, so PhysLens switched
  from a constant-bounciness law to a speed-dependent one.
- peak_pred_cm: next bounce height predicted before it happened; peak_actual_cm: what it reached.
- upside_down = 1: gravity points up in the video, so the picture is upside down.
- reversed = 1: the ball leaves bounces faster than it arrived (energy_gain_pct faster), impossible for a
  real ball, so the clip is playing backwards and no prediction was made."""

PROMPT = """You are the scientist voice of PhysLens, a tool that measures physics from video.
You receive measurements as JSON. Write at most two short, vivid sentences for a live audience explaining
what the measurements show.

What the fields mean:
{glossary}

Rules:
- Use ONLY numbers that appear in the evidence, rounded to at most 2 decimals, always with units.
- Never state a number from general knowledge unless it is in the evidence.
- Restitution is about speed. Never call it energy.
- If gravity differs from memory_g (gap above 3), say that EITHER the clip is slowed down / sped up OR the
  gravity isn't Earth's. One clip cannot tell these apart, so never claim just one.
- Lead with the most surprising finding.
Reply with JSON: {{"text": "..."}}"""

SYSTEM_PROMPT = PROMPT.format(glossary=DROP_GLOSSARY)


@dataclass
class Explanation:
    text: str
    verified: bool
    source: str          # "muse_spark" or "template"
    rejected: list[str]  # numbers a model draft used that weren't in the evidence


def template_explanation(ev: dict[str, float]) -> str:
    g, g_err = ev["g"], ev["g_err"]
    parts = []
    if ev.get("upside_down"):
        parts.append("Gravity points up in this video, so the picture is upside down.")
    if ev.get("slowmo_factor") is not None and ev["gravity_gap_sigma"] > 3 and abs(ev["slowmo_factor"] - 1) > 0.1:
        # One clip can't tell slow motion from weaker gravity: both stretch the fall the same way.
        if ev["slowmo_factor"] > 1:
            parts.append(f"Measured gravity is {g:.2f} ± {g_err:.2f} m/s², not {ev['memory_g']:.2f}: either this "
                         f"clip runs {ev['slowmo_factor']:.1f}× slower than reality, or it isn't Earth gravity.")
        else:
            parts.append(f"Measured gravity is {g:.2f} ± {g_err:.2f} m/s², stronger than {ev['memory_g']:.2f}: "
                         f"either this clip is sped up {1 / ev['slowmo_factor']:.1f}×, or it isn't Earth gravity.")
    elif ev["gravity_gap_sigma"] > 3:
        parts.append(f"Measured gravity is {g:.2f} ± {g_err:.2f} m/s², close to Earth's {ev['memory_g']:.2f} but "
                     f"outside its error bar, so treat this clip's numbers with caution.")
    else:
        parts.append(f"Measured gravity is {g:.2f} ± {g_err:.2f} m/s², matching Earth's {ev['memory_g']:.2f}.")
    if ev.get("reversed"):
        parts.append(f"But the ball leaves each bounce {ev['energy_gain_pct']:.0f}% faster than it arrived, which "
                     f"no real ball can do: this clip is playing backwards, so no prediction is made.")
    elif ev.get("speed_dependent") and "contact2_restitution_pct" in ev:
        parts.append(f"Bounciness isn't constant: it kept {ev['contact1_restitution_pct']:.0f}% of its speed at "
                     f"{ev['contact1_impact_speed']:.1f} m/s but {ev['contact2_restitution_pct']:.0f}% at "
                     f"{ev['contact2_impact_speed']:.1f} m/s, so PhysLens switched to a speed-dependent bounce law.")
        if "peak_actual_cm" in ev:
            parts.append(f"Next peak called at {ev['peak_pred_cm']:.1f} cm; it reached {ev['peak_actual_cm']:.1f} cm.")
    elif "restitution_pct" in ev and "peak_actual_cm" in ev:
        parts.append(f"Each bounce keeps {ev['restitution_pct']:.0f}% of its speed, so the next peak was called at "
                     f"{ev['peak_pred_cm']:.1f} cm and it reached {ev['peak_actual_cm']:.1f} cm.")
    return " ".join(parts)


class ScientistAgent:
    def __init__(self):
        self.client = None
        self.model = os.environ.get("MUSE_MODEL", "")
        key = os.environ.get("META_API_KEY")
        if key and self.model:
            from openai import OpenAI
            # The Meta Model API is OpenAI-compatible. Base URL and model id are configurable because the
            # hackathon details may differ from the public docs.
            self.client = OpenAI(api_key=key, base_url=os.environ.get("META_API_BASE", "https://api.meta.ai/v1"))

    @property
    def name(self) -> str:
        return f"muse-spark ({self.model})" if self.client else "template"

    def explain(self, evidence: dict[str, float], template=None, glossary: str | None = None) -> Explanation:
        """`template` and `glossary` let each experiment describe its own fields; the drop experiment's are the
        default."""
        fallback = (template or template_explanation)(evidence)
        system = PROMPT.format(glossary=glossary) if glossary else SYSTEM_PROMPT
        if self.client is None:
            ok, bad = verifier.check(fallback, evidence)
            return Explanation(fallback, ok, "template", bad)
        from .. import budget
        try:
            budget.check(0.05)
            reply = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps({k: round(v, 4) for k, v in evidence.items()})}],
                temperature=0.4,
                # A two-sentence explanation of numbers it is handed doesn't need long hidden reasoning:
                # "low" answers in about half the time.
                reasoning_effort=os.environ.get("MUSE_EFFORT", "low"),
            )
            budget.record_tokens(self.model, reply.usage, what="explain")
            raw = reply.choices[0].message.content or ""
            text = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])["text"].strip()
        except Exception:
            ok, bad = verifier.check(fallback, evidence)
            return Explanation(fallback, ok, "template", bad)
        ok, bad = verifier.check(text, evidence)
        if ok:
            return Explanation(text, True, "muse_spark", [])
        ok_fb, _ = verifier.check(fallback, evidence)
        return Explanation(fallback, ok_fb, "template", bad)
