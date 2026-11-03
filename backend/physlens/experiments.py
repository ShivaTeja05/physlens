"""Experiment cards: the fixed settings that make a stage experiment reliable."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    prompt: str          # what SAM is asked to follow ("red can")
    color: str | None    # colour hint for the offline tracker


@dataclass(frozen=True)
class ExperimentCard:
    id: str
    title: str
    hook: str
    prompt: str                 # what the tracker should follow
    color: str | None           # colour hint for the OpenCV fallback tracker
    reference_diameter_m: float  # known size of the tracked object, for pixel → metre scale
    lock_after_contacts: int    # lock predictions at this floor contact
    memory_g: float             # what a model "remembers" about gravity
    narration: dict[str, str]
    kind: str = "drop"          # which analysis runs: drop | projectile | race | springs
    objects: tuple[ObjectSpec, ...] = ()
    params: dict = field(default_factory=dict)  # defaults a presenter can override per run (tape measurements)


CARDS = {
    "drop": ExperimentCard(
        id="drop",
        title="Drop & bounce",
        hook="Measures gravity and bounciness, then calls the next bounce before it happens.",
        prompt="orange ball",
        color="orange",
        reference_diameter_m=0.065,
        lock_after_contacts=3,
        memory_g=9.81,
        narration={
            "intro": "One ball. Let's measure it instead of guessing.",
            "measuring": "Tracking every frame. Gravity so far: {g:.2f} metres per second squared.",
            "upside_down": "Wait. In this video the ball falls up. The picture is upside down, so I'm flipping it.",
            "reversed": "Something's wrong: this ball leaves every bounce {gain:.0f} percent faster than it arrived. "
                        "A real ball can't do that. This clip is playing backwards.",
            "bounce_law": "This ball loses more energy in hard bounces, so I'm switching to a speed-dependent bounce law.",
            "locked": "Prediction locked. The next bounce peaks at {peak_cm:.1f} centimetres.",
            "hit": "It reached {actual_cm:.1f}. Inside the prediction.",
            "miss": "It reached {actual_cm:.1f}. Outside the prediction, and we keep that on the scoreboard.",
        },
    ),
    "cup": ExperimentCard(
        id="cup",
        title="Catch the cup",
        hook="Watches a ball roll off a table and calls where it lands before it falls.",
        prompt="orange ball",
        color="orange",
        reference_diameter_m=0.04,
        lock_after_contacts=0,
        memory_g=9.81,
        kind="projectile",
        objects=(ObjectSpec("ball", "orange ball", "orange"),),
        params={"ball_diameter_m": 0.04, "table_height_m": 0.75},
        narration={
            "intro": "A ball, a table, and a cup we haven't placed yet.",
            "locked": "Locked. It lands {cm:.1f} centimetres past the edge. Put the cup there.",
            "hit": "It landed at {cm:.1f}. In the cup.",
            "miss": "It landed at {cm:.1f}. Outside the call, and that goes on the scoreboard.",
        },
    ),
    "cans": ExperimentCard(
        id="cans",
        title="Mystery Cans",
        hook="Two identical-looking cans race. The first part of the race reveals where each hides its weight.",
        prompt="red can",
        color="red",
        reference_diameter_m=0.066,
        lock_after_contacts=0,
        memory_g=9.81,
        kind="race",
        objects=(ObjectSpec("red", "red can", "red"), ObjectSpec("blue", "blue can", "blue")),
        params={"can_diameter_m": 0.066, "finish_distance_m": 1.0},
        narration={
            "intro": "Two cans. Same size, same weight. One of them is hiding something.",
            "locked": "Locked: the {winner} can wins by {margin:.2f} seconds.",
            "hit": "The {winner} can wins by {margin:.2f}. As called.",
            "miss": "The {winner} can wins by {margin:.2f}. Not what we called, and it goes on the scoreboard.",
        },
    ),
    "weigh": ExperimentCard(
        id="weigh",
        title="Weigh it by watching",
        hook="A known weight and a mystery weight bounce on identical springs; rhythm alone gives the mass.",
        prompt="green weight",
        color="green",
        reference_diameter_m=0.04,
        lock_after_contacts=0,
        memory_g=9.81,
        kind="springs",
        objects=(ObjectSpec("ref", "green weight", "green"), ObjectSpec("mystery", "red weight", "red")),
        params={"weight_diameter_m": 0.04, "reference_mass_g": 100.0, "lock_after_s": 2.0},
        narration={
            "intro": "One weight we know, one we don't. No scale. Just watching.",
            "locked": "Locked: the mystery weighs {mass:.0f} grams.",
            "hit": "The scale says {mass:.0f} grams. Inside the prediction.",
            "miss": "The scale says {mass:.0f} grams. Outside the prediction, and that goes on the scoreboard.",
        },
    ),
}
