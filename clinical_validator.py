"""ASR-aware contradiction checking for clinical transcripts.

Speech-to-text output is noisy, so a rule that fires on a literal string match
produces false positives faster than a reviewer can clear them. Every check here
is therefore scored rather than fatal: a rule contributes a severity-weighted
penalty, negated and hypothetical phrasing suppresses it, and transcript length
and clinical term density earn confidence back. Callers route on the score
instead of on a boolean, so a long, dense, mostly-sound transcript with one
MEDIUM hit still clears while a short garbled one does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

IGNORE_CASE = re.IGNORECASE

# Severity weights. CRITICAL alone sinks a transcript; two MEDIUMs do not.
PENALTY = {"CRITICAL": 0.6, "HIGH": 0.4, "MEDIUM": 0.2}
UNIT_PENALTY = 0.1

# Credit back for signals that a transcript is substantial rather than garbled.
LENGTH_CREDIT = [(200, 0.1), (500, 0.05)]
TERM_DENSITY_CREDIT = 0.05
TERM_DENSITY_MIN = 5

# Characters of transcript scanned either side of a hit for negation cues.
CONTEXT_WINDOW = 100


@dataclass(frozen=True)
class Rule:
    """Two patterns that cannot both describe the same patient at once."""

    left: str
    right: str
    severity: str
    reason: str


# Mutually exclusive clinical states. Both sides must appear, in either order.
CONTRADICTION_RULES = [
    Rule(r"\bdeceased|\bexpired\b", r"\b(ambulat\w+|responsive|conversant)\b", "CRITICAL",
         "a deceased patient cannot be ambulatory or responsive"),
    Rule(r"\bbenign\b", r"\bmetastat\w+|\bmetastas[ei]s\b", "HIGH",
         "benign lesions do not metastasise"),
    Rule(r"\bafebrile\b", r"\b(fever|febrile|pyrexia)\b", "HIGH",
         "afebrile and febrile are mutually exclusive"),
    Rule(r"\bnormotensive\b", r"\b(hypertensive|hypotensive)\b", "HIGH",
         "normotensive excludes both hypertension and hypotension"),
    Rule(r"\b(paraplegic|quadriplegic|paralysed|paralyzed)\b", r"\b(ambulat\w+|gait unremarkable)\b", "HIGH",
         "a paralysed patient has no independent gait"),
    Rule(r"\b(nil by mouth|NBM|npo)\b", r"\b(tolerating|ate|oral intake)\b", "MEDIUM",
         "nil by mouth conflicts with recorded oral intake"),
    Rule(r"\banuric\b", r"\burine output of \d", "MEDIUM",
         "anuria conflicts with a measured urine output"),
]

# Units attached to the wrong observation. Cheap, and a common ASR failure:
# spoken units are short, similar-sounding and easy to misattach.
UNIT_RULES = [
    (re.compile(r"\btemperature\s*:?\s*[\d.]+\s*mm\s?hg\b", IGNORE_CASE),
     "temperature reported in mmHg"),
    (re.compile(r"\b(blood pressure|bp)\s*:?\s*[\d/]+\s*(°|degrees?)\s*[cf]\b", IGNORE_CASE),
     "blood pressure reported in degrees"),
    (re.compile(r"\b(heart rate|pulse)\s*:?\s*\d+\s*mm\s?hg\b", IGNORE_CASE),
     "heart rate reported in mmHg"),
    (re.compile(r"\b(weight|height)\s*:?\s*[\d.]+\s*mm\s?hg\b", IGNORE_CASE),
     "weight or height reported in mmHg"),
]

# Phrasing that discusses a finding without asserting it. Suppresses a hit.
HEDGES = re.compile(
    r"\b(no|not|denies|denied|without|negative for|ruled out|resolved|"
    r"if|should|would|in case of|query|suspected|rather than|as opposed to|"
    r"previously|history of|family history)\b",
    IGNORE_CASE,
)

# Density signal only. Not a terminology whitelist, so it stays short.
CLINICAL_TERMS = re.compile(
    r"\b(patient|history|examination|diagnosis|assessment|plan|medication|"
    r"dose|mg|ml|allergy|observation|referral|follow[- ]up|vitals|"
    r"presents|reports|denies|prescribed)\b",
    IGNORE_CASE,
)

COMPILED_RULES = [(re.compile(r.left, IGNORE_CASE), re.compile(r.right, IGNORE_CASE), r) for r in CONTRADICTION_RULES]


@dataclass
class Result:
    confidence: float
    findings: List[str] = field(default_factory=list)

    def accepted(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold


def _hedged(text: str, start: int, end: int) -> bool:
    """True when the span sits inside negated or hypothetical phrasing."""
    return bool(HEDGES.search(text[max(0, start - CONTEXT_WINDOW):end + CONTEXT_WINDOW]))


def validate(text: str) -> Result:
    """Score a transcript. Higher confidence means safer to pass through."""
    text = text or ""
    findings: List[str] = []
    score = 1.0

    for left, right, rule in COMPILED_RULES:
        hit_left, hit_right = left.search(text), right.search(text)
        if not (hit_left and hit_right):
            continue
        start = min(hit_left.start(), hit_right.start())
        end = max(hit_left.end(), hit_right.end())
        if _hedged(text, start, end):
            continue
        findings.append(f"{rule.severity}: {rule.reason}")
        score -= PENALTY[rule.severity]

    for pattern, reason in UNIT_RULES:
        hit = pattern.search(text)
        if hit and not _hedged(text, hit.start(), hit.end()):
            findings.append(f"UNIT: {reason}")
            score -= UNIT_PENALTY

    for length, credit in LENGTH_CREDIT:
        if len(text) > length:
            score += credit
    if len(CLINICAL_TERMS.findall(text)) > TERM_DENSITY_MIN:
        score += TERM_DENSITY_CREDIT

    return Result(confidence=max(0.0, min(1.0, score)), findings=findings)


def _self_check() -> None:
    clean = (
        "Patient presents for follow-up. History of asthma, no acute distress. "
        "Observation: temperature 37.1 degrees C, blood pressure 122/78 mmHg, heart rate 72. "
        "Assessment: stable. Plan: continue prescribed medication at the current dose, "
        "review allergy list, referral to respiratory clinic, follow-up in six weeks."
    )
    assert validate(clean).findings == [], validate(clean).findings
    assert validate(clean).accepted()

    critical = validate("The deceased patient was ambulatory on the ward this morning.")
    assert any(f.startswith("CRITICAL") for f in critical.findings)
    assert not critical.accepted()

    # Negation suppresses a rule that would otherwise fire.
    assert validate("Afebrile, negative for fever throughout admission.").findings == []
    assert validate("Benign lesion, no metastatic disease identified.").findings == []

    # Hedged discussion of a contradiction is not an assertion of one.
    assert validate("Benign on biopsy, so metastasis was ruled out.").findings == []

    units = validate("Observation: temperature 120 mmHg, pulse 80 mmHg.")
    assert len(units.findings) == 2, units.findings

    # Length and density pull a single MEDIUM hit back over the threshold,
    # which is the whole point of scoring instead of failing outright.
    padded = clean + " Nil by mouth since midnight, though tolerating sips of water."
    scored = validate(padded)
    assert scored.findings and scored.accepted(), (scored.confidence, scored.findings)

    assert validate("").confidence == 1.0
    print(f"ok - {len(CONTRADICTION_RULES)} contradiction rules, {len(UNIT_RULES)} unit rules")


if __name__ == "__main__":
    _self_check()
