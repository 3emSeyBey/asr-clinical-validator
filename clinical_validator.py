"""Scores clinical transcripts for contradictions instead of rejecting them.

Rules fire against speech-to-text output, which is noisy, so each one costs
confidence rather than failing the transcript outright. Negated and
hypothetical phrasing suppresses a hit. Length and clinical term density
give confidence back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

IGNORE_CASE = re.IGNORECASE

PENALTY = {"CRITICAL": 0.6, "HIGH": 0.4, "MEDIUM": 0.2, "UNIT": 0.1}

LENGTH_CREDIT = [(200, 0.1), (500, 0.05)]
TERM_DENSITY_CREDIT = 0.05
TERM_DENSITY_MIN = 5

CONTEXT_WINDOW = 100

# (left, right, severity, reason). Both patterns must hit, in either order.
# right=None means the left pattern alone is the finding.
RULES: List[Tuple[str, Optional[str], str, str]] = [
    (r"\bdeceased\b|\bexpired\b", r"\b(ambulat\w+|responsive|conversant)\b", "CRITICAL",
     "deceased patient recorded as ambulatory or responsive"),
    (r"\bbenign\b", r"\bmetastat\w+|\bmetastas[ei]s\b", "HIGH",
     "benign lesion recorded as metastasising"),
    (r"\bafebrile\b", r"\b(fever|febrile|pyrexia)\b", "HIGH",
     "afebrile and febrile in the same note"),
    (r"\bnormotensive\b", r"\b(hypertensive|hypotensive)\b", "HIGH",
     "normotensive with hypertension or hypotension"),
    (r"\b(paraplegic|quadriplegic|paralysed|paralyzed)\b", r"\b(ambulat\w+|gait unremarkable)\b", "HIGH",
     "paralysed patient recorded with independent gait"),
    (r"\b(nil by mouth|NBM|npo)\b", r"\b(tolerating|ate|oral intake)\b", "MEDIUM",
     "nil by mouth with recorded oral intake"),
    (r"\banuric\b", r"\burine output of \d", "MEDIUM",
     "anuria with a measured urine output"),
    # Units attach to the wrong observation often: spoken units are short and
    # sound alike.
    (r"\btemperature\s*:?\s*[\d.]+\s*mm\s?hg\b", None, "UNIT",
     "temperature in mmHg"),
    (r"\b(blood pressure|bp)\s*:?\s*[\d/]+\s*(°|degrees?)\s*[cf]\b", None, "UNIT",
     "blood pressure in degrees"),
    (r"\b(heart rate|pulse)\s*:?\s*\d+\s*mm\s?hg\b", None, "UNIT",
     "heart rate in mmHg"),
    (r"\b(weight|height)\s*:?\s*[\d.]+\s*mm\s?hg\b", None, "UNIT",
     "weight or height in mmHg"),
]

COMPILED = [
    (re.compile(left, IGNORE_CASE), re.compile(right, IGNORE_CASE) if right else None, severity, reason)
    for left, right, severity, reason in RULES
]

# Phrasing that discusses a finding without asserting it.
HEDGES = re.compile(
    r"\b(no|not|denies|denied|without|negative for|ruled out|resolved|"
    r"if|should|would|in case of|query|suspected|rather than|as opposed to|"
    r"previously|history of|family history)\b",
    IGNORE_CASE,
)

# Density signal, not a terminology whitelist.
CLINICAL_TERMS = re.compile(
    r"\b(patient|history|examination|diagnosis|assessment|plan|medication|"
    r"dose|mg|ml|allergy|observation|referral|follow[- ]up|vitals|"
    r"presents|reports|denies|prescribed)\b",
    IGNORE_CASE,
)


@dataclass
class Result:
    confidence: float
    findings: List[str] = field(default_factory=list)

    def accepted(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold


def _hedged(text: str, start: int, end: int) -> bool:
    return bool(HEDGES.search(text[max(0, start - CONTEXT_WINDOW):end + CONTEXT_WINDOW]))


def validate(text: str) -> Result:
    """Score a transcript. Higher confidence means safer to pass through."""
    text = text or ""
    findings: List[str] = []
    score = 1.0

    for left, right, severity, reason in COMPILED:
        hit_left = left.search(text)
        if not hit_left:
            continue
        hit_right = right.search(text) if right else hit_left
        if not hit_right:
            continue
        start = min(hit_left.start(), hit_right.start())
        end = max(hit_left.end(), hit_right.end())
        if _hedged(text, start, end):
            continue
        findings.append(f"{severity}: {reason}")
        score -= PENALTY[severity]

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

    assert validate("Afebrile, negative for fever throughout admission.").findings == []
    assert validate("Benign lesion, no metastatic disease identified.").findings == []
    assert validate("Benign on biopsy, so metastasis was ruled out.").findings == []

    units = validate("Observation: temperature 120 mmHg, pulse 80 mmHg.")
    assert len(units.findings) == 2, units.findings

    # One MEDIUM hit in a long dense transcript still clears.
    scored = validate(clean + " Nil by mouth since midnight, though tolerating sips of water.")
    assert scored.findings and scored.accepted(), (scored.confidence, scored.findings)

    assert validate("").confidence == 1.0
    print(f"ok - {len(RULES)} rules")


if __name__ == "__main__":
    _self_check()
