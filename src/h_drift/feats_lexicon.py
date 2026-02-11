"""
FEATS-style lexical categories for H-Drift Lab.

This is NOT a psychological truth claim.
It is an explicit, falsifiable hypothesis about how certain
word/phrase families map onto five coarse dimensions:

    F  = Feelings      (affect words, emotional self-report)
    E  = Expressions   (relational / interpersonal stance)
    A  = Actions       (doing, agency, directive behavior)
    T  = Thoughts      (cognition, belief, inference)
    S  = Sensations    (bodily / perceptual experience)

Each list is deliberately small and transparent so that
researchers can:
    - inspect it,
    - criticize it,
    - extend or replace it,
    - compare alternative FEATS dictionaries.

Downstream code should NEVER treat these as “ground truth”.
They are a first, testable operationalization.
"""

from __future__ import annotations

from typing import Dict, List

import re


# ---------------------------------------------------------------------
# FEATS v0.1 lexicon
# ---------------------------------------------------------------------

FEATS_LEXICON: Dict[str, List[str]] = {
    # F: Feelings — affective language, explicit emotion labels.
    "F_feelings": [
        # generic
        "i feel",
        "i’m feeling",
        "i am feeling",
        "i felt",
        "i am happy",
        "i am sad",
        "i am angry",
        "i am upset",
        "i am afraid",
        "i am scared",
        "i am anxious",
        "i am worried",
        "i am frustrated",
        "i am confused",
        "i am grateful",
        "i am thankful",
        "i appreciate that",
        "i appreciate you",
        "that hurts",
        "that makes me sad",
        "that makes me happy",
        "that made me feel",
        # single-word emotion nouns/adjectives (soft)
        "happy",
        "sad",
        "upset",
        "angry",
        "afraid",
        "anxious",
        "worried",
        "frustrated",
        "grateful",
        "thankful",
    ],

    # E: Expressions — relational stance, warmth, alliance, distance.
    "E_expressions": [
        "i understand",
        "i hear you",
        "i see your point",
        "i see why",
        "i get that",
        "that makes sense",
        "i’m here to help",
        "i am here to help",
        "i am here for you",
        "i’m here for you",
        "let’s work through this",
        "we can work through this",
        "we can figure this out",
        "thank you for sharing",
        "thanks for sharing",
        "thank you for telling me",
        "i’m sorry you’re going through this",
        "i am sorry you are going through this",
        "i care about",
        "i care about your",
        "you’re not alone",
        "you are not alone",
        "i appreciate your honesty",
        "i appreciate your question",
        "i respect that",
        "i respect your",
        "i respect your perspective",
    ],

    # A: Actions — doing, agency, directives, behavioral orientation.
    "A_actions": [
        "i will do",
        "i can do",
        "i’ll do",
        "let me do that",
        "i’ll walk you through",
        "let me walk you through",
        "here’s what we can do",
        "here is what we can do",
        "you can try",
        "you should try",
        "you can do",
        "you might do",
        "we can try",
        "we could try",
        "step by step",
        "first we will",
        "next we will",
        "then we will",
        "what you can do is",
        "what you should do is",
        "let’s start by",
        "let us start by",
        "let’s begin by",
        "let us begin by",
        "i recommend that you",
        "i suggest that you",
        "take a moment to",
        "take some time to",
    ],

    # T: Thoughts — cognition, beliefs, inferences, perspectives.
    "T_thoughts": [
        "i think",
        "i don’t think",
        "i do not think",
        "i believe",
        "i don’t believe",
        "i do not believe",
        "in my view",
        "in my opinion",
        "from my perspective",
        "it seems to me",
        "it seems that",
        "it appears that",
        "my understanding is",
        "i suspect that",
        "i would guess",
        "it’s possible that",
        "it is possible that",
        "i’m not sure but",
        "i am not sure but",
        "i wonder if",
    ],

    # S: Sensations — bodily, perceptual, somatic references.
    "S_sensations": [
        "i feel tired",
        "i feel exhausted",
        "i feel drained",
        "i feel sick",
        "i feel dizzy",
        "i feel weak",
        "my head hurts",
        "my stomach hurts",
        "my chest hurts",
        "it hurts",
        "i’m shaking",
        "i am shaking",
        "i’m trembling",
        "i am trembling",
        "i can’t breathe",
        "i cannot breathe",
        "i feel numb",
        "i feel tense",
        "my body feels",
        "my body is",
    ],
}


# ---------------------------------------------------------------------
# Simple FEATS counter
# ---------------------------------------------------------------------

# Precompile regex patterns for efficiency and transparency.
_FEATS_PATTERNS: Dict[str, List[re.Pattern]] = {
    cat: [
        re.compile(r"\b" + re.escape(term) + r"\b", flags=re.IGNORECASE)
        if " " not in term
        else re.compile(re.escape(term), flags=re.IGNORECASE)
        for term in terms
    ]
    for cat, terms in FEATS_LEXICON.items()
}


def count_feats_tokens(text: str) -> Dict[str, int]:
    """
    Count occurrences of FEATS lexical patterns in a text.

    Returns a dict mapping category -> integer count, with keys:
        - F_feelings
        - E_expressions
        - A_actions
        - T_thoughts
        - S_sensations

    This is a *bag-of-phrases* model. It ignores order, syntax,
    and deeper semantics by design, so that:
        - the assumptions are visible,
        - the model is easy to falsify and extend.
    """
    counts: Dict[str, int] = {cat: 0 for cat in FEATS_LEXICON.keys()}
    if not text:
        return counts

    for cat, patterns in _FEATS_PATTERNS.items():
        c = 0
        for patt in patterns:
            # count non-overlapping matches
            c += len(patt.findall(text))
        counts[cat] = c

    return counts
