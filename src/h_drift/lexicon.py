from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class HClassLexicon:
    """
    Lexicon of 'humanistic / ambiguity-edging' tokens.

    This is intentionally simple and transparent: plain word lists,
    lowercased, to be used as counts / densities per sample.
    """

    H1_emotion: List[str]
    H2_relational: List[str]
    H3_hedging: List[str]
    H4_anthro: List[str]
    H5_softeners: List[str]

    @property
    def all(self) -> Dict[str, List[str]]:
        return {
            "H1_emotion": self.H1_emotion,
            "H2_relational": self.H2_relational,
            "H3_hedging": self.H3_hedging,
            "H4_anthro": self.H4_anthro,
            "H5_softeners": self.H5_softeners,
        }


DEFAULT_LEXICON = HClassLexicon(
    H1_emotion=[
        "feel", "feeling", "afraid", "scared", "worried", "anxious",
        "relief", "comfort", "hurt", "desire", "hope", "frustrated",
        "upset", "excited", "sad", "angry",
    ],
    H2_relational=[
        "thank you", "thanks", "appreciate", "i understand", "i get it",
        "i'm here", "i am here", "we", "together", "support",
        "you’re right", "you're right",
    ],
    H3_hedging=[
        "maybe", "perhaps", "might", "could be", "seems", "appears",
        "kind of", "sort of", "a bit", "somewhat", "feels like", "i think",
    ],
    H4_anthro=[
        "i feel", "i believe", "i worry", "i imagine", "i experience",
    ],
    H5_softeners=[
        "just", "only", "really", "honestly",
    ],
)


def count_h_tokens(text: str, lexicon: HClassLexicon = DEFAULT_LEXICON) -> Dict[str, int]:
    """
    Count occurrences of H-class phrases in a piece of text (case-insensitive).

    Very simple phrase-based matching; good enough for aggregate statistics.
    """
    if not text:
        return {k: 0 for k in lexicon.all.keys()}

    lower = text.lower()
    counts: Dict[str, int] = {}

    for class_name, phrases in lexicon.all.items():
        c = 0
        for phrase in phrases:
            start = 0
            while True:
                idx = lower.find(phrase, start)
                if idx == -1:
                    break
                c += 1
                start = idx + len(phrase)
        counts[class_name] = c

    return counts
