"""
Constructive Rewriting Engine for ToxicBuddy 2.0
Transforms toxic, aggressive, sarcastic, or rude messages into respectful,
constructive alternatives while preserving the original intent.
Supports tones: Neutral, Friendly, Professional, Constructive.
"""

import re
import random

# Mapping of hostile/toxic phrases to polite intent placeholders
PHRASE_REPLACEMENTS = [
    (r"\b(that is|that'?s|this is) (a )?stupid (idea|plan|thought|code)\b", "I have concerns about this approach"),
    (r"\byou (do not|don'?t) know (anything|what you'?re doing)\b", "let's review the requirements together"),
    (r"\bstop wasting (our|my|the) time\b", "let's focus our discussion on key priorities"),
    (r"\bcan you do anything right\b", "let's walk through the steps to get this right"),
    (r"\byou'?re (so|really|totally)? ?(annoying|useless|incompetent|dumb|foolish)\b", "I would appreciate a more collaborative approach"),
    (r"\bthis (code|work|report|doc) is (complete|total)? ?garbage\b", "this deliverable needs further revision and improvement"),
    (r"\bwhat the (fuck|hell|shit) is this\b", "could you please clarify this implementation?"),
    (r"\bshut up\b", "please allow others to share their input"),
    (r"\bpiss off|fuck off|get lost\b", "let's take a step back from this topic"),
    (r"\byou never listen\b", "it feels like my feedback isn't being addressed"),
    (r"\bi'?m (seriously|really)? ?(pissed off|furious|mad)\b", "I am feeling quite frustrated with this issue"),
    (r"\bkya bakwaas (kar raha hai|hai)\b", "this explanation seems unclear to me"),
    (r"\barey chup kar\b", "let's pause and continue calmly"),
    (r"\bi will (break your face|hurt you|ruin your life)\b", "I feel strongly about this issue and need us to resolve it fairly"),
    (r"\b(oh wow, )?(another )?brilliant idea from a total genius\b", "I have some reservations about this suggestion"),
    (r"\b(turn out )?just great like your last disaster\b", "let's ensure we avoid past issues"),
    (r"\bsuch a genius move by you again\b", "let's carefully reconsider this move"),
    (r"\btruly inspiring failure\b", "an outcome we should learn from"),
    (r"\bcomplete disaster waiting to happen\b", "an approach that carries significant risk"),
]

# Profanity cleaner
PROFANITY_PATTERN = re.compile(
    r"\b(fuck|fucking|shit|bullshit|bitch|bastard|asshole|crap|damn)\b",
    re.IGNORECASE
)

class ConstructiveRewriter:
    """
    Constructive Text Rewriter.
    Reframes flagged toxic/rude inputs into constructive, polite alternatives.
    """

    def rewrite(self, text: str, tone: str = "Constructive") -> dict:
        """
        Rewrites a given text into the specified tone.
        Target tones: 'Neutral', 'Friendly', 'Professional', 'Constructive'
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "rewritten_text": text,
                "target_tone": tone,
                "changes_applied": False
            }

        tone_clean = tone.capitalize() if tone else "Constructive"
        if tone_clean not in ["Neutral", "Friendly", "Professional", "Constructive"]:
            tone_clean = "Constructive"

        original_clean = text.strip()
        working_text = original_clean

        # Step 1: Replace known hostile phrase patterns
        changes_made = False
        for pattern, replacement in PHRASE_REPLACEMENTS:
            if re.search(pattern, working_text, flags=re.IGNORECASE):
                working_text = re.sub(pattern, replacement, working_text, flags=re.IGNORECASE)
                changes_made = True

        # Step 2: Remove or replace explicit profanities
        if PROFANITY_PATTERN.search(working_text):
            working_text = PROFANITY_PATTERN.sub("issue", working_text)
            changes_made = True

        # Step 3: Remove trailing hostile emojis (e.g. 🤬, 😡, 😤)
        working_text = re.sub(r"[🤬😡😤👎🤮💥😠]+", "", working_text).strip()

        # Step 4: Adapt to requested target tone
        final_text = self._apply_tone_framing(working_text, tone_clean, original_clean, changes_made)

        return {
            "original_text": original_clean,
            "rewritten_text": final_text,
            "target_tone": tone_clean,
            "changes_applied": changes_made or (final_text != original_clean)
        }

    def _apply_tone_framing(self, text: str, tone: str, original: str, changes_made: bool) -> str:
        """Formats the reframed text into the requested target tone."""
        # Clean up double spaces or sentence formatting
        text = re.sub(r"\s+", " ", text).strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and text[-1] not in [".", "!", "?"]:
            text += "."

        if tone == "Neutral":
            return text

        elif tone == "Friendly":
            prefix = "Hey! "
            return f"{prefix}{text}"

        elif tone == "Professional":
            if not text.startswith("I recommend") and not text.startswith("Please") and not text.startswith("I would"):
                text = f"I would like to suggest: {text}"
            return text

        elif tone == "Constructive":
            if changes_made:
                return f"{text} What do you think about exploring this together?"
            else:
                return f"Here is a constructive perspective: {text}"

        return text

# Singleton instance
_rewriter_instance = None

def get_rewriter() -> ConstructiveRewriter:
    global _rewriter_instance
    if _rewriter_instance is None:
        _rewriter_instance = ConstructiveRewriter()
    return _rewriter_instance
