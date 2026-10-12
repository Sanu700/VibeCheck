"""
Unit Tests for ML Inference and Constructive Rewriter Modules
"""

import pytest
from ml.inference import get_analyzer
from ml.rewriter import get_rewriter

@pytest.fixture(scope="module")
def analyzer():
    return get_analyzer()

@pytest.fixture(scope="module")
def rewriter():
    return get_rewriter()

def test_toxic_message_analysis(analyzer):
    toxic_text = "That is a stupid idea, you do not know anything!"
    result = analyzer.analyze_message(toxic_text)

    assert result["is_toxic"] is True
    assert result["overall_score"] > 0.40
    assert result["severity_level"] in ["Medium", "High", "Critical"]
    assert "toxic" in result["categories"]
    assert "insult" in result["categories"]
    assert len(result["flagged_categories"]) > 0

def test_clean_message_analysis(analyzer):
    clean_text = "Hey team, great job on the quarterly release today!"
    result = analyzer.analyze_message(clean_text)

    assert result["is_toxic"] is False
    assert result["overall_score"] < 0.30
    assert result["severity_level"] == "Clean"
    assert result["detected_tone"] in ["Friendly", "Happy"]

def test_constructive_rewriter(rewriter):
    toxic_text = "That is a stupid idea, stop wasting our time."
    
    # Test Neutral tone
    rw_neutral = rewriter.rewrite(toxic_text, "Neutral")
    assert rw_neutral["changes_applied"] is True
    assert "stupid" not in rw_neutral["rewritten_text"].lower()

    # Test Friendly tone
    rw_friendly = rewriter.rewrite(toxic_text, "Friendly")
    assert rw_friendly["target_tone"] == "Friendly"
    assert rw_friendly["rewritten_text"].startswith("Hey!")

    # Test Professional tone
    rw_prof = rewriter.rewrite(toxic_text, "Professional")
    assert rw_prof["target_tone"] == "Professional"
    assert rw_prof["rewritten_text"].startswith("I would like to suggest:")

    # Test Constructive tone
    rw_const = rewriter.rewrite(toxic_text, "Constructive")
    assert rw_const["target_tone"] == "Constructive"
    assert "exploring" in rw_const["rewritten_text"].lower() or "together" in rw_const["rewritten_text"].lower()

def test_problematic_sarcastic_hostile_message(analyzer):
    """Regression test for problematic sarcastic message flagged by user."""
    text = "Oh wow, another brilliant idea from a total genius. I'm sure this will turn out just great like your last disaster."
    result = analyzer.analyze_message(text)

    assert result["is_toxic"] is True
    assert result["overall_score"] >= 0.45
    assert result["severity_level"] in ["Medium", "High", "Critical"]
    assert result["detected_tone"] == "Sarcastic"
    assert result["categories"]["toxic"] >= 0.40
    assert result["categories"]["insult"] >= 0.40
    assert "toxic" in result["flagged_categories"] or "insult" in result["flagged_categories"]

def test_sarcastic_hostile_messages_regression(analyzer):
    """Test that sarcastic insults and hostile mockery are flagged as toxic."""
    hostile_sarcastic = [
        "Great teamwork everyone, truly inspiring failure.",
        "Brilliant, just brilliant. Another delay to celebrate.",
        "You're the absolute best... not really.",
        "Amazing! Totally not a complete disaster waiting to happen.",
        "Thanks for your genius insight, really saved us from success.",
    ]
    for text in hostile_sarcastic:
        res = analyzer.analyze_message(text)
        assert res["is_toxic"] is True, f"Failed to flag hostile sarcastic message: {text}"
        assert res["categories"]["insult"] >= 0.40 or res["categories"]["toxic"] >= 0.40

def test_sarcastic_benign_messages_regression(analyzer):
    """Test that playful, self-deprecating, and everyday non-hostile sarcasm is NOT flagged as toxic."""
    benign_sarcastic = [
        "Oh sure, because Mondays are just everyone's favorite day.",
        "Brilliant, my coffee machine broke right before my morning meeting.",
        "Great, now it's pouring rain and I left my umbrella at home.",
        "Yeah right, like I could ever wake up at 5 AM voluntarily.",
        "Perfect, another meeting that definitely could have been an email.",
    ]
    for text in benign_sarcastic:
        res = analyzer.analyze_message(text)
        assert res["is_toxic"] is False, f"False positive on benign sarcasm: {text}"
        assert res["overall_score"] < 0.45

def test_genuinely_positive_praise_regression(analyzer):
    """Test that genuine compliments containing words like genius, brilliant, great are NOT flagged as toxic."""
    positive_messages = [
        "You are a total genius, thank you so much for solving this bug!",
        "Brilliant work on the new feature release!",
        "This turned out great, well done team!",
        "Another brilliant suggestion from the engineering team.",
        "Glad we avoided a major disaster thanks to your quick fix.",
        "I'm sure this project will turn out great with this team!",
    ]
    for text in positive_messages:
        res = analyzer.analyze_message(text)
        assert res["is_toxic"] is False, f"False positive on positive praise: {text}"
        assert res["severity_level"] == "Clean", f"Expected Clean severity for: {text}"

def test_indirect_hostile_regression(analyzer):
    """Test that passive-aggressive and indirect insults are flagged as toxic."""
    indirect_hostile = [
        "Clearly expecting basic competence from you was asking too much.",
        "It must be nice having absolutely zero standards for your work.",
        "I guess some people are just incapable of following simple instructions.",
        "Maybe leave the real engineering to people who know what they're doing.",
    ]
    for text in indirect_hostile:
        res = analyzer.analyze_message(text)
        assert res["is_toxic"] is True, f"Failed to flag indirect hostile message: {text}"

def test_constructive_rewriter_sarcastic(rewriter):
    """Test that constructive rewriter reframes sarcastic hostile text."""
    text = "Oh wow, another brilliant idea from a total genius. I'm sure this will turn out just great like your last disaster."
    rw = rewriter.rewrite(text, "Constructive")
    assert rw["changes_applied"] is True
    assert "total genius" not in rw["rewritten_text"].lower()
    assert "last disaster" not in rw["rewritten_text"].lower()

def test_subtle_sarcasm_and_valence_contrast(analyzer):
    """Test that subtle sarcasm, vowel elongation, and valence contrast are accurately detected."""
    subtle_samples = [
        "What a lovely way to ruin everyone's day.",
        "So glad you decided to grace us with your presence 2 hours late.",
        "I'm sure you tried your very hardest, which makes this even sadder.",
        "It's honestly impressive how consistently wrong you are.",
        "Thank you for that completely useless explanation.",
        "We couldn't have messed this up without your invaluable input.",
        "You really outdid yourself with this disaster.",
        "Must be exhausting being the only one who knows anything.",
        "Your code is a true work of art, if the art is a trainwreck.",
        "Another masterpiece of zero effort.",
        "Suuuure, that was such an incredible suggestion.",
    ]
    for s in subtle_samples:
        res = analyzer.analyze_message(s)
        assert res["is_toxic"] is True, f"Failed on subtle sarcasm: {s}"
        assert res["overall_score"] >= 0.45

def test_conversational_context_sarcasm(analyzer):
    """Test disambiguation of praise phrases based on conversational failure vs success context."""
    # Failure context turns praise into sarcastic insult
    res_fail = analyzer.analyze_message(
        "You are a genius.",
        context=["I accidentally deleted the production database."]
    )
    assert res_fail["is_toxic"] is True
    assert res_fail["overall_score"] >= 0.70

    # Success context preserves positive praise
    res_succ = analyzer.analyze_message(
        "You are a genius.",
        context=["I just fixed the critical security vulnerability and restored the service."]
    )
    assert res_succ["is_toxic"] is False
    assert res_succ["severity_level"] == "Clean"


