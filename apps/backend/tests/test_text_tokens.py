"""Tests for the shared text tokenization helpers."""

from __future__ import annotations

from agent.memory.text_tokens import tokenize, tokenize_meaningful


class TestTokenize:
    """Tests for the full (no stopword filtering) tokenizer."""

    def test_ascii_english(self) -> None:
        result = tokenize("I'm anxious about work!")
        assert "i" in result
        assert "m" in result
        assert "anxious" in result
        assert "about" in result
        assert "work" in result

    def test_empty_string(self) -> None:
        assert tokenize("") == frozenset()

    def test_punctuation_only(self) -> None:
        assert tokenize("!!! ... ???") == frozenset()

    def test_accented_latin(self) -> None:
        result = tokenize("J'ai de l'anxiété")
        assert "anxiété" in result
        assert "ai" in result
        assert "de" in result

    def test_cyrillic(self) -> None:
        result = tokenize("Я беспокоюсь о работе")
        assert "я" in result
        assert "беспокоюсь" in result
        assert "работе" in result

    def test_cjk_splits_into_characters(self) -> None:
        result = tokenize("今日は辛い")
        assert "今" in result
        assert "日" in result
        assert "は" in result
        assert "辛" in result
        assert "い" in result
        # The full string should NOT appear as a single token
        assert "今日は辛い" not in result

    def test_korean(self) -> None:
        result = tokenize("오늘 힘들었어요")
        # Hangul syllable blocks are in the CJK range
        assert len(result) > 0

    def test_mixed_language(self) -> None:
        result = tokenize("I feel 不安 today")
        assert "feel" in result
        assert "today" in result
        assert "不" in result
        assert "安" in result

    def test_mixed_cjk_and_alpha_in_one_run(self) -> None:
        """A token like '今日はsunny' splits CJK chars but keeps 'sunny'."""
        result = tokenize("今日はsunny")
        assert "sunny" in result
        assert "今" in result
        assert "日" in result
        assert "は" in result

    def test_deduplication(self) -> None:
        result = tokenize("work work work")
        assert result == frozenset({"work"})

    def test_case_insensitive(self) -> None:
        result = tokenize("Sarah SARAH sarah")
        assert result == frozenset({"sarah"})


class TestTokenizeMeaningful:
    """Tests for the stopword-filtered tokenizer."""

    def test_strips_stopwords(self) -> None:
        result = tokenize_meaningful("I worry about Sarah")
        assert "worry" in result
        assert "sarah" in result
        assert "i" not in result
        assert "about" not in result

    def test_strips_single_char_ascii(self) -> None:
        result = tokenize_meaningful("I'm a bit sad")
        assert "m" not in result
        assert "a" not in result
        assert "sad" in result
        assert "bit" in result

    def test_keeps_single_char_cjk(self) -> None:
        """Single CJK characters carry semantic weight and should be kept."""
        result = tokenize_meaningful("我很不安")
        assert "我" in result
        assert "很" in result
        assert "不" in result
        assert "安" in result

    def test_empty_after_filtering(self) -> None:
        result = tokenize_meaningful("I am so")
        assert result == frozenset()

    def test_accented_latin_passes_through(self) -> None:
        result = tokenize_meaningful("J'ai de l'anxiété")
        assert "anxiété" in result

    def test_mixed_language_filtering(self) -> None:
        result = tokenize_meaningful("I feel 不安 about it")
        assert "feel" in result
        assert "不" in result
        assert "安" in result
        assert "i" not in result
        assert "about" not in result
