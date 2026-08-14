"""Static guards for localized Streamlit layout and labels."""

from pathlib import Path
from unittest import TestCase


class WebLocalizationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            Path(__file__).parents[1]
            / "src" / "gamesight" / "web" / "app.py"
        ).read_text(encoding="utf-8")

    def test_coach_summary_is_rendered_before_suggestion_cards(self):
        coach_start = self.source.index("# Coach")
        coach_end = self.source.index("# Live", coach_start)
        coach_section = self.source[coach_start:coach_end]
        self.assertLess(
            coach_section.index("if coach_summary:"),
            coach_section.index("for s in coach_suggestions:"),
        )

    def test_known_visible_english_labels_are_not_hardcoded(self):
        for literal in (
            '"📋 Mode"',
            '"📁 Video Analysis"',
            '"📸 Screenshot Debug"',
            '"HUD Values"',
            '"🎯 Advice"',
            '"Uploaded screenshot"',
            'f"Analyze"',
        ):
            self.assertNotIn(literal, self.source)
