"""Tests for i18n loader."""

from unittest import TestCase

from gamesight.i18n.loader import I18nLoader


class I18nLoaderBasicTests(TestCase):
    def setUp(self) -> None:
        self.en = I18nLoader("en")
        self.zh = I18nLoader("zh-CN")

    def test_locale_property(self) -> None:
        self.assertEqual(self.en.locale, "en")
        self.assertEqual(self.zh.locale, "zh-CN")

    def test_lang_name(self) -> None:
        self.assertEqual(self.en.lang_name, "English")
        self.assertEqual(self.zh.lang_name, "简体中文")

    def test_simple_lookup(self) -> None:
        self.assertEqual(self.en.t("app.title"), "GameSight AI")

    def test_nested_lookup(self) -> None:
        self.assertEqual(self.en.t("overview.match_overview"), "Match Overview")

    def test_interpolation(self) -> None:
        result = self.en.t("run.processing", w=1920, h=1080, fps=60)
        self.assertIn("1920", result)
        self.assertIn("1080", result)
        self.assertIn("60", result)

    def test_missing_key_fallback(self) -> None:
        self.assertEqual(self.en.t("nonexistent.key.path"), "nonexistent.key.path")

    def test_chinese_locale(self) -> None:
        self.assertEqual(self.zh.t("app.subtitle"), "CS2 POV 分析管线")

    def test_chinese_interpolation(self) -> None:
        result = self.zh.t("evidence.count", n=42)
        self.assertIn("42", result)

    def test_available_locales(self) -> None:
        locales = I18nLoader.available_locales()
        self.assertIn("en", locales)
        self.assertIn("zh-CN", locales)

    def test_invalid_locale_raises(self) -> None:
        with self.assertRaises(ValueError):
            I18nLoader("fr")
