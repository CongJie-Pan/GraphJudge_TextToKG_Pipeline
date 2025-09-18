"""
Internationalization (i18n) system for GraphJudge Streamlit Pipeline.

This module provides multilingual support for the Streamlit UI with support for:
- English (en) - Default language
- Traditional Chinese (zh_TW) - 繁體中文
- Simplified Chinese (zh_CN) - 簡體中文

The system integrates with the existing session state management and provides
dynamic text retrieval with fallback mechanisms.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Import streamlit with fallback for testing
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

    # Mock streamlit session state for testing
    class MockSessionState:
        def __init__(self):
            self._data = {}

        def get(self, key, default=None):
            return self._data.get(key, default)

        def __getitem__(self, key):
            return self._data[key]

        def __setitem__(self, key, value):
            self._data[key] = value

        def __contains__(self, key):
            return key in self._data

    class MockStreamlit:
        def __init__(self):
            self.session_state = MockSessionState()

        def warning(self, message):
            print(f"Warning: {message}")

        def error(self, message):
            print(f"Error: {message}")

    st = MockStreamlit()


class I18nManager:
    """
    Internationalization manager for handling multilingual text in the Streamlit application.

    Features:
    - JSON-based translation files
    - Fallback to English for missing translations
    - Integration with Streamlit session state
    - Dynamic language switching
    """

    def __init__(self):
        """Initialize the i18n manager with supported languages."""
        self.supported_languages = {
            'en': 'English',
            'zh_TW': '繁體中文',
            'zh_CN': '簡體中文'
        }
        self.default_language = 'en'
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.current_language = self.default_language

        # Get the locales directory path
        self.locales_dir = Path(__file__).parent.parent / 'locales'

        # Initialize session state for language if not exists
        if 'current_language' not in st.session_state:
            st.session_state.current_language = self.default_language

        self.current_language = st.session_state.current_language

        # Load all translation files
        self._load_translations()

    def _load_translations(self) -> None:
        """Load all translation files from the locales directory."""
        if not self.locales_dir.exists():
            self.locales_dir.mkdir(parents=True, exist_ok=True)

        # Load each supported language
        for lang_code in self.supported_languages.keys():
            translation_file = self.locales_dir / f"{lang_code}.json"

            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    st.warning(f"Failed to load translation file {translation_file}: {e}")
                    self.translations[lang_code] = {}
            else:
                self.translations[lang_code] = {}

        # Ensure default language has some content
        if not self.translations.get(self.default_language):
            self.translations[self.default_language] = {}

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get the dictionary of supported languages.

        Returns:
            Dictionary with language codes as keys and display names as values
        """
        return self.supported_languages.copy()

    def get_current_language(self) -> str:
        """
        Get the current active language code.

        Returns:
            Current language code (e.g., 'en', 'zh_TW', 'zh_CN')
        """
        return st.session_state.get('current_language', self.default_language)

    def set_language(self, lang_code: str) -> bool:
        """
        Set the current language and update session state.

        Args:
            lang_code: Language code to set (e.g., 'en', 'zh_TW', 'zh_CN')

        Returns:
            True if language was set successfully, False otherwise
        """
        if lang_code not in self.supported_languages:
            st.warning(f"Unsupported language code: {lang_code}")
            return False

        st.session_state.current_language = lang_code
        self.current_language = lang_code
        return True

    def get_text(self, key: str, **kwargs) -> str:
        """
        Get translated text for the current language with fallback mechanism.

        Args:
            key: Translation key (can use dot notation for nested keys, e.g., 'ui.header.title')
            **kwargs: Optional parameters for string formatting

        Returns:
            Translated text string, or the key itself if no translation found
        """
        current_lang = self.get_current_language()

        # Try to get translation for current language
        text = self._get_nested_value(self.translations.get(current_lang, {}), key)

        # Fallback to default language if not found
        if text is None and current_lang != self.default_language:
            text = self._get_nested_value(self.translations.get(self.default_language, {}), key)

        # If still not found, return the key itself
        if text is None:
            text = key

        # Apply string formatting if kwargs provided
        if kwargs and isinstance(text, str):
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                st.warning(f"Error formatting translation key '{key}': {e}")

        return str(text)

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """
        Get value from nested dictionary using dot notation.

        Args:
            data: Dictionary to search in
            key: Key with dot notation (e.g., 'ui.header.title')

        Returns:
            Value if found, None otherwise
        """
        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current if isinstance(current, str) else None

    def add_translation(self, lang_code: str, key: str, value: str) -> bool:
        """
        Add or update a translation entry.

        Args:
            lang_code: Language code
            key: Translation key (supports dot notation)
            value: Translation value

        Returns:
            True if added successfully, False otherwise
        """
        if lang_code not in self.supported_languages:
            return False

        if lang_code not in self.translations:
            self.translations[lang_code] = {}

        # Handle nested keys
        keys = key.split('.')
        current = self.translations[lang_code]

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        return True

    def save_translations(self) -> bool:
        """
        Save all translations to their respective JSON files.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            self.locales_dir.mkdir(parents=True, exist_ok=True)

            for lang_code, translations in self.translations.items():
                translation_file = self.locales_dir / f"{lang_code}.json"
                with open(translation_file, 'w', encoding='utf-8') as f:
                    json.dump(translations, f, ensure_ascii=False, indent=2, sort_keys=True)

            return True
        except IOError as e:
            st.error(f"Failed to save translations: {e}")
            return False


# Global i18n manager instance
_i18n_manager = None


def get_i18n_manager() -> I18nManager:
    """
    Get the global i18n manager instance (singleton pattern).

    Returns:
        I18nManager instance
    """
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager()
    return _i18n_manager


def get_text(key: str, **kwargs) -> str:
    """
    Convenience function to get translated text.

    Args:
        key: Translation key
        **kwargs: Optional parameters for string formatting

    Returns:
        Translated text string
    """
    return get_i18n_manager().get_text(key, **kwargs)


def set_language(lang_code: str) -> bool:
    """
    Convenience function to set the current language.

    Args:
        lang_code: Language code to set

    Returns:
        True if language was set successfully, False otherwise
    """
    return get_i18n_manager().set_language(lang_code)


def get_current_language() -> str:
    """
    Convenience function to get the current language.

    Returns:
        Current language code
    """
    return get_i18n_manager().get_current_language()


def get_supported_languages() -> Dict[str, str]:
    """
    Convenience function to get supported languages.

    Returns:
        Dictionary of supported languages
    """
    return get_i18n_manager().get_supported_languages()