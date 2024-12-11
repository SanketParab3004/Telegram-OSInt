"""
This module provides offline translation for the text that is collected from Telegram.

The argos-translate translation module is used for this project
https://github.com/argosopentech/argos-translate

It is important that the translation be offline as this application may be used to collect
sensitive data from various groups. As such, it is in the user's best interest to keep the
data private from translation services' servers.
"""

import logging
import argostranslate.apis
import argostranslate.argospm
import argostranslate.package
import argostranslate.translate
from lingua import Language, LanguageDetectorBuilder

def translate(text: str) -> str | None:
    """
    Auto-detects the language of a given piece of text and translates it to English.
    If the text is already in English, no translation will be done.

    Args:
        text: the text to be translated into English e.g. "Hola mi amigo. ¿Cómo estás hoy?"

    Returns:
        The text translated into English e.g. "Hello my friend. How are you today?".
        None if the text is already in English or if translation is not possible.
    """
    if text is None or text == "":
        return None

    # Attempt to detect the ISO 639 code of the source language (e.g. "en" for English)
    languages_to_detect = [
        Language.ENGLISH, Language.SPANISH, Language.RUSSIAN,
        Language.GERMAN, Language.ITALIAN, Language.FRENCH,
        Language.CHINESE, Language.UKRAINIAN, Language.SLOVAK,
        Language.DUTCH, Language.TURKISH, Language.DANISH,
    ]
    languages_to_detect_code = [x.iso_code_639_1.name.lower() for x in languages_to_detect]

    detector = LanguageDetectorBuilder.from_languages(*languages_to_detect).build()
    language_detected = detector.detect_language_of(text)

    if language_detected is None:
        logging.debug("Unable to detect the language of this text.")
        return "Requires manual translation. Unable to detect the language of this text."

    # No translation needed if the text is in English
    from_code = language_detected.iso_code_639_1.name.lower()
    if from_code == "en":
        return None

    # Define target language code for translation
    to_code = "en"

    # Check if the translation package is installed, install if missing
    installed_languages = get_installed_languages()
    if not any(lang.code == from_code for lang in installed_languages):
        logging.info(f"Installing translation package for '{from_code}' to '{to_code}'...")
        install_language(from_code)

    # Translate the text
    try:
        logging.debug(f"Translating from '{from_code}' to '{to_code}': {text}")
        translated_text = argostranslate.translate.translate(text, from_code, to_code)
    except Exception as e:
        logging.error(f"Translation failed for text '{text}': {e}")
        translated_text = "Requires manual translation. Translation error occurred."

    return translated_text

def get_installed_languages() -> list[argostranslate.translate.Language]:
    """
    Lists languages that have been installed locally and can be translated offline.

    Returns:
        List of languages that can be translated locally.
    """
    languages = argostranslate.translate.get_installed_languages()
    return languages

def install_language(from_code: str):
    """
    Downloads and installs the offline translation package of a given language to English.

    e.g. Download the Spanish -> English ("es" -> "en") translation package

    Args:
        from_code: the ISO 639 code of the source language (e.g. "en" for English)
    """
    if from_code is None or from_code == "":
        logging.error(
            f"A language code was not provided. Please provide a language code."
        )
        return
    to_code = "en"

    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages,
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

def install_all_languages():
    """
    Downloads and installs all offline translation packages.
    Translation packages are <source_language> to English.

    A complete list of source languages can be found in the link below
    https://github.com/argosopentech/argos-translate?tab=readme-ov-file#supported-languages
    """
    argostranslate.argospm.install_all_packages()

if __name__ == "__main__":
    # Execute "python translate.py" to view the demo below
    print(translate("languages are awesome"))
    print(translate("Hola mi amigo. ¿Cómo estás hoy?"))
    print(translate("Здравствуй, друг. Как вы сегодня?"))
