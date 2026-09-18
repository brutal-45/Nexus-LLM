#!/usr/bin/env python3
"""
Internationalization (i18n) Example - Nexus-LLM
==================================================
Demonstrates how to use Nexus-LLM's internationalization features
for multilingual conversations and localized responses.
"""

from nexus_llm import InferenceEngine, Conversation
from nexus_llm.i18n import (
    Locale,
    I18nManager,
    TranslationPipeline,
    LanguageDetector,
)


def main():
    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

    # --- Initialize i18n manager ---
    i18n = I18nManager(
        default_locale=Locale.EN_US,
        supported_locales=[
            Locale.EN_US,       # English (US)
            Locale.EN_GB,       # English (UK)
            Locale.ZH_CN,       # Chinese (Simplified)
            Locale.JA_JP,       # Japanese
            Locale.KO_KR,       # Korean
            Locale.ES_ES,       # Spanish
            Locale.FR_FR,       # French
            Locale.DE_DE,       # German
            Locale.AR_SA,       # Arabic
            Locale.PT_BR,       # Portuguese (Brazil)
        ],
        fallback_locale=Locale.EN_US,
    )

    # --- Auto-detect language ---
    print("=" * 60)
    print("Language Detection")
    print("=" * 60)

    detector = LanguageDetector()
    test_texts = [
        "Hello, how are you today?",
        "こんにちは、お元気ですか？",
        "你好，你好吗？",
        "Hola, ¿cómo estás?",
        "Bonjour, comment allez-vous?",
        "안녕하세요, 어떻게 지내세요?",
    ]

    for text in test_texts:
        detected = detector.detect(text)
        locale = i18n.get_locale(detected.language)
        print(f"  '{text}' -> {detected.language} ({locale.display_name}), "
              f"confidence: {detected.confidence:.2f}")

    # --- Multilingual conversation ---
    print("\n" + "=" * 60)
    print("Multilingual Conversation")
    print("=" * 60)

    conversation = Conversation(
        system_prompt="You are a multilingual assistant. Respond in the same language as the user.",
    )

    # Ask in different languages
    queries = [
        ("What are the best practices for software development?", Locale.EN_US),
        ("软件开发有哪些最佳实践？", Locale.ZH_CN),
        ("ソフトウェア開発のベストプラクティスは何ですか？", Locale.JA_JP),
    ]

    for query, expected_locale in queries:
        conversation.add_user_message(query)
        response = engine.chat(conversation, locale=expected_locale)
        detected = detector.detect(response.text)
        print(f"\nQ: {query}")
        print(f"A: {response.text[:150]}...")
        print(f"Response language: {detected.language} (expected: {expected_locale})")

    # --- Translation pipeline ---
    print("\n" + "=" * 60)
    print("Translation Pipeline")
    print("=" * 60)

    translator = TranslationPipeline(engine=engine)

    text = "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."

    translations = translator.translate_batch(
        text=text,
        source_locale=Locale.EN_US,
        target_locales=[Locale.ZH_CN, Locale.JA_JP, Locale.ES_ES, Locale.FR_FR],
    )

    print(f"Source (en-US): {text}")
    for locale, translated in translations.items():
        print(f"  {locale}: {translated}")

    # --- Localized system prompts ---
    print("\n" + "=" * 60)
    print("Localized System Prompts")
    print("=" * 60)

    for locale in [Locale.EN_US, Locale.ZH_CN, Locale.JA_JP]:
        system_prompt = i18n.get_system_prompt(
            locale=locale,
            persona="professional_assistant",
        )
        print(f"\n{locale}:\n  {system_prompt}")

    # --- Locale-aware formatting ---
    print("\n" + "=" * 60)
    print("Locale-Aware Output Formatting")
    print("=" * 60)

    data = {
        "date": "2024-03-15",
        "number": 1234567.89,
        "currency": 99.99,
    }

    for locale in [Locale.EN_US, Locale.DE_DE, Locale.FR_FR, Locale.ZH_CN]:
        formatted = i18n.format_data(data, locale=locale)
        print(f"\n{locale}:")
        print(f"  Date: {formatted['date']}")
        print(f"  Number: {formatted['number']}")
        print(f"  Currency: {formatted['currency']}")

    # --- Save i18n configuration ---
    i18n.save_config("./config/i18n_config.yaml")
    print("\ni18n configuration saved to ./config/i18n_config.yaml")


if __name__ == "__main__":
    main()
