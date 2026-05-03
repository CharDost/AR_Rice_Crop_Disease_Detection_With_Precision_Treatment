package com.ricedisease.detector.i18n

import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat

/**
 * App-wide language manager using AppCompat per-app locales.
 * Supported languages: English, Hindi, Kannada.
 * Language persists automatically across sessions via AppCompat.
 */
object LocaleManager {
    enum class AppLanguage(val code: String) {
        ENGLISH("en"),
        HINDI  ("hi"),
        KANNADA("kn"),
    }

    fun applyLanguage(language: AppLanguage) {
        AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(language.code))
    }

    fun getCurrentLanguage(): AppLanguage {
        val tags = AppCompatDelegate.getApplicationLocales().toLanguageTags()
        return when {
            tags.startsWith("hi") -> AppLanguage.HINDI
            tags.startsWith("kn") -> AppLanguage.KANNADA
            else                  -> AppLanguage.ENGLISH
        }
    }
}
