package com.ricedisease.detector.data

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.IOException
import java.util.Locale

/**
 * Repository for accessing treatment recommendations.
 * Loads locale-aware treatment data:
 *   - treatments_hi.json  → Hindi
 *   - treatments_kn.json  → Kannada
 *   - treatments_te.json  → Telugu  (falls back to English if absent)
 *   - treatments.json     → English (default)
 */
class TreatmentRepository(context: Context) {

    companion object {
        private const val TAG = "TreatmentRepository"

        /** Map locale language tags to asset file names */
        private fun assetFileForLocale(languageTag: String): String = when {
            languageTag.startsWith("hi") -> "treatments_hi.json"
            languageTag.startsWith("kn") -> "treatments_kn.json"
            else -> "treatments.json"
        }
    }

    private val treatments: Map<String, DiseaseInfo>

    init {
        treatments = loadTreatments(context)
        Log.d(TAG, "Loaded ${treatments.size} treatments for locale: ${Locale.getDefault().toLanguageTag()}")
    }

    private fun loadTreatments(context: Context): Map<String, DiseaseInfo> {
        val locale = Locale.getDefault().toLanguageTag()
        val primaryFile = assetFileForLocale(locale)

        // Try locale-specific file first, fall back to English
        for (fileName in listOf(primaryFile, "treatments.json")) {
            try {
                val json = context.assets.open(fileName).bufferedReader().use { it.readText() }
                val result = parseJson(JSONObject(json))
                if (result.isNotEmpty()) {
                    Log.d(TAG, "Loaded treatments from $fileName")
                    return result
                }
            } catch (e: IOException) {
                Log.w(TAG, "Could not open $fileName: ${e.message}")
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing $fileName: ${e.message}")
            }
        }
        Log.e(TAG, "All treatment files failed to load")
        return emptyMap()
    }

    private fun parseJson(jsonObject: JSONObject): Map<String, DiseaseInfo> {
        val result = mutableMapOf<String, DiseaseInfo>()
        jsonObject.keys().forEach { key ->
            try {
                result[key] = DiseaseInfo.fromJson(key, jsonObject.getJSONObject(key))
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing disease $key: ${e.message}")
            }
        }
        return result
    }

    /**
     * Get treatment info for a disease.
     * @param diseaseName Display name e.g. "Bacterial Blight" or internal key "bacterial_blight"
     */
    fun getTreatment(diseaseName: String): DiseaseInfo? {
        val key = diseaseName.lowercase().replace(" ", "_")
        return treatments[key]
    }

    fun getAllTreatments(): List<DiseaseInfo> = treatments.values.toList()
}

/** Data class representing disease information and treatment */
data class DiseaseInfo(
    val id: String,
    val displayName: String,
    val scientificName: String,
    val severity: String,
    val symptoms: List<String>,
    val treatment: Treatment,
    val prevention: List<String>
) {
    companion object {
        fun fromJson(id: String, json: JSONObject): DiseaseInfo {
            val symptoms   = json.getJSONArray("symptoms").let  { a -> (0 until a.length()).map { a.getString(it) } }
            val prevention = json.getJSONArray("prevention").let { a -> (0 until a.length()).map { a.getString(it) } }
            return DiseaseInfo(
                id            = id,
                displayName   = json.getString("display_name"),
                scientificName= json.getString("scientific_name"),
                severity      = json.getString("severity"),
                symptoms      = symptoms,
                treatment     = Treatment.fromJson(json.getJSONObject("treatment")),
                prevention    = prevention,
            )
        }
    }

    fun getSeverityColorHex(): String = when (severity.lowercase()) {
        "high",   "अधिक", "ಅಧಿಕ", "అధికం" -> "#E53935"
        "medium", "मध्यम","ಮಧ್ಯಮ","మధ్యమం"  -> "#FB8C00"
        "low",    "कम",   "ಕಡಿಮೆ","తక్కువ"  -> "#43A047"
        "none",   "कोई नहीं","ಇಲ್ಲ","ఏదీ లేదు" -> "#4CAF50"
        else -> "#757575"
    }
}

/** Treatment recommendations */
data class Treatment(
    val immediate: List<String>,
    val chemical:  List<String>,
    val cultural:  List<String>
) {
    companion object {
        fun fromJson(json: JSONObject): Treatment = Treatment(
            immediate = json.getJSONArray("immediate").let { a -> (0 until a.length()).map { a.getString(it) } },
            chemical  = json.getJSONArray("chemical").let  { a -> (0 until a.length()).map { a.getString(it) } },
            cultural  = json.getJSONArray("cultural").let  { a -> (0 until a.length()).map { a.getString(it) } },
        )
    }
}
