package com.ricedisease.detector.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity representing a single scan result stored in history.
 */
@Entity(tableName = "scan_records")
data class ScanRecord(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,

    /** Predicted disease display name (e.g. "Bacterial Blight") */
    val predictedClass: String,

    /** Confidence score 0..1 */
    val confidence: Float,

    /** Whether the model accepted this as a valid rice leaf */
    val isValidInput: Boolean,

    /** true = healthy, false = diseased or rejected */
    val isHealthy: Boolean,

    /** Unix epoch timestamp in milliseconds */
    val timestamp: Long = System.currentTimeMillis(),

    /** Absolute file path to compressed thumbnail (or null if not saved) */
    val imagePath: String? = null,

    /** Inference time in milliseconds */
    val inferenceTimeMs: Long = 0L,

    /** ISO 639-1 language code active at scan time, e.g. "en", "hi", "kn" */
    val language: String = "en",
)
