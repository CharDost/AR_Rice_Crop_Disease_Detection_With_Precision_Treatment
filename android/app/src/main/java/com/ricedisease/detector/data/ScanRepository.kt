package com.ricedisease.detector.data

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

/**
 * Repository that wraps ScanDao and handles image compression / persistence.
 */
class ScanRepository(context: Context) {

    companion object {
        private const val TAG = "ScanRepository"
        private const val THUMB_QUALITY = 80
        private const val THUMB_SIZE    = 224
    }

    private val dao  = AppDatabase.getInstance(context).scanDao()
    private val dir  = File(context.filesDir, "scan_thumbs").also { it.mkdirs() }

    // --- Flow queries (observe from UI) ---
    fun getAllScans(): Flow<List<ScanRecord>>     = dao.getAllScans()
    fun getHealthyScans(): Flow<List<ScanRecord>> = dao.getHealthyScans()
    fun getDiseasedScans(): Flow<List<ScanRecord>> = dao.getDiseasedScans()
    fun getScansByDateRange(startMs: Long, endMs: Long): Flow<List<ScanRecord>> =
        dao.getScansByDateRange(startMs, endMs)
    fun getScansByDisease(disease: String): Flow<List<ScanRecord>> =
        dao.getScansByDisease("%$disease%")

    // --- Suspend queries ---
    suspend fun getTotalCount(): Int   = dao.getTotalCount()
    suspend fun getHealthyCount(): Int = dao.getHealthyCount()
    suspend fun getDiseasedCount(): Int= dao.getDiseasedCount()
    suspend fun getRecentScans(n: Int = 30): List<ScanRecord> = dao.getRecentScans(n)

    /**
     * Save a scan. Optionally compresses and saves the bitmap thumbnail.
     */
    suspend fun saveScan(
        predictedClass: String,
        confidence: Float,
        isValidInput: Boolean,
        isHealthy: Boolean,
        inferenceTimeMs: Long,
        bitmap: Bitmap? = null,
    ): Long = withContext(Dispatchers.IO) {
        val imagePath = bitmap?.let { saveThumbnail(it) }
        val record = ScanRecord(
            predictedClass  = predictedClass,
            confidence      = confidence,
            isValidInput    = isValidInput,
            isHealthy       = isHealthy,
            timestamp       = System.currentTimeMillis(),
            imagePath       = imagePath,
            inferenceTimeMs = inferenceTimeMs,
            language        = Locale.getDefault().language,
        )
        dao.insert(record).also { Log.d(TAG, "Saved scan id=$it class=$predictedClass") }
    }

    suspend fun deleteScan(record: ScanRecord) = withContext(Dispatchers.IO) {
        record.imagePath?.let { File(it).delete() }
        dao.delete(record)
    }

    suspend fun deleteAll() = withContext(Dispatchers.IO) {
        dir.listFiles()?.forEach { it.delete() }
        dao.deleteAll()
    }

    private fun saveThumbnail(bitmap: Bitmap): String? = try {
        val scaled = Bitmap.createScaledBitmap(bitmap, THUMB_SIZE, THUMB_SIZE, true)
        val file = File(dir, "scan_${System.currentTimeMillis()}.jpg")
        FileOutputStream(file).use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, THUMB_QUALITY, out)
        }
        file.absolutePath
    } catch (e: Exception) {
        Log.e(TAG, "Thumbnail save failed: ${e.message}")
        null
    }
}
