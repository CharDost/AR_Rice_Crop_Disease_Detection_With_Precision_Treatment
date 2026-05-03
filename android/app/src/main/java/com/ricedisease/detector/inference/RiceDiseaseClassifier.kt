package com.ricedisease.detector.inference

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln

/**
 * Rice Disease Classifier using TensorFlow Lite
 * 
 * Includes validation to reject non-rice-leaf images
 */
class RiceDiseaseClassifier(private val context: Context) {

    companion object {
        private const val TAG = "RiceDiseaseClassifier"
        private const val MODEL_FILE = "rice_disease_model.tflite"
        private const val FALLBACK_MODEL_FILE = "rice_disease_model_legacy.tflite"
        private const val INPUT_SIZE = 224
        private const val PIXEL_SIZE = 3  // RGB
        private const val NUM_CLASSES = 6   // 5 diseases + background

        // Model v5 thresholds (aligned with calibration_config_v5.json balanced profile)
        const val THRESHOLD_CONSERVATIVE = 0.96f
        const val THRESHOLD_BALANCED     = 0.94f   // balanced profile v5
        const val THRESHOLD_PERMISSIVE   = 0.90f

        // Background class index — if model predicts this, it IS the rejection signal
        private const val BACKGROUND_CLASS_IDX = 5

        // ENTROPY NOTE: Python calibration uses NORMALIZED entropy (H / ln(6)).
        // Android calculateEntropy() computes RAW Shannon entropy H = -sum(p*ln(p)).
        // ln(6) = 1.7918. Conversion: android_raw = python_normalized * ln(6)
        // Keep these in sync with calibration_config_v5.json.
        private const val MAX_ENTROPY_CONSERVATIVE = 0.448f   // normalized 0.25 * ln(6)
        private const val MAX_ENTROPY_BALANCED     = 0.627f   // normalized 0.35 * ln(6) — DEFAULT v5
        private const val MAX_ENTROPY_PERMISSIVE   = 0.896f   // normalized 0.50 * ln(6)
        private const val MAX_ENTROPY_FOR_VALID    = MAX_ENTROPY_BALANCED

        // Leaf detection thresholds (secondary safety net for completely non-leaf inputs)
        private const val MIN_CONFIDENCE_FOR_VALID  = 0.50f  // relaxed — model's bg class handles rejection
        private const val MIN_GREEN_RATIO            = 0.08f
        private const val MIN_CONFIDENCE_MARGIN      = 0.08f
        private const val MIN_TEXTURE_VARIANCE       = 8f
        private const val MIN_SHARPNESS              = 15f
        private const val CENTER_CROP_RATIO          = 0.78f
    }

    private var interpreter: Interpreter? = null
    // Display labels: 6th class shown as "Not a Rice Leaf" for UX clarity
    private val labels = listOf("Bacterial Blight", "Blast", "Brown Spot", "Healthy", "Hispa", "Not a Rice Leaf")
    private var confidenceThreshold: Float = THRESHOLD_BALANCED
    private var activeModelFile: String = MODEL_FILE

    init {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            val candidates = listOf(MODEL_FILE, FALLBACK_MODEL_FILE)
            var lastError: Exception? = null

            for (candidate in candidates) {
                try {
                    val modelBuffer = loadModelFile(candidate)
                    interpreter = Interpreter(modelBuffer, options)
                    activeModelFile = candidate
                    Log.d(TAG, "Model loaded successfully from $candidate. Labels: $labels")
                    break
                } catch (e: Exception) {
                    lastError = e
                    Log.w(TAG, "Failed to load model candidate $candidate: ${e.message}")
                }
            }

            if (interpreter == null) {
                throw RuntimeException("All model candidates failed. Last error: ${lastError?.message}", lastError)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            throw RuntimeException("Failed to initialize classifier: ${e.message}", e)
        }
    }

    private fun loadModelFile(assetName: String): ByteBuffer {
        // Use stream-based loading so model loading works even if asset packaging changes.
        context.assets.open(assetName).use { input ->
            val modelBytes = input.readBytes()
            return ByteBuffer.allocateDirect(modelBytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(modelBytes)
                rewind()
            }
        }
    }

    fun setConfidenceThreshold(threshold: Float) {
        confidenceThreshold = threshold.coerceIn(0f, 1f)
    }

    /**
     * Classify a bitmap image with validation
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        val startTime = System.currentTimeMillis()

        // Focus detection on center ROI to reduce random background objects.
        val roiBitmap = cropCenterRegion(bitmap, CENTER_CROP_RATIO)

        // Image quality metrics (logged for diagnostics, NOT used for rejection)
        val greenRatio = calculateGreenRatio(roiBitmap)
        val sharpness = calculateSharpness(roiBitmap)

        // Preprocess and run inference
        val inputBuffer = preprocessImage(roiBitmap)
        val outputArray = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter?.run(inputBuffer, outputArray)

        val inferenceTime = System.currentTimeMillis() - startTime
        val probabilities = outputArray[0]

        // Calculate entropy (measure of uncertainty)
        val entropy = calculateEntropy(probabilities)

        // Get top predictions
        val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
        val topIdx = sortedIndices[0]
        val secondIdx = sortedIndices[1]
        val maxProb = probabilities[topIdx]
        val secondProb = probabilities[secondIdx]
        val confidenceMargin = maxProb - secondProb

        Log.d(TAG, "Green=$greenRatio, Sharp=$sharpness, Entropy=$entropy")
        Log.d(TAG, "Top: ${labels[topIdx]}=$maxProb, Second: ${labels[secondIdx]}=$secondProb, Margin=$confidenceMargin")

        // ---------------------------------------------------------------
        // V6 REJECTION LOGIC — Model-first, minimal fallback
        //
        // The 6-class model was trained with a dedicated Background class.
        // It IS the primary rejection mechanism. Heuristic gates (texture,
        // green ratio, sharpness) were removed because they incorrectly
        // rejected valid disease images:
        //   - Healthy leaves: smooth texture (variance ~5) → false reject
        //   - Hispa: white streaks, low green ratio → false reject
        //   - Bacterial Blight: yellow areas, low texture → false reject
        //
        // Rejection now fires ONLY when:
        //   1. Model predicts Background class (idx 5), OR
        //   2. Model confidence is extremely low (< 30%) — total garbage
        // ---------------------------------------------------------------
        val isModelBackground = (topIdx == BACKGROUND_CLASS_IDX) && (maxProb >= 0.40f)
        val isGarbagePrediction = maxProb < 0.30f

        val isValidInput = !isModelBackground && !isGarbagePrediction
        val predictedClass = if (isValidInput) labels[topIdx] else "Not a Rice Leaf"
        val isConfident = isValidInput && maxProb >= confidenceThreshold

        Log.d(TAG, "modelBG=$isModelBackground garbage=$isGarbagePrediction valid=$isValidInput confident=$isConfident")
        Log.d(TAG, "Final: $predictedClass (conf=${maxProb})")

        // For the probability map shown to users, exclude the background class
        val diseaseProbs = labels.zip(probabilities.toList())
            .filter { it.first != "Not a Rice Leaf" }
            .toMap()

        return ClassificationResult(
            predictedClass = predictedClass,
            confidence = if (isValidInput) maxProb else 0f,
            isConfident = isConfident,
            isValidInput = isValidInput,
            thresholdUsed = confidenceThreshold,
            allProbabilities = diseaseProbs,
            inferenceTimeMs = inferenceTime,
            entropy = entropy,
            greenRatio = greenRatio,
            sharpness = sharpness
        )
    }

    private fun cropCenterRegion(bitmap: Bitmap, ratio: Float): Bitmap {
        val clamped = ratio.coerceIn(0.5f, 1f)
        val cropW = (bitmap.width * clamped).toInt().coerceAtLeast(1)
        val cropH = (bitmap.height * clamped).toInt().coerceAtLeast(1)
        val left = ((bitmap.width - cropW) / 2).coerceAtLeast(0)
        val top = ((bitmap.height - cropH) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(bitmap, left, top, cropW, cropH)
    }

    /**
     * Calculate Shannon entropy of probability distribution
     * High entropy = uncertain/spread out predictions = likely invalid input
     */
    private fun calculateEntropy(probabilities: FloatArray): Float {
        var entropy = 0f
        for (p in probabilities) {
            if (p > 0.0001f) {  // Avoid log(0)
                entropy -= p * ln(p)
            }
        }
        return entropy
    }

    /**
     * Calculate ratio of green-ish pixels in the image
     * Rice leaves should have significant green content
     */
    /**
     * Calculate ratio of leaf-like pixels in the image.
     *
     * Accounts for the full spectrum of rice leaf appearances:
     * - Healthy leaves: pure green
     * - Blast / brown_spot: yellow-green with brown lesions
     * - Bacterial blight: large yellow/white water-soaked areas
     * - Hispa: whitish streaks on green background
     *
     * We count pixels that look like any part of a rice leaf (green, yellow-green,
     * yellow, or pale/diseased areas), NOT just pure green.
     */
    private fun calculateGreenRatio(bitmap: Bitmap): Float {
        val scaled = Bitmap.createScaledBitmap(bitmap, 100, 100, true)
        val pixels = IntArray(100 * 100)
        scaled.getPixels(pixels, 0, 100, 0, 0, 100, 100)

        var leafPixels = 0
        for (pixel in pixels) {
            val r = Color.red(pixel)
            val g = Color.green(pixel)
            val b = Color.blue(pixel)

            // Pure green (healthy leaves)
            val isGreen = g > 60 && r < 180 && b < 150 && (g - r) > 15 && (g - b) > 15

            // Yellow-green (early disease / stressed leaves)
            val isYellowGreen = g > 100 && r > 80 && b < 130 && g >= r * 0.7f

            // Yellow (bacterial blight water-soaked areas, severe stress)
            val isYellow = r > 120 && g > 100 && b < 100 && r > b && g > b

            // Pale/white diseased areas (hispa white streaks, late-stage blast)
            // Must have low saturation AND some green channel dominance
            val isPale = r > 150 && g > 150 && b > 140 && (g - b) > 5

            // Brown lesion areas (brown_spot characteristic lesions)
            val isBrownLesion = r > 80 && r > g && r > b && g > 50 && b < 100

            if (isGreen || isYellowGreen || isYellow || isPale || isBrownLesion) {
                leafPixels++
            }
        }

        return leafPixels.toFloat() / pixels.size
    }

    /**
     * Calculate texture variance using local pixel differences
     * Natural leaves have organic, varied textures unlike solid/artificial surfaces
     */
    private fun calculateTextureVariance(bitmap: Bitmap): Float {
        val scaled = Bitmap.createScaledBitmap(bitmap, 50, 50, true)
        val pixels = IntArray(50 * 50)
        scaled.getPixels(pixels, 0, 50, 0, 0, 50, 50)
        
        var totalVariance = 0f
        var count = 0
        
        // Calculate local variance by comparing adjacent pixels
        for (y in 1 until 49) {
            for (x in 1 until 49) {
                val idx = y * 50 + x
                val current = pixels[idx]
                val right = pixels[idx + 1]
                val below = pixels[idx + 50]
                
                // Calculate brightness difference
                val currBrightness = (Color.red(current) + Color.green(current) + Color.blue(current)) / 3f
                val rightBrightness = (Color.red(right) + Color.green(right) + Color.blue(right)) / 3f
                val belowBrightness = (Color.red(below) + Color.green(below) + Color.blue(below)) / 3f
                
                totalVariance += kotlin.math.abs(currBrightness - rightBrightness)
                totalVariance += kotlin.math.abs(currBrightness - belowBrightness)
                count += 2
            }
        }
        
        return if (count > 0) totalVariance / count else 0f
    }

    /**
     * Estimate sharpness with gradient magnitude variance.
     */
    private fun calculateSharpness(bitmap: Bitmap): Float {
        val scaled = Bitmap.createScaledBitmap(bitmap, 64, 64, true)
        val pixels = IntArray(64 * 64)
        scaled.getPixels(pixels, 0, 64, 0, 0, 64, 64)

        var sum = 0f
        var sumSq = 0f
        var count = 0

        for (y in 1 until 63) {
            for (x in 1 until 63) {
                val idx = y * 64 + x
                val left = pixels[idx - 1]
                val right = pixels[idx + 1]
                val up = pixels[idx - 64]
                val down = pixels[idx + 64]

                val gx = (((Color.red(right) + Color.green(right) + Color.blue(right))
                    - (Color.red(left) + Color.green(left) + Color.blue(left))) / 3f)
                val gy = (((Color.red(down) + Color.green(down) + Color.blue(down))
                    - (Color.red(up) + Color.green(up) + Color.blue(up))) / 3f)

                val grad = kotlin.math.abs(gx) + kotlin.math.abs(gy)
                sum += grad
                sumSq += grad * grad
                count++
            }
        }

        if (count == 0) return 0f
        val mean = sum / count
        return (sumSq / count) - (mean * mean)
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val bufferSize = 1 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE * 4
        val buffer = ByteBuffer.allocateDirect(bufferSize).apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        // EfficientNetB0 includes its own internal preprocessing layer
        // (Rescaling + Normalization). Feed RAW 0-255 float values.
        for (pixel in pixels) {
            buffer.putFloat((pixel shr 16 and 0xFF).toFloat())  // R
            buffer.putFloat((pixel shr 8  and 0xFF).toFloat())  // G
            buffer.putFloat((pixel        and 0xFF).toFloat())  // B
        }

        buffer.rewind()
        return buffer
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * Data class for classification results
 */
data class ClassificationResult(
    val predictedClass: String,
    val confidence: Float,
    val isConfident: Boolean,
    val isValidInput: Boolean,
    val thresholdUsed: Float,
    val allProbabilities: Map<String, Float>,
    val inferenceTimeMs: Long,
    val entropy: Float = 0f,
    val greenRatio: Float = 0f,
    val sharpness: Float = 0f
) {
    fun confidencePercent(): String = "%.1f%%".format(confidence * 100)
    
    fun getConfidenceLevel(): String = when {
        !isValidInput -> "Invalid"
        confidence >= 0.95f -> "Very High"
        confidence >= 0.85f -> "High"
        confidence >= 0.80f -> "Good"
        confidence >= 0.70f -> "Moderate"
        else -> "Low"
    }
    
    fun getValidationMessage(): String = when {
        !isValidInput -> "This doesn't appear to be a rice leaf. Please capture a clear image of a rice leaf."
        !isConfident -> "Detection confidence is low. Try capturing a clearer image."
        else -> ""
    }
}
