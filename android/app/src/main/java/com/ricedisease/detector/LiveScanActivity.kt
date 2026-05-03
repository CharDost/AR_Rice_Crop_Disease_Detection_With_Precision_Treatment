package com.ricedisease.detector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.ricedisease.detector.databinding.ActivityLiveScanBinding
import com.ricedisease.detector.inference.ClassificationResult
import com.ricedisease.detector.inference.RiceDiseaseClassifier
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class LiveScanActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "LiveScanActivity"
        private const val INFERENCE_INTERVAL_MS = 500L  // Run inference every 500ms
        private const val STABILITY_WINDOW = 5
        private const val STABILITY_MIN_MATCH = 3
    }

    private lateinit var binding: ActivityLiveScanBinding
    private var classifier: RiceDiseaseClassifier? = null
    private lateinit var cameraExecutor: ExecutorService
    
    private var lastInferenceTime = 0L
    private var currentResult: ClassificationResult? = null
    private val recentValidPredictions = ArrayDeque<String>()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, getString(R.string.permission_camera_required), Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLiveScanBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        
        initializeClassifier()
        setupUI()
        checkCameraPermission()
    }

    private fun initializeClassifier() {
        try {
            classifier = RiceDiseaseClassifier(this)
            Log.d(TAG, "Classifier initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize classifier: ${e.message}", e)
            Toast.makeText(this, "${getString(R.string.error_model_load)}\n${e.message}", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    private fun setupUI() {
        binding.btnBack.setOnClickListener {
            finish()
        }
        
        binding.btnCapture.setOnClickListener {
            // Freeze current result and show details
            currentResult?.let { result ->
                if (result.isValidInput) {
                    showResultDetails(result)
                } else {
                    Toast.makeText(this, getString(R.string.scan_instruction), Toast.LENGTH_SHORT).show()
                }
            }
        }
        
        binding.btnTreatment.setOnClickListener {
            currentResult?.let { result ->
                if (result.isValidInput && result.predictedClass != "Healthy") {
                    showTreatmentPopup(result)
                }
            }
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            // Image analysis for continuous inference
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed: ${e.message}")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        
        // Throttle inference to avoid overwhelming the device
        if (currentTime - lastInferenceTime < INFERENCE_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        
        lastInferenceTime = currentTime
        
        try {
            val bitmap = imageProxy.toBitmap()
            val result = classifier?.classify(bitmap)
            
            result?.let {
                val stableResult = stabilizeResult(it)
                currentResult = stableResult
                runOnUiThread {
                    updateOverlay(it, stableResult)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Frame processing error: ${e.message}")
        } finally {
            imageProxy.close()
        }
    }

    private fun updateOverlay(rawResult: ClassificationResult, stableResult: ClassificationResult?) {
        val uiResult = stableResult ?: rawResult

        // Update the overlay view
        binding.overlayView.setResult(uiResult)
        binding.overlayView.invalidate()
        
        // Update bottom info panel
        if (stableResult != null && stableResult.isValidInput) {
            binding.tvDetectedDisease.text = stableResult.predictedClass
            binding.tvConfidenceValue.text = stableResult.confidencePercent()
            binding.tvInferenceValue.text = "${stableResult.inferenceTimeMs}ms"
            
            // Set color based on detection
            val statusColor = when (stableResult.predictedClass) {
                "Healthy" -> Color.parseColor("#4CAF50")
                else -> Color.parseColor("#FF5722")
            }
            binding.statusIndicator.setBackgroundColor(statusColor)
            
            // Show/hide treatment button
            binding.btnTreatment.visibility = if (stableResult.predictedClass != "Healthy") 
                View.VISIBLE else View.GONE
        } else if (rawResult.isValidInput) {
            // Wait for stability across frames before surfacing disease class.
            binding.tvDetectedDisease.text = getString(R.string.status_scanning)
            binding.tvConfidenceValue.text = rawResult.confidencePercent()
            binding.tvInferenceValue.text = "${rawResult.inferenceTimeMs}ms"
            binding.statusIndicator.setBackgroundColor(Color.parseColor("#03A9F4"))
            binding.btnTreatment.visibility = View.GONE
        } else {
            // Not a valid rice leaf - show warning
            binding.tvDetectedDisease.text = getString(R.string.not_rice_leaf)
            binding.tvConfidenceValue.text = getString(R.string.status_not_applicable)
            binding.tvInferenceValue.text = "${rawResult.inferenceTimeMs}ms"
            binding.statusIndicator.setBackgroundColor(Color.parseColor("#FFC107")) // Yellow/warning
            binding.btnTreatment.visibility = View.GONE
        }
    }

    private fun stabilizeResult(result: ClassificationResult): ClassificationResult? {
        if (!result.isValidInput || !result.isConfident) {
            recentValidPredictions.clear()
            return null
        }

        recentValidPredictions.addLast(result.predictedClass)
        while (recentValidPredictions.size > STABILITY_WINDOW) {
            recentValidPredictions.removeFirst()
        }

        val matchCount = recentValidPredictions.count { it == result.predictedClass }
        return if (matchCount >= STABILITY_MIN_MATCH) result else null
    }

    private fun showResultDetails(result: ClassificationResult) {
        // Navigate to result activity with the current detection
        val intent = android.content.Intent(this, ResultActivity::class.java).apply {
            putExtra(MainActivity.EXTRA_RESULT_CLASS, result.predictedClass)
            putExtra(MainActivity.EXTRA_RESULT_CONFIDENCE, result.confidence)
            putExtra(MainActivity.EXTRA_RESULT_INFERENCE_TIME, result.inferenceTimeMs)
        }
        startActivity(intent)
    }

    private fun showTreatmentPopup(result: ClassificationResult) {
        // Show treatment in a bottom sheet or dialog
        val treatmentDialog = TreatmentBottomSheet.newInstance(result.predictedClass)
        treatmentDialog.show(supportFragmentManager, "treatment")
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier?.close()
    }
}

/**
 * ImageProxy extension to convert to Bitmap
 */
@androidx.camera.core.ExperimentalGetImage
fun ImageProxy.toBitmap(): Bitmap {
    val image = this.image ?: throw IllegalStateException("Image is null")
    
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, 
        image.width, image.height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 90, out)
    val imageBytes = out.toByteArray()
    
    return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}
