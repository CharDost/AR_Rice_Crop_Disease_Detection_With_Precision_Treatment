package com.ricedisease.detector

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.appcompat.app.AlertDialog
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.ricedisease.detector.databinding.ActivityMainBinding
import com.ricedisease.detector.inference.RiceDiseaseClassifier
import com.ricedisease.detector.i18n.LocaleManager
import com.ricedisease.detector.data.ScanRepository
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var classifier: RiceDiseaseClassifier? = null
    private var modelInitError: String? = null
    private lateinit var scanRepo: ScanRepository
    private var lastBitmap: Bitmap? = null

    companion object {
        private const val TAG = "MainActivity"
        const val EXTRA_RESULT_CLASS = "result_class"
        const val EXTRA_RESULT_CONFIDENCE = "result_confidence"
        const val EXTRA_RESULT_INFERENCE_TIME = "result_inference_time"
        const val EXTRA_IMAGE_URI = "image_uri"
    }

    // Permission request launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }

    // Camera launcher
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap ->
        if (bitmap != null) {
            Log.d(TAG, "Camera captured image: ${bitmap.width}x${bitmap.height}")
            processImage(bitmap)
        } else {
            Log.d(TAG, "Camera returned null bitmap")
        }
    }

    // Gallery launcher
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            Log.d(TAG, "Gallery returned URI: $uri")
            loadAndProcessImage(uri)
        } else {
            Log.d(TAG, "Gallery returned null URI")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        scanRepo = ScanRepository(this)
        initializeClassifier()
        setupClickListeners()
    }

    private fun initializeClassifier() {
        try {
            classifier = RiceDiseaseClassifier(this)
            modelInitError = null
            Log.d(TAG, "Classifier initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize classifier: ${e.message}", e)
            e.printStackTrace()
            modelInitError = e.message
            Toast.makeText(
                this,
                "Failed to load AI model: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun setupClickListeners() {
        binding.btnLanguage.setOnClickListener {
            showLanguageSelector()
        }

        binding.btnCamera.setOnClickListener {
            checkCameraPermissionAndOpen()
        }

        binding.btnGallery.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        // Live Scan (AR Mode) button
        binding.btnLiveScan.setOnClickListener {
            startActivity(Intent(this, LiveScanActivity::class.java))
        }

        // Scan History button
        binding.btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        // Confidence threshold selector
        binding.chipGroupThreshold.setOnCheckedChangeListener { _, checkedId ->
            val threshold = when (checkedId) {
                R.id.chipConservative -> RiceDiseaseClassifier.THRESHOLD_CONSERVATIVE
                R.id.chipBalanced     -> RiceDiseaseClassifier.THRESHOLD_BALANCED
                R.id.chipPermissive   -> RiceDiseaseClassifier.THRESHOLD_PERMISSIVE
                else                  -> RiceDiseaseClassifier.THRESHOLD_BALANCED
            }
            classifier?.setConfidenceThreshold(threshold)
        }
    }

    private fun checkCameraPermissionAndOpen() {
        when {
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                openCamera()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                Toast.makeText(
                    this,
                    getString(R.string.permission_camera_rationale),
                    Toast.LENGTH_LONG
                ).show()
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun openCamera() {
        takePictureLauncher.launch(null)
    }

    private fun loadAndProcessImage(uri: Uri) {
        try {
            Log.d(TAG, "Loading image from URI: $uri")
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()
            
            if (bitmap != null) {
                Log.d(TAG, "Image loaded successfully: ${bitmap.width}x${bitmap.height}")
                processImage(bitmap, uri)
            } else {
                Log.e(TAG, "BitmapFactory.decodeStream returned null")
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading image: ${e.message}", e)
            e.printStackTrace()
            Toast.makeText(this, "Error loading image: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun processImage(bitmap: Bitmap, imageUri: Uri? = null) {
        lastBitmap = bitmap
        val localClassifier = classifier
        if (localClassifier == null) {
            val detail = modelInitError?.takeIf { it.isNotBlank() }
            Toast.makeText(this,
                if (detail != null) "${getString(R.string.error_model_load)}\n$detail"
                else getString(R.string.error_model_load),
                Toast.LENGTH_LONG).show()
            return
        }

        binding.btnCamera.isEnabled = false
        binding.btnGallery.isEnabled = false

        try {
            val result = localClassifier.classify(bitmap)
            Log.d(TAG, "Classification result: ${result.predictedClass} (${result.confidencePercent()})")

            // Auto-save to history
            lifecycleScope.launch {
                scanRepo.saveScan(
                    predictedClass  = result.predictedClass,
                    confidence      = result.confidence,
                    isValidInput    = result.isValidInput,
                    isHealthy       = result.predictedClass.equals("Healthy", ignoreCase = true),
                    inferenceTimeMs = result.inferenceTimeMs,
                    bitmap          = if (result.isValidInput) bitmap else null,
                )
            }

            val intent = Intent(this, ResultActivity::class.java).apply {
                putExtra(EXTRA_RESULT_CLASS, result.predictedClass)
                putExtra(EXTRA_RESULT_CONFIDENCE, result.confidence)
                putExtra(EXTRA_RESULT_INFERENCE_TIME, result.inferenceTimeMs)
                putExtra("is_valid_input", result.isValidInput)
                putExtra("is_confident", result.isConfident)
                if (imageUri != null) putExtra(EXTRA_IMAGE_URI, imageUri.toString())
            }
            startActivity(intent)

        } catch (e: Exception) {
            Log.e(TAG, "Classification error: ${e.message}", e)
            Toast.makeText(this, getString(R.string.error_classification), Toast.LENGTH_LONG).show()
        } finally {
            binding.btnCamera.isEnabled = true
            binding.btnGallery.isEnabled = true
        }
    }

    private fun showLanguageSelector() {
        val languages = LocaleManager.AppLanguage.entries
        val items = arrayOf(
            getString(R.string.language_english),
            getString(R.string.language_hindi),
            getString(R.string.language_kannada),
        )
        val selected = languages.indexOf(LocaleManager.getCurrentLanguage()).coerceAtLeast(0)

        AlertDialog.Builder(this)
            .setTitle(getString(R.string.select_language))
            .setSingleChoiceItems(items, selected) { dialog, which ->
                LocaleManager.applyLanguage(languages[which])
                dialog.dismiss()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
        classifier = null
    }
}
