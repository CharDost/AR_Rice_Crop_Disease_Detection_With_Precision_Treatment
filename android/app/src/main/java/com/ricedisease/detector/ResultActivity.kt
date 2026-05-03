package com.ricedisease.detector

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import coil.load
import com.ricedisease.detector.data.TreatmentRepository
import com.ricedisease.detector.databinding.ActivityResultBinding

class ResultActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "ResultActivity"
    }

    private lateinit var binding: ActivityResultBinding
    private var treatmentRepository: TreatmentRepository? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            binding = ActivityResultBinding.inflate(layoutInflater)
            setContentView(binding.root)

            supportActionBar?.setDisplayHomeAsUpEnabled(true)
            supportActionBar?.title = getString(R.string.diagnosis_label)

            try { treatmentRepository = TreatmentRepository(this) }
            catch (e: Exception) { Log.e(TAG, "TreatmentRepository init failed: ${e.message}") }

            displayResults()
            setupClickListeners()
        } catch (e: Exception) {
            Log.e(TAG, "onCreate error: ${e.message}", e)
            Toast.makeText(this, "Error displaying results", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun displayResults() {
        val predictedClass = intent.getStringExtra(MainActivity.EXTRA_RESULT_CLASS) ?: "Unknown"
        val confidence     = intent.getFloatExtra(MainActivity.EXTRA_RESULT_CONFIDENCE, 0f)
        val inferenceTime  = intent.getLongExtra(MainActivity.EXTRA_RESULT_INFERENCE_TIME, 0L)
        val imageUriStr    = intent.getStringExtra(MainActivity.EXTRA_IMAGE_URI)
        val isValidInput   = intent.getBooleanExtra("is_valid_input", true)
        val isConfident    = intent.getBooleanExtra("is_confident", true)

        Log.d(TAG, "Result: $predictedClass conf=$confidence valid=$isValidInput confident=$isConfident")

        // ── Image preview ──────────────────────────────────────────
        if (!imageUriStr.isNullOrEmpty()) {
            try {
                binding.imagePreview.load(Uri.parse(imageUriStr)) {
                    crossfade(true)
                    listener(onError = { _, _ ->
                        binding.imagePreview.visibility = View.GONE
                    })
                }
                binding.imagePreview.visibility = View.VISIBLE
            } catch (e: Exception) {
                binding.imagePreview.visibility = View.GONE
            }
        }

        // ── Confidence bar ─────────────────────────────────────────
        binding.tvPredictedClass.text = predictedClass
        binding.tvConfidence.text = getString(R.string.format_confidence, confidence * 100)
        binding.tvInferenceTime.text  = getString(R.string.format_inference, inferenceTime)

        val confColor = when {
            !isValidInput       -> Color.parseColor("#9E9E9E")
            confidence >= 0.95f -> Color.parseColor("#4CAF50")
            confidence >= 0.85f -> Color.parseColor("#8BC34A")
            confidence >= 0.75f -> Color.parseColor("#FFC107")
            else                -> Color.parseColor("#FF9800")
        }
        binding.confidenceIndicator.setBackgroundColor(confColor)

        // ── Low confidence / invalid quality warning ───────────────
        val warningMsg = when {
            !isValidInput -> getString(R.string.warning_not_rice_leaf)
            !isConfident  -> getString(R.string.warning_low_confidence)
            else          -> null
        }
        if (warningMsg != null) {
            binding.cardWarning.visibility = View.VISIBLE
            binding.tvWarning.text = warningMsg
        } else {
            binding.cardWarning.visibility = View.GONE
        }

        // ── Main content ───────────────────────────────────────────
        when {
            !isValidInput || predictedClass.equals("Not a Rice Leaf", ignoreCase = true) ->
                showRejectedResult()
            predictedClass.equals("Healthy", ignoreCase = true) ->
                showHealthyResult()
            else ->
                showDiseaseResult(predictedClass)
        }
    }

    private fun showRejectedResult() {
        binding.cardTreatment.visibility = View.GONE
        binding.cardHealthy.visibility   = View.GONE
        binding.cardWarning.visibility   = View.VISIBLE
        binding.tvWarning.text           = getString(R.string.warning_not_rice_leaf)
    }

    private fun showHealthyResult() {
        binding.cardTreatment.visibility = View.GONE
        binding.cardHealthy.visibility   = View.VISIBLE
        binding.tvHealthyMessage.text    = getString(R.string.healthy_recommendation)
    }

    private fun showDiseaseResult(diseaseName: String) {
        binding.cardHealthy.visibility   = View.GONE
        binding.cardTreatment.visibility = View.VISIBLE

        val info = treatmentRepository?.getTreatment(diseaseName)

        if (info != null) {
            binding.tvDiseaseName.text     = info.displayName
            binding.tvScientificName.text  = info.scientificName

            binding.tvSeverity.text = info.severity.uppercase()
            binding.tvSeverity.setTextColor(Color.WHITE)
            binding.cardSeverity.setCardBackgroundColor(Color.parseColor(info.getSeverityColorHex()))

            binding.tvSymptoms.text          = info.symptoms.joinToString("\n")    { "• $it" }
            binding.tvTreatmentImmediate.text = info.treatment.immediate.joinToString("\n") { "• $it" }
            binding.tvTreatmentChemical.text  = info.treatment.chemical.joinToString("\n")  { "• $it" }
            binding.tvTreatmentCultural.text  = info.treatment.cultural.joinToString("\n")  { "• $it" }
            binding.tvPrevention.text         = info.prevention.joinToString("\n")  { "• $it" }
        } else {
            binding.tvDiseaseName.text     = diseaseName
            binding.tvScientificName.text  = "Treatment data unavailable"
            binding.layoutTreatmentDetails.visibility = View.GONE
        }
    }

    private fun setupClickListeners() {
        binding.btnScanAnother.setOnClickListener { finish() }
        binding.btnShare.setOnClickListener { shareResults() }
    }

    private fun shareResults() {
        val predictedClass = intent.getStringExtra(MainActivity.EXTRA_RESULT_CLASS) ?: "Unknown"
        val confidence     = intent.getFloatExtra(MainActivity.EXTRA_RESULT_CONFIDENCE, 0f)

        val shareText = """
            ${getString(R.string.share_header)}

            ${getString(R.string.share_diagnosis, predictedClass)}
            ${getString(R.string.share_confidence, "%.1f".format(confidence * 100))}

            ${getString(R.string.share_footer)}
        """.trimIndent()

        try {
            startActivity(
                Intent.createChooser(
                    Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        putExtra(Intent.EXTRA_TEXT, shareText)
                    },
                    getString(R.string.share_title)
                )
            )
        } catch (e: Exception) {
            Toast.makeText(this, "Could not share result", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onSupportNavigateUp(): Boolean { finish(); return true }
}
