package com.ricedisease.detector

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import com.google.android.material.bottomsheet.BottomSheetDialogFragment
import com.ricedisease.detector.data.TreatmentRepository

/**
 * Bottom sheet dialog to display treatment information for detected diseases.
 * Provides a quick, dismissible view of treatment recommendations.
 */
class TreatmentBottomSheet : BottomSheetDialogFragment() {

    companion object {
        private const val ARG_DISEASE = "disease"

        fun newInstance(disease: String): TreatmentBottomSheet {
            return TreatmentBottomSheet().apply {
                arguments = Bundle().apply {
                    putString(ARG_DISEASE, disease)
                }
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.bottom_sheet_treatment, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val disease = arguments?.getString(ARG_DISEASE) ?: return

        val repository = TreatmentRepository(requireContext())
        val diseaseInfo = repository.getTreatment(disease)

        view.findViewById<TextView>(R.id.tvSheetTitle).text = disease
        view.findViewById<TextView>(R.id.tvDescription).text = diseaseInfo?.let {
            "${it.scientificName}\n${getString(R.string.sheet_severity, it.severity)}"
        } ?: getString(R.string.sheet_no_info)
        view.findViewById<TextView>(R.id.tvSymptoms).text = diseaseInfo?.symptoms?.joinToString("\n• ", prefix = "• ") ?: ""
        view.findViewById<TextView>(R.id.tvChemicalControl).text = diseaseInfo?.treatment?.chemical?.joinToString("\n• ", prefix = "• ") ?: ""
        view.findViewById<TextView>(R.id.tvPreventiveMeasures).text = diseaseInfo?.prevention?.joinToString("\n• ", prefix = "• ") ?: ""

        view.findViewById<View>(R.id.btnClose).setOnClickListener {
            dismiss()
        }
    }
}
