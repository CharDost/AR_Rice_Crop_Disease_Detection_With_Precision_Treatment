package com.ricedisease.detector

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.load
import com.google.android.material.chip.ChipGroup
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.ricedisease.detector.data.ScanRecord
import com.ricedisease.detector.data.ScanRepository
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistoryActivity : AppCompatActivity() {

    private lateinit var repo: ScanRepository
    private lateinit var adapter: ScanHistoryAdapter

    // Views
    private lateinit var tvEmpty: TextView
    private lateinit var recycler: RecyclerView
    private lateinit var chipGroup: ChipGroup
    private lateinit var progressHealthy: LinearProgressIndicator
    private lateinit var tvHealthyStat: TextView
    private lateinit var tvDiseasedStat: TextView
    private lateinit var fabDeleteAll: FloatingActionButton

    private var currentJob: kotlinx.coroutines.Job? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        supportActionBar?.apply {
            title = getString(R.string.history_title)
            setDisplayHomeAsUpEnabled(true)
        }

        repo = ScanRepository(this)
        bindViews()
        setupRecycler()
        setupFilters()
        setupFab()
        observeFilter(FilterType.ALL)
    }

    private fun bindViews() {
        tvEmpty         = findViewById(R.id.tvHistoryEmpty)
        recycler        = findViewById(R.id.recyclerHistory)
        chipGroup       = findViewById(R.id.chipGroupFilter)
        progressHealthy = findViewById(R.id.progressHealthy)
        tvHealthyStat   = findViewById(R.id.tvHealthyStat)
        tvDiseasedStat  = findViewById(R.id.tvDiseasedStat)
        fabDeleteAll    = findViewById(R.id.fabDeleteAll)
    }

    private fun setupRecycler() {
        adapter = ScanHistoryAdapter { record ->
            confirmDelete(
                message = getString(R.string.history_delete_confirm),
                onConfirm = { lifecycleScope.launch { repo.deleteScan(record) } }
            )
        }
        recycler.apply {
            layoutManager = LinearLayoutManager(this@HistoryActivity)
            adapter = this@HistoryActivity.adapter
            isNestedScrollingEnabled = false   // parent NestedScrollView handles scrolling
        }
    }

    private fun setupFilters() {
        chipGroup.setOnCheckedStateChangeListener { _, ids ->
            val filter = when (ids.firstOrNull()) {
                R.id.chipFilterHealthy  -> FilterType.HEALTHY
                R.id.chipFilterDiseased -> FilterType.DISEASED
                else                    -> FilterType.ALL
            }
            observeFilter(filter)
        }
    }

    private fun setupFab() {
        fabDeleteAll.setOnClickListener {
            confirmDelete(
                message = getString(R.string.history_delete_all_confirm),
                onConfirm = {
                    lifecycleScope.launch {
                        repo.deleteAll()
                        refreshStats()
                    }
                }
            )
        }
    }

    private fun observeFilter(filter: FilterType) {
        currentJob?.cancel()
        currentJob = lifecycleScope.launch {
            val flow = when (filter) {
                FilterType.ALL      -> repo.getAllScans()
                FilterType.HEALTHY  -> repo.getHealthyScans()
                FilterType.DISEASED -> repo.getDiseasedScans()
            }
            flow.collectLatest { records ->
                adapter.submitList(records)
                val isEmpty = records.isEmpty()
                tvEmpty.visibility  = if (isEmpty) View.VISIBLE else View.GONE
                recycler.visibility = if (isEmpty) View.GONE    else View.VISIBLE
                refreshStats()
            }
        }
    }

    private fun refreshStats() {
        lifecycleScope.launch {
            val healthy  = repo.getHealthyCount()
            val diseased = repo.getDiseasedCount()
            val total    = healthy + diseased

            tvHealthyStat.text  = getString(R.string.history_healthy_scans, healthy)
            tvDiseasedStat.text = getString(R.string.history_diseased_scans, diseased)

            val pct = if (total > 0) (healthy * 100 / total) else 0
            progressHealthy.setProgressCompat(pct, true)
        }
    }

    private fun confirmDelete(message: String, onConfirm: () -> Unit) {
        AlertDialog.Builder(this)
            .setMessage(message)
            .setPositiveButton(getString(R.string.history_delete)) { _, _ -> onConfirm() }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    enum class FilterType { ALL, HEALTHY, DISEASED }
}

// ── RecyclerView Adapter ──────────────────────────────────────────────────────

class ScanHistoryAdapter(
    private val onDelete: (ScanRecord) -> Unit,
) : ListAdapter<ScanRecord, ScanHistoryAdapter.ViewHolder>(DIFF) {

    companion object {
        private val DIFF = object : DiffUtil.ItemCallback<ScanRecord>() {
            override fun areItemsTheSame(a: ScanRecord, b: ScanRecord) = a.id == b.id
            override fun areContentsTheSame(a: ScanRecord, b: ScanRecord) = a == b
        }
        private val DATE_FMT = SimpleDateFormat("dd MMM yyyy  HH:mm", Locale.getDefault())
    }

    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val thumb:     ImageView = view.findViewById(R.id.ivScanThumb)
        val tvClass:   TextView  = view.findViewById(R.id.tvScanClass)
        val tvConf:    TextView  = view.findViewById(R.id.tvScanConfidence)
        val tvDate:    TextView  = view.findViewById(R.id.tvScanDate)
        val tvDelete:  TextView  = view.findViewById(R.id.tvDeleteScan)
        val statusDot: View      = view.findViewById(R.id.viewStatusDot)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder =
        ViewHolder(
            LayoutInflater.from(parent.context)
                .inflate(R.layout.item_scan_history, parent, false)
        )

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val r = getItem(position)

        holder.tvClass.text = r.predictedClass
        holder.tvConf.text  = "%.1f%%".format(r.confidence * 100)
        holder.tvDate.text  = DATE_FMT.format(Date(r.timestamp))

        // Status dot colour
        holder.statusDot.setBackgroundColor(
            when {
                !r.isValidInput -> 0xFF9E9E9E.toInt()
                r.isHealthy     -> 0xFF4CAF50.toInt()
                else            -> 0xFFE53935.toInt()
            }
        )

        // Thumbnail
        val imgFile = r.imagePath?.let { File(it) }
        if (imgFile != null && imgFile.exists()) {
            holder.thumb.load(imgFile) { crossfade(true) }
        } else {
            holder.thumb.setImageResource(R.drawable.ic_leaf_placeholder)
        }

        holder.tvDelete.setOnClickListener { onDelete(r) }
    }
}
