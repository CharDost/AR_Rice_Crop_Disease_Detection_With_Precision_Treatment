package com.ricedisease.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.ricedisease.detector.inference.ClassificationResult

/**
 * Custom view that draws detection results overlaid on the camera preview.
 * This creates the AR-like effect of showing disease information in real-time.
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var result: ClassificationResult? = null
    
    // Detection box paint
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }
    
    // Corner brackets paint
    private val cornerPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 10f
        isAntiAlias = true
    }
    
    // Background for text
    private val textBackgroundPaint = Paint().apply {
        color = Color.parseColor("#DD000000")
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Main label text
    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 56f
        isFakeBoldText = true
        isAntiAlias = true
    }
    
    // Confidence text
    private val confidencePaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
    }
    
    // Status indicator paint
    private val statusPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    // Scanning animation
    private val scanLinePaint = Paint().apply {
        color = Color.parseColor("#8000FF00")
        strokeWidth = 4f
        isAntiAlias = true
    }
    
    private var scanLinePosition = 0f
    private var scanDirection = 1
    private val scanningText by lazy { context.getString(R.string.scan_instruction) }
    private val warningText by lazy { context.getString(R.string.not_rice_leaf) }
    private val warningHintText by lazy { context.getString(R.string.not_rice_leaf_hint) }
    
    fun setResult(result: ClassificationResult) {
        this.result = result
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        // Draw targeting frame
        drawTargetFrame(canvas)
        
        result?.let { res ->
            if (res.isValidInput) {
                drawValidDetection(canvas, res)
            } else {
                drawInvalidInputState(canvas)
            }
        } ?: drawScanningState(canvas)
    }
    
    private fun drawTargetFrame(canvas: Canvas) {
        val padding = 60f
        val cornerLength = 80f
        val rect = RectF(padding, padding * 2, width - padding, height - padding * 3)
        
        // Set color based on detection state
        val frameColor = when {
            result?.isValidInput == true && result?.predictedClass == "Healthy" -> 
                Color.parseColor("#4CAF50") // Green
            result?.isValidInput == true -> 
                Color.parseColor("#FF5722") // Orange/Red for disease
            result?.isValidInput == false ->
                Color.parseColor("#FFC107") // Yellow when not a rice leaf
            else -> 
                Color.parseColor("#FFFFFF") // White when scanning (no result yet)
        }
        
        cornerPaint.color = frameColor
        
        // Draw corner brackets (AR-style targeting)
        // Top-left corner
        canvas.drawLine(rect.left, rect.top, rect.left + cornerLength, rect.top, cornerPaint)
        canvas.drawLine(rect.left, rect.top, rect.left, rect.top + cornerLength, cornerPaint)
        
        // Top-right corner
        canvas.drawLine(rect.right - cornerLength, rect.top, rect.right, rect.top, cornerPaint)
        canvas.drawLine(rect.right, rect.top, rect.right, rect.top + cornerLength, cornerPaint)
        
        // Bottom-left corner
        canvas.drawLine(rect.left, rect.bottom, rect.left + cornerLength, rect.bottom, cornerPaint)
        canvas.drawLine(rect.left, rect.bottom - cornerLength, rect.left, rect.bottom, cornerPaint)
        
        // Bottom-right corner
        canvas.drawLine(rect.right - cornerLength, rect.bottom, rect.right, rect.bottom, cornerPaint)
        canvas.drawLine(rect.right, rect.bottom - cornerLength, rect.right, rect.bottom, cornerPaint)
        
        // Draw subtle dashed border
        boxPaint.color = Color.parseColor("#44FFFFFF")
        canvas.drawRect(rect, boxPaint)
    }
    
    private fun drawValidDetection(canvas: Canvas, result: ClassificationResult) {
        val padding = 60f
        val rect = RectF(padding, padding * 2, width - padding, height - padding * 3)
        
        // Draw label at top of frame
        val label = result.predictedClass
        val confidence = result.confidencePercent()
        
        // Calculate text dimensions
        val labelWidth = labelPaint.measureText(label)
        val confidenceWidth = confidencePaint.measureText(confidence)
        val maxTextWidth = maxOf(labelWidth, confidenceWidth)
        
        val textPadding = 20f
        val labelBgRect = RectF(
            rect.left,
            rect.top - 120f,
            rect.left + maxTextWidth + textPadding * 2,
            rect.top - 10f
        )
        
        // Draw background
        canvas.drawRoundRect(labelBgRect, 12f, 12f, textBackgroundPaint)
        
        // Draw status indicator
        val statusColor = if (result.predictedClass == "Healthy") 
            Color.parseColor("#4CAF50") else Color.parseColor("#FF5722")
        statusPaint.color = statusColor
        canvas.drawCircle(
            labelBgRect.left + textPadding + 12f,
            labelBgRect.top + (labelBgRect.height() / 2),
            10f,
            statusPaint
        )
        
        // Draw label text
        canvas.drawText(
            label,
            labelBgRect.left + textPadding + 32f,
            labelBgRect.top + 50f,
            labelPaint
        )
        
        // Draw confidence
        canvas.drawText(
            confidence,
            labelBgRect.left + textPadding + 32f,
            labelBgRect.top + 90f,
            confidencePaint
        )
        
        // Draw "DETECTED" badge if disease found
        if (result.predictedClass != "Healthy") {
            val badgeText = "DISEASE DETECTED"
            val badgeWidth = confidencePaint.measureText(badgeText)
            val badgeRect = RectF(
                rect.right - badgeWidth - textPadding * 2,
                rect.bottom + 20f,
                rect.right,
                rect.bottom + 70f
            )
            
            val badgePaint = Paint().apply {
                color = Color.parseColor("#FF5722")
                style = Paint.Style.FILL
            }
            canvas.drawRoundRect(badgeRect, 8f, 8f, badgePaint)
            
            canvas.drawText(
                badgeText,
                badgeRect.left + textPadding,
                badgeRect.top + 35f,
                Paint().apply {
                    color = Color.WHITE
                    textSize = 28f
                    isFakeBoldText = true
                }
            )
        }
    }
    
    private fun drawScanningState(canvas: Canvas) {
        val padding = 60f
        val rect = RectF(padding, padding * 2, width - padding, height - padding * 3)
        
        // Draw scanning animation line
        scanLinePosition += scanDirection * 8
        if (scanLinePosition > rect.height() || scanLinePosition < 0) {
            scanDirection *= -1
        }
        
        canvas.drawLine(
            rect.left + 10,
            rect.top + scanLinePosition,
            rect.right - 10,
            rect.top + scanLinePosition,
            scanLinePaint
        )
        
        // Draw "Scanning..." text
        val scanText = scanningText
        val textWidth = confidencePaint.measureText(scanText)
        
        val textBgRect = RectF(
            (width - textWidth) / 2 - 20f,
            rect.top - 80f,
            (width + textWidth) / 2 + 20f,
            rect.top - 20f
        )
        
        canvas.drawRoundRect(textBgRect, 8f, 8f, textBackgroundPaint)
        canvas.drawText(
            scanText,
            (width - textWidth) / 2,
            rect.top - 40f,
            confidencePaint
        )
        
        // Request redraw for animation
        postInvalidateDelayed(50)
    }
    
    /**
     * Draw state when input is detected but not a valid rice leaf
     */
    private fun drawInvalidInputState(canvas: Canvas) {
        val padding = 60f
        val rect = RectF(padding, padding * 2, width - padding, height - padding * 3)
        
        // Draw "Not a Rice Leaf" warning
        val hintText = warningHintText
        
        val warningWidth = labelPaint.measureText(warningText)
        val hintWidth = confidencePaint.measureText(hintText)
        val maxWidth = maxOf(warningWidth, hintWidth)
        
        val textPadding = 24f
        val labelBgRect = RectF(
            (width - maxWidth) / 2 - textPadding,
            rect.top - 110f,
            (width + maxWidth) / 2 + textPadding,
            rect.top - 10f
        )
        
        // Draw background with warning color tint
        val warningBgPaint = Paint().apply {
            color = Color.parseColor("#DD331100")
            style = Paint.Style.FILL
            isAntiAlias = true
        }
        canvas.drawRoundRect(labelBgRect, 12f, 12f, warningBgPaint)
        
        // Draw warning icon
        val warningIconPaint = Paint().apply {
            color = Color.parseColor("#FFC107")
            textSize = 48f
            isFakeBoldText = true
            isAntiAlias = true
        }
        canvas.drawText(
            "⚠",
            labelBgRect.left + textPadding,
            labelBgRect.top + 48f,
            warningIconPaint
        )
        
        // Draw warning text in yellow/orange
        val warningTextPaint = Paint().apply {
            color = Color.parseColor("#FFC107")
            textSize = 44f
            isFakeBoldText = true
            isAntiAlias = true
        }
        canvas.drawText(
            warningText,
            labelBgRect.left + textPadding + 50f,
            labelBgRect.top + 48f,
            warningTextPaint
        )
        
        // Draw hint text
        val hintPaint = Paint().apply {
            color = Color.parseColor("#AAAAAA")
            textSize = 28f
            isAntiAlias = true
        }
        canvas.drawText(
            hintText,
            (width - hintWidth) / 2,
            labelBgRect.top + 85f,
            hintPaint
        )
    }
}
