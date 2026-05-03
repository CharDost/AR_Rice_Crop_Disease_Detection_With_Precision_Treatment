"""
Production-Ready Rice Disease Inference Module

This module provides a robust, production-ready interface for rice disease
detection with confidence thresholding, input validation, and error handling.

Features:
- Confidence thresholding with configurable levels
- Input validation (size, format, quality checks)
- Structured prediction results
- Error handling and graceful degradation
- Support for both Keras and TFLite models
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image
import json
import math


class ConfidenceLevel(Enum):
    """Confidence threshold presets for different use cases"""
    CONSERVATIVE = 0.95  # Reject if confidence < 95% (high precision)
    BALANCED = 0.85      # Reject if confidence < 85% (recommended)
    PERMISSIVE = 0.80    # Reject if confidence < 80% (high coverage)
    NONE = 0.0           # Accept all predictions


@dataclass
class PredictionResult:
    """Structured prediction result with metadata"""
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    is_confident: bool
    threshold_used: float
    inference_time_ms: float
    is_rejected: bool = False
    rejection_reason: str = ""
    entropy: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'predicted_class': self.predicted_class,
            'confidence': float(self.confidence),
            'all_probabilities': {k: float(v) for k, v in self.all_probabilities.items()},
            'is_confident': self.is_confident,
            'threshold_used': float(self.threshold_used),
            'inference_time_ms': float(self.inference_time_ms),
            'is_rejected': self.is_rejected,
            'rejection_reason': self.rejection_reason,
            'entropy': float(self.entropy),
        }


@dataclass
class ValidationResult:
    """Result of input image validation"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class RiceDiseaseDetector:
    """
    Production-ready rice disease detection with confidence thresholding.
    
    Example:
        >>> detector = RiceDiseaseDetector(model_path="model.keras")
        >>> result = detector.predict(image, confidence_level=ConfidenceLevel.BALANCED)
        >>> if result.is_confident:
        ...     print(f"Disease: {result.predicted_class} ({result.confidence:.2%})")
        ... else:
        ...     print("Uncertain - please retake image")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        use_tflite: bool = False,
        enable_validation: bool = True,
        log_predictions: bool = False,
        calibration_config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the rice disease detector.
        
        Args:
            model_path: Path to Keras (.keras) or TFLite (.tflite) model
            use_tflite: Force TFLite interpreter (auto-detected from extension)
            enable_validation: Enable input image validation
            log_predictions: Log predictions for monitoring (production use)
        """
        self.model_path = Path(model_path)
        self.enable_validation = enable_validation
        self.log_predictions = log_predictions

        # Calibrated robustness defaults; may be overridden by config.
        self.temperature = 1.0
        self.entropy_threshold = 0.90
        self.default_confidence_threshold = ConfidenceLevel.BALANCED.value
        # Leaf gating: Use permissive thresholds that accept diseased leaves.
        # Rice disease lesions (blast, brown_spot, bacterial_blight, hispa) cause:
        #   - large non-green areas (yellow, brown, white lesions)
        #   - reduced green pixel ratio well below 12%
        # Setting too high here causes valid diseased leaves to be falsely rejected.
        self.min_green_ratio = 0.08          # Was 0.12 — blast/blight leaves can be <10% green
        self.min_texture_variance = 10.0     # Was 15.0 — slightly relaxed
        self.min_connected_green_ratio = 0.03  # Was 0.05 — smaller contiguous areas accepted

        # Model configuration
        self.img_size = (224, 224)
        self.classes = ['bacterial_blight', 'blast', 'brown_spot', 'healthy', 'hispa']
        self.class_display_names = {
            'bacterial_blight': 'Bacterial Blight',
            'blast': 'Blast',
            'brown_spot': 'Brown Spot',
            'healthy': 'Healthy',
            'hispa': 'Hispa'
        }
        
        # Auto-detect model type
        if not use_tflite:
            use_tflite = self.model_path.suffix.lower() == '.tflite'
        
        self.use_tflite = use_tflite
        
        # Load model
        self._load_model()

        # Load optional calibration configuration for reliable rejection.
        if calibration_config_path is not None:
            self._load_calibration_config(Path(calibration_config_path))
        
        # Statistics
        self.prediction_count = 0
        self.rejection_count = 0

    def _load_calibration_config(self, config_path: Path) -> None:
        """Load post-hoc calibration/rejection settings."""
        if not config_path.exists():
            raise FileNotFoundError(f"Calibration config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.temperature = float(payload.get("temperature", 1.0))
        rej = payload.get("rejection", {})
        self.default_confidence_threshold = float(
            rej.get("confidence_threshold", self.default_confidence_threshold)
        )
        self.entropy_threshold = float(rej.get("entropy_threshold", self.entropy_threshold))

        if self.log_predictions:
            print(
                "[CALIBRATION] loaded "
                f"T={self.temperature:.3f}, conf>={self.default_confidence_threshold:.2f}, "
                f"entropy<={self.entropy_threshold:.2f}"
            )

    @staticmethod
    def _apply_temperature_to_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling on probabilities via log-space transform."""
        temperature = max(1e-6, float(temperature))
        p = np.clip(probs, 1e-12, 1.0)
        logits_like = np.log(p) / temperature
        logits_like = logits_like - np.max(logits_like)
        exps = np.exp(logits_like)
        scaled = exps / np.sum(exps)
        return scaled.astype(np.float32)

    @staticmethod
    def _normalized_entropy(probs: np.ndarray) -> float:
        """Return H(p) / ln(N) ∈ [0,1].  Matches calibration_config_v3 entropy thresholds.
        NOTE: Android calculateEntropy() returns RAW H(p). To convert Android threshold to
        Python: python_thr = android_thr / ln(5).  See RiceDiseaseClassifier companion object."""
        p = np.clip(probs, 1e-12, 1.0)
        h = -float(np.sum(p * np.log(p)))
        return float(h / math.log(len(p)))

    def _leaf_likelihood(self, image: np.ndarray) -> Tuple[float, float, float]:
        """Return (green_ratio, texture_variance, connected_green_ratio) heuristics.
        
        Counts any pixel that could belong to a rice leaf — including:
        - Pure green (healthy)
        - Yellow-green (stressed/early disease)
        - Yellow (bacterial blight / severe stress)
        - Brown (brown_spot lesions)
        - Pale/white (hispa streaks, blast lesions)
        """
        img = image.astype(np.float32)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # Pure green pixels
        green_mask = (g > 60) & (r < 200) & (b < 170) & ((g - r) > 15) & ((g - b) > 15)
        # Yellow-green (early disease, stressed leaves)
        yellow_green_mask = (g > 100) & (r > 80) & (b < 140) & (g >= r * 0.7)
        # Yellow (bacterial blight water-soaked / severely stressed)
        yellow_mask = (r > 120) & (g > 100) & (b < 110) & (r > b) & (g > b)
        # Brown lesion pixels (brown_spot characteristic pattern)
        brown_mask = (r > 80) & (r > g) & (r > b) & (g > 50) & (b < 110)
        # Pale/whitish areas (hispa streaks, blast lesions) — low saturation but on leaf
        pale_mask = (r > 150) & (g > 150) & (b > 140) & ((g.astype(int) - b.astype(int)) > 3)

        leaf_mask = green_mask | yellow_green_mask | yellow_mask | brown_mask | pale_mask
        green_ratio = float(np.mean(leaf_mask))   # Use leaf_mask for ratio check

        # Largest connected leaf-like area helps reject random noisy pixels.
        leaf_u8 = (leaf_mask.astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_u8, connectivity=8)
        largest_area = 0
        for i in range(1, num_labels):
            largest_area = max(largest_area, int(stats[i, cv2.CC_STAT_AREA]))
        connected_green_ratio = float(largest_area / leaf_u8.size)

        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        texture_variance = float(np.var(gray))
        return green_ratio, texture_variance, connected_green_ratio
    
    def _load_model(self):
        """Load the model (Keras or TFLite)"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if self.use_tflite:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Verify input/output shapes
            expected_input_shape = [1, 224, 224, 3]
            actual_input_shape = self.input_details[0]['shape'].tolist()
            if actual_input_shape != expected_input_shape:
                raise ValueError(f"Model input shape mismatch: expected {expected_input_shape}, "
                               f"got {actual_input_shape}")
            
            self.model = None
            print(f"[OK] TFLite model loaded: {self.model_path}")
        else:
            # Load Keras model
            self.model = keras.models.load_model(str(self.model_path))
            self.interpreter = None
            print(f"[OK] Keras model loaded: {self.model_path}")
    
    def validate_image(self, image: np.ndarray) -> ValidationResult:
        """
        Validate input image quality and characteristics.
        
        Args:
            image: Input image (RGB, 0-255 range)
        
        Returns:
            ValidationResult with validation status and messages
        """
        if not self.enable_validation:
            return ValidationResult(is_valid=True)
        
        warnings = []
        
        # Check shape
        if len(image.shape) != 3:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid image dimensions: expected 3D array, got {len(image.shape)}D"
            )
        
        if image.shape[2] != 3:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid number of channels: expected 3 (RGB), got {image.shape[2]}"
            )
        
        # Check value range
        if image.max() <= 1.0:
            return ValidationResult(
                is_valid=False,
                error_message="Image appears to be normalized (values ≤1.0). Expected 0-255 range."
            )
        
        if image.min() < 0 or image.max() > 255:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid pixel values: range [{image.min():.2f}, {image.max():.2f}]. "
                             "Expected [0, 255]."
            )
        
        # Check for extreme darkness
        mean_brightness = np.mean(image)
        if mean_brightness < 30:
            warnings.append(f"Image is very dark (brightness: {mean_brightness:.1f}). "
                          "This may affect prediction accuracy.")
        
        # Check for extreme brightness
        if mean_brightness > 240:
            warnings.append(f"Image is very bright (brightness: {mean_brightness:.1f}). "
                          "This may affect prediction accuracy.")
        
        # Check for blur using Laplacian variance
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            warnings.append(f"Image appears blurry (sharpness: {laplacian_var:.1f}). "
                          "Consider retaking with better focus.")
        
        # Check contrast
        std_dev = np.std(image)
        if std_dev < 20:
            warnings.append(f"Low contrast detected (std: {std_dev:.1f}). "
                          "Image may lack detail.")

        # Leaf-likelihood checks to reject obvious non-rice/OOD inputs.
        green_ratio, texture_variance, connected_green_ratio = self._leaf_likelihood(image)
        if green_ratio < self.min_green_ratio:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Likely non-leaf image (green ratio {green_ratio:.2f} < {self.min_green_ratio:.2f})."
                ),
                warnings=warnings,
            )
        if connected_green_ratio < self.min_connected_green_ratio:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    "Likely non-leaf image (green pixels are too fragmented: "
                    f"{connected_green_ratio:.3f} < {self.min_connected_green_ratio:.3f})."
                ),
                warnings=warnings,
            )
        if texture_variance < self.min_texture_variance:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    f"Likely artificial/flat image (texture variance {texture_variance:.1f} < "
                    f"{self.min_texture_variance:.1f})."
                ),
                warnings=warnings,
            )
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
        
        Returns:
            Preprocessed image array (1, 224, 224, 3) with values 0-255 FLOAT32
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Validate
        validation = self.validate_image(image)
        if not validation.is_valid:
            raise ValueError(f"Image validation failed: {validation.error_message}")
        
        # Log warnings
        if validation.warnings and self.log_predictions:
            for warning in validation.warnings:
                print(f"⚠️  {warning}")
        
        # Resize to model input size
        image_resized = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Ensure correct dtype and range (0-255 FLOAT32)
        # Model has built-in Rescaling(1/255) layer
        image_array = image_resized.astype(np.float32)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_batch
    
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        confidence_level: Union[ConfidenceLevel, float] = ConfidenceLevel.BALANCED,
        return_top_k: int = 5,
        reject_unknown: bool = True,
    ) -> PredictionResult:
        """
        Predict rice disease with confidence thresholding.
        
        Args:
            image: Input image (various formats supported)
            confidence_level: Confidence threshold (enum or float 0-1)
            return_top_k: Number of top predictions to include
        
        Returns:
            PredictionResult with prediction and metadata
        """
        import time
        start_time = time.perf_counter()
        
        # Get threshold value
        if isinstance(confidence_level, ConfidenceLevel):
            threshold = confidence_level.value
        else:
            threshold = float(confidence_level)

        # If a calibration config is loaded, use its tuned threshold by default.
        if isinstance(confidence_level, ConfidenceLevel):
            threshold = max(threshold, self.default_confidence_threshold)
        
        # Preprocess image
        try:
            image_array = self.preprocess_image(image)
        except ValueError as e:
            # Return error result for invalid images
            return PredictionResult(
                predicted_class="ERROR",
                confidence=0.0,
                all_probabilities={},
                is_confident=False,
                threshold_used=threshold,
                inference_time_ms=0.0,
                is_rejected=True,
                rejection_reason="invalid_input",
            )
        
        # Run inference
        if self.use_tflite:
            # TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            # Keras inference
            predictions = self.model.predict(image_array, verbose=0)[0]

        # Apply optional post-hoc temperature scaling.
        predictions = self._apply_temperature_to_probs(predictions, self.temperature)
        
        # Get prediction results
        predicted_idx = np.argmax(predictions)
        predicted_class = self.classes[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Build probability dictionary
        all_probs = {
            self.class_display_names[cls]: float(predictions[i])
            for i, cls in enumerate(self.classes)
        }
        
        # Sort and take top k
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:return_top_k])
        
        # Confidence + entropy based rejection for robustness on OOD/ambiguous inputs.
        pred_entropy = self._normalized_entropy(predictions)
        is_confident = confidence >= threshold
        is_rejected = False
        rejection_reason = ""
        if reject_unknown and ((not is_confident) or (pred_entropy > self.entropy_threshold)):
            is_rejected = True
            if not is_confident and pred_entropy > self.entropy_threshold:
                rejection_reason = "low_confidence_and_high_entropy"
            elif not is_confident:
                rejection_reason = "low_confidence"
            else:
                rejection_reason = "high_entropy"
        
        # Calculate inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Update statistics
        self.prediction_count += 1
        if not is_confident:
            self.rejection_count += 1
        
        # Create result
        result = PredictionResult(
            predicted_class=(
                "Unknown / Not a rice leaf"
                if is_rejected
                else self.class_display_names[predicted_class]
            ),
            confidence=confidence,
            all_probabilities=sorted_probs,
            is_confident=(is_confident and not is_rejected),
            threshold_used=threshold,
            inference_time_ms=inference_time,
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            entropy=pred_entropy,
        )
        
        # Log if enabled
        if self.log_predictions:
            status = "[REJECTED]" if is_rejected else ("[CONFIDENT]" if is_confident else "[UNCERTAIN]")
            print(f"{status} | {result.predicted_class} ({confidence:.2%}) | "
                  f"{inference_time:.1f}ms | Threshold: {threshold:.2f} | Entropy: {pred_entropy:.3f}")
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str, Path]],
        confidence_level: Union[ConfidenceLevel, float] = ConfidenceLevel.BALANCED
    ) -> List[PredictionResult]:
        """
        Predict multiple images (processes sequentially with progress).
        
        Args:
            images: List of images
            confidence_level: Confidence threshold
        
        Returns:
            List of PredictionResults
        """
        results = []
        for i, image in enumerate(images):
            if self.log_predictions:
                print(f"Processing image {i+1}/{len(images)}...")
            result = self.predict(image, confidence_level)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics for monitoring.
        
        Returns:
            Dictionary with prediction statistics
        """
        rejection_rate = (self.rejection_count / self.prediction_count * 100) if self.prediction_count > 0 else 0
        
        return {
            'total_predictions': self.prediction_count,
            'rejections': self.rejection_count,
            'rejection_rate_percent': rejection_rate,
            'model_type': 'TFLite' if self.use_tflite else 'Keras',
            'model_path': str(self.model_path)
        }
    
    def reset_statistics(self):
        """Reset prediction statistics"""
        self.prediction_count = 0
        self.rejection_count = 0


# Convenience function for quick inference
def quick_predict(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = None,
    confidence_threshold: float = 0.85
) -> Dict:
    """
    Quick prediction function for simple use cases.
    
    Args:
        image_path: Path to image file
        model_path: Path to model (auto-detected if None)
        confidence_threshold: Confidence threshold (0.0-1.0)
    
    Returns:
        Prediction result as dictionary
    """
    if model_path is None:
        # Try to find model in standard location
        base_dir = Path(__file__).parent.parent
        keras_path = base_dir / "models" / "rice_disease_model_final.keras"
        tflite_path = base_dir / "models" / "rice_disease_model.tflite"
        
        if tflite_path.exists():
            model_path = tflite_path
        elif keras_path.exists():
            model_path = keras_path
        else:
            raise FileNotFoundError("No model found in standard locations")
    
    detector = RiceDiseaseDetector(model_path, log_predictions=False)
    result = detector.predict(image_path, confidence_level=confidence_threshold)
    return result.to_dict()


if __name__ == "__main__":
    # Example usage
    print("Rice Disease Detector - Production Module")
    print("="*70)
    print("\nExample usage:")
    print("""
    from production_inference import RiceDiseaseDetector, ConfidenceLevel
    
    # Initialize detector
    detector = RiceDiseaseDetector("models/rice_disease_model.tflite")
    
    # Make prediction
    result = detector.predict("test_image.jpg", ConfidenceLevel.BALANCED)
    
    if result.is_confident:
        print(f"Disease: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.2%}")
    else:
        print("Uncertain - please retake image")
    """)
