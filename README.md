# AR Rice Crop Disease Detection with Precision Treatment

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/notnova/Rice_Leaf_Dataset)

An end-to-end precision agriculture solution featuring a deep learning pipeline (EfficientNetB0) and an Android Augmented Reality (AR) application for real-time rice leaf disease detection and treatment recommendations.

## 🚀 Overview
This project addresses rice crop health through a multi-stage approach:
1.  **Data Pipeline**: Automated cleaning and clustering of rice leaf images.
2.  **Machine Learning**: EfficientNetB0-based classification model (TFLite) optimized for mobile deployment.
3.  **Android App**: Real-time AR scanning with on-device inference, uncertainty calibration (entropy gating), and multilingual treatment advice.

## 📊 Dataset
The dataset contains over 5,400 cleaned images across 6 classes (including background/noise). Due to size constraints (2.3 GB), the raw images are hosted on Hugging Face.

**Download link**: [notnova/Rice_Leaf_Dataset](https://huggingface.co/datasets/notnova/Rice_Leaf_Dataset)

### Classes
- Bacterial Blight
- Blast
- Brown Spot
- Healthy
- Hispa
- Background (OOD)

## 📱 Mobile Application
The Android application (`/android`) uses the TFLite model to provide:
- **AR Overlays**: Visual indicators of disease severity and location.
- **Precision Treatment**: Chemical and organic treatment suggestions based on the detected disease.
- **Multilingual Support**: English, Hindi, and Kannada.
- **Uncertainty Gating**: Rejects non-leaf inputs or low-confidence predictions to prevent false positives in the field.

## 🛠️ Setup & Reproducibility

### 1. Requirements
- Python 3.10+
- Android Studio (for app development)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (optional, for downloading data)

### 2. Download Data
To retrain the model, download the dataset into the project root:
```powershell
hf download notnova/Rice_Leaf_Dataset --local-dir Dataset_clean_6class --repo-type dataset
```

### 3. Run Pipeline
To reproduce the training and calibration:
```powershell
python scripts/production_pipeline_v5_6class.py
```

## 📈 Project Status
See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed metrics, model comparison (EfficientNet vs MobileNet), and the latest integration logs.

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
