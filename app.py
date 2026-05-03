"""
Rice Disease Detection - Production Streamlit App
With Presentation Mode for demos and reviews
"""

import streamlit as st
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add scripts to path for production module
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from production_inference import (
    RiceDiseaseDetector,
    ConfidenceLevel,
    PredictionResult
)

# Try to import treatment engine
try:
    from treatment_recommendations import TreatmentRecommendationEngine
    TREATMENT_ENGINE_AVAILABLE = True
except ImportError:
    TREATMENT_ENGINE_AVAILABLE = False

# Configuration
MODEL_PATH = BASE_DIR / "models" / "rice_disease_model.tflite"
DEMO_IMAGES_DIR = BASE_DIR / "demo_images"
IMG_SIZE = (224, 224)
CLASSES = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Healthy', 'Hispa']
CLASS_COLORS = {
    'Bacterial Blight': '#FF6B6B',
    'Blast': '#FFA500',
    'Brown Spot': '#FFD700',
    'Healthy': '#4CAF50',
    'Hispa': '#9C27B0'
}

# Disease information (fallback if treatment engine not available)
DISEASE_INFO = {
    'Bacterial Blight': {
        'symptoms': 'Water-soaked lesions on leaves, yellowing, wilting',
        'treatment': 'Use resistant varieties, apply copper-based bactericides',
        'severity': 'High'
    },
    'Blast': {
        'symptoms': 'Diamond-shaped lesions with gray centers, leaf death',
        'treatment': 'Apply fungicides (tricyclazole), improve drainage',
        'severity': 'Very High'
    },
    'Brown Spot': {
        'symptoms': 'Circular brown spots with yellow halos on leaves',
        'treatment': 'Seed treatment, foliar fungicide application',
        'severity': 'Medium'
    },
    'Healthy': {
        'symptoms': 'No disease symptoms present',
        'treatment': 'Continue regular care and monitoring',
        'severity': 'None'
    },
    'Hispa': {
        'symptoms': 'White streaks, scraping damage on leaf surface',
        'treatment': 'Use insecticides, remove damaged leaves',
        'severity': 'Medium'
    }
}

# Page configuration
st.set_page_config(
    page_title="Rice Disease Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stAlert { padding: 1rem; border-radius: 0.5rem; }
    .prediction-box {
        padding: 1.5rem; border-radius: 0.5rem;
        background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-container {
        background-color: white; padding: 1rem;
        border-radius: 0.5rem; text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .presentation-header {
        background: linear-gradient(90deg, #2E7D32, #4CAF50);
        color: white; padding: 2rem; border-radius: 1rem;
        text-align: center; margin-bottom: 2rem;
    }
    .demo-card {
        background: white; border-radius: 1rem;
        padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the production detector"""
    try:
        detector = RiceDiseaseDetector(
            model_path=MODEL_PATH,
            enable_validation=True,
            log_predictions=False
        )
        return detector
    except Exception as e:
        st.error(f"Error loading detector: {str(e)}")
        return None


@st.cache_resource
def load_treatment_engine():
    """Load treatment recommendation engine if available"""
    if TREATMENT_ENGINE_AVAILABLE:
        try:
            return TreatmentRecommendationEngine()
        except:
            return None
    return None


def predict_disease(detector, image, confidence_threshold):
    """Run prediction using production inference module"""
    try:
        result = detector.predict(image, confidence_level=confidence_threshold)
        all_probs = np.zeros(len(CLASSES))
        for i, class_name in enumerate(CLASSES):
            all_probs[i] = result.all_probabilities.get(class_name, 0.0)
        return {
            'class': result.predicted_class,
            'confidence': result.confidence,
            'all_probabilities': all_probs,
            'inference_time': result.inference_time_ms,
            'is_confident': result.is_confident,
            'threshold_used': result.threshold_used
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def plot_predictions(probabilities):
    """Create bar chart of predictions"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [CLASS_COLORS[cls] for cls in CLASSES]
    bars = ax.barh(CLASSES, probabilities * 100, color=colors, alpha=0.8)
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_title('Disease Prediction Confidence', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def get_demo_images():
    """Get list of demo images"""
    if DEMO_IMAGES_DIR.exists():
        images = list(DEMO_IMAGES_DIR.glob('*.jpg')) + list(DEMO_IMAGES_DIR.glob('*.png'))
        return sorted([img for img in images if img.stem != 'README'])
    return []


def display_result_card(result, disease_info, treatment_engine=None):
    """Display prediction result as a styled card"""
    predicted_class = result['class']
    confidence = result['confidence']
    is_confident = result['is_confident']
    
    # Handle OOD / rejected images
    is_rejected = predicted_class in ('Unknown / Not a rice leaf', 'ERROR')
    if is_rejected:
        st.warning(
            "⚠️ **Not a Rice Leaf / Uncertain Image**\n\n"
            "The safeguard system rejected this image. Possible reasons:\n"
            "- Image does not appear to contain a rice leaf\n"
            "- Image is too blurry or low quality\n"
            "- Prediction confidence is too low\n\n"
            "Please upload a clear, well-lit photo of a rice leaf."
        )
        return
    
    border_color = CLASS_COLORS.get(predicted_class, '#666') if is_confident else '#999'
    opacity = '1.0' if is_confident else '0.7'
    
    st.markdown(f"""
    <div style='background-color: {border_color}; padding: 1.5rem; border-radius: 0.5rem; 
                color: white; text-align: center; margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.2); opacity: {opacity};'>
        <h2 style='margin: 0; color: white;'>{predicted_class}</h2>
        <h3 style='margin: 0.5rem 0 0 0; color: white;'>{confidence*100:.1f}% Confidence</h3>
        {'<p style="margin: 0.5rem 0 0 0; color: white; font-size: 0.9rem;">⚠️ Below Threshold</p>' if not is_confident else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Prediction", predicted_class)
    with col_b:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    with col_c:
        st.metric("Threshold", f"{result['threshold_used']:.2f}")
    with col_d:
        st.metric("Inference", f"{result['inference_time']:.1f}ms")
    
    # Confidence indicator
    if is_confident:
        if confidence > 0.95:
            st.success("🟢 Very High Confidence - Highly reliable prediction")
        elif confidence > 0.85:
            st.success("🟢 High Confidence - Reliable prediction")
        else:
            st.info("🟡 Good Confidence - Prediction is reliable")
    else:
        st.error("🔴 Below Confidence Threshold - Prediction may be unreliable")
    
    # Disease/Treatment info
    st.markdown("---")
    st.subheader("📋 Disease Information & Treatment")
    
    info = disease_info.get(predicted_class, {})
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Symptoms:**")
        st.write(info.get('symptoms', 'N/A'))
    with col2:
        st.markdown("**Recommended Treatment:**")
        st.write(info.get('treatment', 'N/A'))
    
    severity = info.get('severity', 'Unknown')
    severity_colors = {'Very High': 'error', 'High': 'error', 'Medium': 'warning', 'None': 'success'}
    if severity != 'None':
        getattr(st, severity_colors.get(severity, 'info'))(f"⚠️ Severity: {severity}")
    else:
        st.success("✅ No disease detected - Plant appears healthy")



def presentation_mode():
    """Run automated presentation mode cycling through demo images"""
    st.markdown("""
    <div class='presentation-header'>
        <h1>🌾 Rice Disease Detection System</h1>
        <h3>Automated Demonstration</h3>
        <p>AI-powered disease detection for precision agriculture</p>
    </div>
    """, unsafe_allow_html=True)
    
    demo_images = get_demo_images()
    if not demo_images:
        st.error("No demo images found in demo_images/ folder")
        return
    
    detector = load_detector()
    if detector is None:
        st.error("Failed to load detector")
        return
    
    # Presentation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auto_advance = st.checkbox("🔄 Auto-advance (3 seconds)", value=False)
        current_idx = st.session_state.get('demo_idx', 0)
        
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.demo_idx = (current_idx - 1) % len(demo_images)
                st.rerun()
        with nav_col2:
            st.markdown(f"<center><b>{current_idx + 1} / {len(demo_images)}</b></center>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.demo_idx = (current_idx + 1) % len(demo_images)
                st.rerun()
    
    # Display current demo image
    current_image_path = demo_images[current_idx]
    image = Image.open(current_image_path)
    
    col_img, col_result = st.columns([1, 1])
    
    with col_img:
        st.markdown("### 📷 Input Image")
        st.image(image, caption=f"Sample: {current_image_path.stem}", use_container_width=True)
        st.info(f"**Expected class:** {current_image_path.stem.replace('_', ' ').title()}")
    
    with col_result:
        st.markdown("### 🔬 Analysis Result")
        result = predict_disease(detector, image, ConfidenceLevel.BALANCED)
        if result:
            display_result_card(result, DISEASE_INFO)
    
    # Detailed chart below
    if result:
        st.markdown("---")
        st.subheader("📊 Confidence Distribution")
        fig = plot_predictions(result['all_probabilities'])
        st.pyplot(fig)
    
    # Auto-advance logic
    if auto_advance:
        time.sleep(3)
        st.session_state.demo_idx = (current_idx + 1) % len(demo_images)
        st.rerun()


def interactive_mode():
    """Standard interactive upload mode"""
    # Header
    st.title("🌾 Rice Disease Detection System")
    st.markdown("### AI-Powered Disease Detection for Precision Agriculture")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Model**: MobileNetV3-Small  
        **TFLite Size**: 4.25 MB  
        **Classes**: 5  
        **Status**: ✅ Production Ready
        """)
        
        st.divider()
        st.header("⚙️ Settings")
        
        threshold_option = st.selectbox(
            "Confidence Threshold:",
            ["Conservative (0.95)", "Balanced (0.85)", "Permissive (0.80)", "None (0.00)"],
            index=1
        )
        threshold_map = {
            "Conservative (0.95)": ConfidenceLevel.CONSERVATIVE,
            "Balanced (0.85)": ConfidenceLevel.BALANCED,
            "Permissive (0.80)": ConfidenceLevel.PERMISSIVE,
            "None (0.00)": ConfidenceLevel.NONE
        }
        selected_threshold = threshold_map[threshold_option]
        
        st.info(f"**Threshold**: {selected_threshold.value:.2f}")
        
        st.divider()
        st.header("🎯 Supported Diseases")
        for disease in CLASSES:
            severity = DISEASE_INFO[disease]['severity']
            emoji = "🔴" if severity == "Very High" else "🟠" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
            st.markdown(f"{emoji} **{disease}**")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a rice leaf image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of a rice leaf"
        )
        
        # Demo images quick select
        st.markdown("---")
        st.subheader("Or try demo images")
        demo_images = get_demo_images()
        if demo_images:
            demo_names = [img.stem.replace('_', ' ').title() for img in demo_images]
            selected_demo = st.selectbox("Select demo image:", ["-- Select --"] + demo_names)
            if selected_demo != "-- Select --":
                idx = demo_names.index(selected_demo)
                uploaded_file = demo_images[idx]
    
    with col2:
        st.header("🔬 Analysis Result")
        
        if uploaded_file is not None:
            # Load and display image
            if isinstance(uploaded_file, Path):
                image = Image.open(uploaded_file)
                filename = uploaded_file.name
            else:
                image = Image.open(uploaded_file)
                filename = uploaded_file.name
            
            st.image(image, caption=f"Uploaded: {filename}", use_container_width=True)
            
            # Run prediction
            detector = load_detector()
            if detector:
                with st.spinner("Analyzing..."):
                    result = predict_disease(detector, image, selected_threshold)
                
                if result:
                    display_result_card(result, DISEASE_INFO)
                    
                    # Detailed chart
                    st.markdown("---")
                    st.subheader("📊 Confidence Distribution")
                    fig = plot_predictions(result['all_probabilities'])
                    st.pyplot(fig)
                    
                    # Export
                    st.markdown("---")
                    results_text = f"""Rice Disease Detection Results
==============================
Image: {filename}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Prediction: {result['class']}
Confidence: {result['confidence']*100:.2f}%
Inference Time: {result['inference_time']:.2f}ms
Confident: {'Yes' if result['is_confident'] else 'No'}

All Probabilities:
"""
                    for cls, prob in zip(CLASSES, result['all_probabilities']):
                        results_text += f"  {cls}: {prob*100:.2f}%\n"
                    
                    st.download_button(
                        label="📥 Download Results",
                        data=results_text,
                        file_name=f"rice_disease_result_{int(time.time())}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("👈 Upload an image or select a demo image to see predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Rice Disease Detection System v2.0</strong> | Powered by MobileNetV3-Small + TFLite</p>
        <p>🎯 Production-Grade Inference | ⚡ Fast & Accurate | 🛡️ Confidence Thresholding</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main app entry point with mode selection"""
    # Mode selector at top
    mode = st.radio(
        "Select Mode:",
        ["🖥️ Interactive Mode", "🎬 Presentation Mode"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if mode == "🎬 Presentation Mode":
        presentation_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
