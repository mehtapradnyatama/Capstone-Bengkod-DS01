# ============================================================================
# MEHTA PRADNYATAMA
# A11.2022.14183
# BENGKOD DS 01
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Obese.",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING - GLASSMORPHISM THEME
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%, rgba(255,255,255,0.1) 100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
        background: linear-gradient(45deg, #ffffff, #f8f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.9;
        position: relative;
        z-index: 2;
        margin: 0;
    }
    
    /* Glass Card Styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
        transition: opacity 0.3s ease;
        opacity: 0;
    }
    
    .glass-card:hover::before {
        opacity: 1;
    }
    
    .glass-card h3 {
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Progress Steps */
    .step-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
    }
    
    .step {
        flex: 1;
        text-align: center;
        position: relative;
        z-index: 2;
    }
    
    .step-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        margin: 0 auto 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        z-index: 2;
    }
    
    .step-circle.active {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.4);
        transform: scale(1.1);
    }
    
    .step-circle.completed {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        box-shadow: 0 8px 25px rgba(81, 207, 102, 0.3);
    }
    
    .step-circle.pending {
        background: rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 0.7);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .step-text {
        color: white;
        font-weight: 500;
        font-size: 0.9rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .step-line {
        position: absolute;
        top: 30px;
        left: 0;
        right: 0;
        height: 3px;
        background: rgba(255, 255, 255, 0.2);
        z-index: 1;
        border-radius: 2px;
    }
    
    .step-line.completed {
        background: linear-gradient(90deg, #51cf66, #40c057);
    }
    
    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        border-color: rgba(255, 255, 255, 0.4);
    }
    
    .metric-card .icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card h2 {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 1rem 0 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 16px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
        background: linear-gradient(135deg, #ff5252 0%, #e53935 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Back Button */
    .stButton > button[data-testid*="back"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .stButton > button[data-testid*="back"]:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0.2) 100%);
    }
    
    /* Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.1) 100%);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255,255,255,0.3);
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        margin: 2rem 0;
        animation: resultSlideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes resultSlideIn {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .prediction-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .prediction-result .main-icon {
        font-size: 4.5rem;
        margin-bottom: 1.5rem;
        animation: pulse 2s ease-in-out infinite;
        position: relative;
        z-index: 2;
    }
    
    .prediction-result h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 2;
    }
    
    .prediction-result p {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 1.5rem 0;
        position: relative;
        z-index: 2;
        line-height: 1.6;
    }
    
    .prediction-result .confidence {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        display: inline-block;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 1rem;
        box-shadow: 0 8px 25px rgba(81, 207, 102, 0.3);
        position: relative;
        z-index: 2;
    }
    
    /* BMI Indicator */
    .bmi-indicator {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .bmi-indicator:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Form Elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
    }
    
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* Labels */
    .stSelectbox > label,
    .stNumberInput > label,
    .stSlider > label {
        color: white !important;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-weight: 800;
        font-size: 2rem;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    
    .section-header i {
        color: #ff6b6b;
        font-size: 1.8rem;
    }
    
    /* Action Buttons Container */
    .action-buttons {
        margin-top: 3rem;
        padding-top: 2rem;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
        
        .step-container {
            padding: 1rem;
        }
        
        .step-circle {
            width: 50px;
            height: 50px;
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_components():
    """Load all model components with error handling"""
    try:
        # Try to load all components
        model_path = "model/model_obesitas_tuned.pkl"
        preprocessor_path = "model/preprocessor_obesitas.pkl"
        encoder_path = "model/label_encoder_obesitas.pkl"
        mapping_path = "model/target_mapping.pkl"
        
        components = {}
        
        # Load model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                components['model'] = pickle.load(f)
        else:
            st.error(f"Model file not found: {model_path}")
            return None
            
        # Load preprocessor
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                components['preprocessor'] = pickle.load(f)
        else:
            st.error(f"Preprocessor file not found: {preprocessor_path}")
            return None
            
        # Load label encoder
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                components['label_encoder'] = pickle.load(f)
        else:
            st.error(f"Label encoder file not found: {encoder_path}")
            return None
            
        # Load target mapping
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                components['target_mapping'] = pickle.load(f)
        else:
            # Create default mapping if file doesn't exist
            components['target_mapping'] = {
                'Insufficient_Weight': 0,
                'Normal_Weight': 1,
                'Obesity_Type_I': 2,
                'Obesity_Type_II': 3,
                'Obesity_Type_III': 4,
                'Overweight_Level_I': 5,
                'Overweight_Level_II': 6
            }
        
        return components
        
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_bmi(weight, height):
    """Calculate BMI and return category"""
    if height <= 0 or weight <= 0:
        return 0, "Invalid"
    
    bmi = weight / (height ** 2)
    
    if bmi < 18.5:
        return bmi, "Underweight"
    elif 18.5 <= bmi < 25:
        return bmi, "Normal"
    elif 25 <= bmi < 30:
        return bmi, "Overweight"
    else:
        return bmi, "Obese"

def get_bmi_color_class(category):
    """Get CSS class for BMI category"""
    classes = {
        "Underweight": "bmi-underweight",
        "Normal": "bmi-normal",
        "Overweight": "bmi-overweight",
        "Obese": "bmi-obese"
    }
    return classes.get(category, "bmi-normal")

def predict_obesity(components, input_data):
    """Make prediction using loaded model components"""
    try:
        # Create DataFrame from input
        feature_names = ['Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP', 
                        'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF', 'TUE', 'CAEC', 'MTRANS']
        
        df = pd.DataFrame([input_data], columns=feature_names)
        
        # Preprocess the data
        X_processed = components['preprocessor'].transform(df)
        
        # Make prediction
        prediction = components['model'].predict(X_processed)[0]
        
        # Get prediction probabilities for confidence
        try:
            probabilities = components['model'].predict_proba(X_processed)[0]
            confidence = np.max(probabilities)
        except:
            confidence = 0.85  # Default confidence
        
        # Map prediction to label
        inverse_mapping = {v: k for k, v in components['target_mapping'].items()}
        prediction_label = inverse_mapping.get(prediction, "Unknown")
        
        return prediction_label, confidence
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0

def get_health_recommendation(prediction, bmi_category):
    """Get health recommendations based on prediction and BMI"""
    recommendations = {
        'Insufficient_Weight': {
            'title': 'Underweight',
            'icon': 'fas fa-arrow-up',
            'color': '#74c0fc',
            'advice': 'Consider increasing your caloric intake with nutritious foods. Consult with a healthcare provider for a proper weight gain plan.',
            'tips': ['Eat frequent, smaller meals', 'Include healthy fats', 'Consider strength training', 'Monitor your progress']
        },
        'Normal_Weight': {
            'title': 'Normal Weight',
            'icon': 'fas fa-check-circle',
            'color': '#51cf66',
            'advice': 'Excellent! Maintain your healthy lifestyle with balanced nutrition and regular physical activity.',
            'tips': ['Continue balanced eating', 'Stay active', 'Regular health check-ups', 'Maintain good sleep habits']
        },
        'Overweight_Level_I': {
            'title': 'Overweight - Level I',
            'icon': 'fas fa-exclamation-triangle',
            'color': '#ffd43b',
            'advice': 'Consider adopting healthier eating habits and increasing physical activity to reach a healthier weight range.',
            'tips': ['Reduce portion sizes', 'Increase daily activity', 'Choose whole foods', 'Stay hydrated']
        },
        'Overweight_Level_II': {
            'title': 'Overweight - Level II',
            'icon': 'fas fa-exclamation-triangle',
            'color': '#ffd43b',
            'advice': 'It\'s important to work on weight reduction through diet and exercise. Consider consulting a healthcare professional.',
            'tips': ['Create a meal plan', 'Regular exercise routine', 'Monitor progress', 'Consider professional guidance']
        },
        'Obesity_Type_I': {
            'title': 'Obesity - Type I',
            'icon': 'fas fa-hospital',
            'color': '#ff8787',
            'advice': 'Medical consultation is recommended. A structured weight loss program with professional guidance would be beneficial.',
            'tips': ['Medical consultation', 'Structured diet plan', 'Regular monitoring', 'Professional support']
        },
        'Obesity_Type_II': {
            'title': 'Obesity - Type II',
            'icon': 'fas fa-hospital',
            'color': '#ff6b6b',
            'advice': 'Immediate medical attention is advised. Comprehensive lifestyle changes and possibly medical intervention are needed.',
            'tips': ['Immediate medical attention', 'Comprehensive plan', 'Regular monitoring', 'Lifestyle modification']
        },
        'Obesity_Type_III': {
            'title': 'Obesity - Type III',
            'icon': 'fas fa-hospital-symbol',
            'color': '#ff5252',
            'advice': 'Urgent medical consultation required. Comprehensive treatment plan including possible surgical options should be considered.',
            'tips': ['Urgent medical care', 'Comprehensive treatment', 'Specialist consultation', 'Ongoing support']
        }
    }
    
    return recommendations.get(prediction, recommendations['Normal_Weight'])

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Obese.</h1>
        <p>Obesity Risk & Health Prediction App</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model components
    components = load_model_components()
    
    if components is None:
        st.error("⚠️ Unable to load the prediction model. Please check if model files are available.")
        return
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    # Progress Steps
    steps = ["Personal Info", "Health Habits", "Lifestyle", "Results"]
    
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    for i, step_name in enumerate(steps, 1):
        if i < st.session_state.step:
            circle_class = "completed"
            line_class = "completed" if i < len(steps) else ""
        elif i == st.session_state.step:
            circle_class = "active"
            line_class = ""
        else:
            circle_class = "pending"
            line_class = ""
        
        st.markdown(f"""
        <div class="step">
            <div class="step-circle {circle_class}">{i}</div>
            <div class="step-text">{step_name}</div>
            {f'<div class="step-line {line_class}"></div>' if i < len(steps) else ''}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 1: Personal Information
    if st.session_state.step == 1:
        st.markdown('<h2 class="section-header"><i class="fas fa-user"></i>Personal Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3>Basic Information</h3>', unsafe_allow_html=True)
            age = st.slider("Age", 16, 80, 25, help="Your current age in years")
            gender = st.selectbox("Gender", ["Female", "Male"], help="Select your biological gender")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3>Physical Measurements</h3>', unsafe_allow_html=True)
            height = st.number_input("Height (m)", 1.40, 2.20, 1.70, 0.01, help="Your height in meters")
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.5, help="Your current weight in kilograms")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time BMI calculation
        if height > 0 and weight > 0:
            bmi, bmi_category = calculate_bmi(weight, height)
            
            st.markdown(f"""
            <div class="bmi-indicator">
                <i class="fas fa-chart-line" style="margin-right: 10px;"></i>
                <strong>BMI: {bmi:.1f} ({bmi_category})</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Family History
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3></i> Family Background</h3>', unsafe_allow_html=True)
        family_history = st.selectbox("Family History of Overweight", 
                                    ["No", "Yes"], 
                                    help="Do you have family members with a history of being overweight?")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Next: Health Habits →"):
            st.session_state.step = 2
            st.session_state.personal_info = {
                'age': age,
                'gender': 1 if gender == "Male" else 0,
                'height': height,
                'weight': weight,
                'family_history': 1 if family_history == "Yes" else 0
            }
            st.rerun()
    
    # Step 2: Health Habits
    elif st.session_state.step == 2:
        st.markdown('<h2 class="section-header"><i class="fas fa-utensils"></i>Health & Eating Habits</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3></i> Eating Patterns</h3>', unsafe_allow_html=True)
            favc = st.selectbox("Frequent High-Calorie Food Consumption", 
                              ["No", "Yes"], 
                              help="Do you frequently consume high-calorie foods?")
            fcvc = st.slider("Vegetable Consumption (servings/day)", 1.0, 3.0, 2.0, 0.1,
                           help="How many servings of vegetables do you eat daily?")
            ncp = st.slider("Main Meals Per Day", 1.0, 4.0, 3.0, 0.1,
                          help="How many main meals do you eat per day?")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3>Consumption Habits</h3>', unsafe_allow_html=True)
            caec = st.selectbox("Food Between Meals", 
                              ["No", "Sometimes", "Frequently", "Always"],
                              help="How often do you eat food between meals?")
            ch2o = st.slider("Water Consumption (liters/day)", 1.0, 3.0, 2.0, 0.1,
                           help="How much water do you drink daily?")
            scc = st.selectbox("Calorie Consumption Monitoring", 
                             ["No", "Yes"],
                             help="Do you monitor your daily calorie consumption?")
            st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Next: Lifestyle →"):
                st.session_state.step = 3
                caec_mapping = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
                st.session_state.health_habits = {
                    'favc': 1 if favc == "Yes" else 0,
                    'fcvc': fcvc,
                    'ncp': ncp,
                    'caec': caec_mapping[caec],
                    'ch2o': ch2o,
                    'scc': 1 if scc == "Yes" else 0
                }
                st.rerun()
    
    # Step 3: Lifestyle
    elif st.session_state.step == 3:
        st.markdown('<h2 class="section-header"><i class="fas fa-running"></i>Lifestyle & Activities</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3>Personal Habits</h3>', unsafe_allow_html=True)
            calc = st.selectbox("Alcohol Consumption", 
                              ["No", "Sometimes", "Frequently", "Always"],
                              help="How often do you consume alcohol?")
            smoke = st.selectbox("Smoking", 
                               ["No", "Yes"],
                               help="Do you smoke?")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h3>Physical Activity</h3>', unsafe_allow_html=True)
            faf = st.slider("Physical Activity (days/week)", 0.0, 7.0, 2.0, 0.1,
                          help="How many days per week do you do physical activity?")
            tue = st.slider("Technology Use (hours/day)", 0.0, 8.0, 2.0, 0.1,
                          help="How many hours per day do you use technology devices?")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3>Transportation</h3>', unsafe_allow_html=True)
        mtrans = st.selectbox("Main Transportation", 
                            ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"],
                            help="What is your main mode of transportation?")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("Get Prediction →"):
                calc_mapping = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
                mtrans_mapping = {"Automobile": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Walking": 4}
                
                st.session_state.lifestyle = {
                    'calc': calc_mapping[calc],
                    'smoke': 1 if smoke == "Yes" else 0,
                    'faf': faf,
                    'tue': tue,
                    'mtrans': mtrans_mapping[mtrans]
                }
                st.session_state.step = 4
                st.rerun()
    
    # Step 4: Results
    elif st.session_state.step == 4:
        st.markdown('<h2 class="section-header"><i class="fas fa-chart-line"></i>Health Assessment Results</h2>', unsafe_allow_html=True)
        
        # Prepare input data
        input_data = [
            st.session_state.personal_info['age'],
            st.session_state.personal_info['gender'],
            st.session_state.personal_info['height'],
            st.session_state.personal_info['weight'],
            st.session_state.lifestyle['calc'],
            st.session_state.health_habits['favc'],
            st.session_state.health_habits['fcvc'],
            st.session_state.health_habits['ncp'],
            st.session_state.health_habits['scc'],
            st.session_state.lifestyle['smoke'],
            st.session_state.health_habits['ch2o'],
            st.session_state.personal_info['family_history'],
            st.session_state.lifestyle['faf'],
            st.session_state.lifestyle['tue'],
            st.session_state.health_habits['caec'],
            st.session_state.lifestyle['mtrans']
        ]
        
        # Make prediction
        prediction, confidence = predict_obesity(components, input_data)
        
        # Calculate BMI
        bmi, bmi_category = calculate_bmi(st.session_state.personal_info['weight'], 
                                        st.session_state.personal_info['height'])
        
        # Get recommendations
        recommendation = get_health_recommendation(prediction, bmi_category)
        
        # Display prediction result
        st.markdown(f"""
        <div class="prediction-result">
            <i class="{recommendation['icon']} main-icon" style="color: {recommendation['color']};"></i>
            <h1>Health Status: {recommendation['title']}</h1>
            <p>{recommendation['advice']}</p>
            <div class="confidence">
                Confidence: {confidence:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-chart-bar icon"></i>
                <h3>BMI</h3>
                <h2>{bmi:.1f}</h2>
                <p>{bmi_category}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-weight icon"></i>
                <h3>Weight</h3>
                <h2>{st.session_state.personal_info['weight']}</h2>
                <p>kg</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            risk_level = "Low" if "Normal" in prediction else "Medium" if "Overweight" in prediction else "High"
            risk_color = "#51cf66" if risk_level == "Low" else "#ffd43b" if risk_level == "Medium" else "#ff6b6b"
            st.markdown(f"""
            <div class="metric-card">
                <i class="fas fa-heartbeat icon"></i>
                <h3>Risk Level</h3>
                <h2 style="color: {risk_color};">{risk_level}</h2>
                <p>Assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New Assessment"):
                # Reset session state
                for key in ['step', 'personal_info', 'health_habits', 'lifestyle']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            if st.button("ℹ️ Medical Advice"):
                st.info("💡 Consult with healthcare professionals for personalized medical advice and treatment plans.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()