import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
from database import Database

# Initialize Database
db = Database()

# Page Config
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
# Custom Styling for Creative Login
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #9d4edd;
        --secondary: #7b2cbf;
        --bg-dark: #0f0c29;
        --glass: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    /* Override Streamlit defaults */
    .stApp {
        background: radial-gradient(circle at top right, #240b36, #0f0c29);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    /* Reduce top padding */
    .block-container {
        padding-top: 8rem !important;
        padding-bottom: 0rem !important;
    }

    [data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
        border-right: 1px solid var(--glass-border);
    }

    /* Sidebar Navigation */
    .nav-item {
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.3s ease;
        color: #ccd6f6;
        text-decoration: none !important;
    }

    .nav-item:hover {
        background: var(--glass);
        color: #fff;
    }

    .nav-item.active {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.3);
    }

    /* Login Card 
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 70vh; /* Reduced from 80vh */
    }*/

    .login-card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 3rem;
        width: 100%;
        max-width: 900px;
        display: flex;
        gap: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .login-form-area {
        flex: 1;
    }

    .login-illustration {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 2rem;
    }

    /* Inputs & Buttons */
    .stTextInput>div>div>input {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid var(--glass-border) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
    }

    .stTextInput>div>div>input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(157, 78, 221, 0.2) !important;
    }

    .submit-btn button {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        border-radius: 10px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
    }

    .submit-btn button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px -10px var(--primary) !important;
    }

    /* Heading Styles */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .login-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #fff, #b794f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .login-subtitle {
        color: #a0aec0;
        margin-bottom: 2rem;
    }

    .feature-icon {
        width: 80px;
        height: 80px;
        background: var(--glass);
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--glass-border);
    }

    /* Sidebar Button Styling */
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    
    .stButton > button {
        width: 100% !important;
        border: none !important;
        background: transparent !important;
        color: #ccd6f6 !important;
        padding: 0.8rem 1rem !important;
        margin: 0.2rem 0 !important;
        border-radius: 10px !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        transition: all 0.3s ease !important;
        font-size: 1rem !important;
    }

    .stButton > button:hover {
        background: var(--glass) !important;
        color: #fff !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(157, 78, 221, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Session State for Auth
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"

# Sidebar Navigation
def render_sidebar():
    with st.sidebar:
        if not st.session_state.logged_in:
            st.markdown('<div class="sidebar-title">☰ Navigation</div>', unsafe_allow_html=True)
            
            # Define menu items
            menu_items = {
                "Login": "🔑 Login",
                "Register": "👤 Create Account",
                "Forgot": "❌ Forgot Password?",
                "Reset": "🔄 Reset Password"
            }
            
            for page_key, label in menu_items.items():
                is_active = st.session_state.current_page == page_key
                if st.button(label, key=f"nav_{page_key}", type="primary" if is_active else "secondary"):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
        if st.session_state.logged_in:
            if st.button("Logout", key="logout_btn"):
                st.session_state.logged_in = False
                st.session_state.current_page = "Login"
                st.rerun()

# Constants
CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
MODEL_PATH = 'waste_classifier.keras'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'waste_classifier_final.keras'

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            return None
    return None

def get_gradcam(model, img_array):
    """Generates Grad-CAM heatmap."""
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name or 'relu' in layer.name:
            last_conv_layer = layer
            break
            
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def render_login():
    render_sidebar()
    
    # Use columns to center the card
    _, center_col, _ = st.columns([0.1, 0.8, 0.1])
    
    with center_col:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Create the Card structure with two columns inside
        card_col1, card_col2 = st.columns([1, 1], gap="large")
        
        with card_col1:
            st.markdown('<div class="login-form-area">', unsafe_allow_html=True)
            st.markdown('<div class="login-title">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-subtitle">Enter your credentials to access the AI dashboard</div>', unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="Your unique username")
            password = st.text_input("Password", placeholder="Your password", type="password")
            
            st.markdown('<div class="submit-btn">', unsafe_allow_html=True)
            if st.button("Login"):
                if username and password: # Simple mock check
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Please enter credentials")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with card_col2:
            st.markdown('<div class="login-illustration">', unsafe_allow_html=True)
            st.markdown('<div class="feature-icon">🤖</div>', unsafe_allow_html=True)
            st.markdown('### Smart Waste Intelligence', unsafe_allow_html=True)
            st.markdown("""
                Our AI-driven system helps categorize waste with over 90% accuracy.
                - 🚀 Real-time analysis
                - 📈 Performance tracking
                - ✨ Neural visualization
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

def render_register():
    render_sidebar()
    _, center_col, _ = st.columns([0.1, 0.8, 0.1])
    with center_col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown('<div class="login-form-area">', unsafe_allow_html=True)
            st.markdown('<div class="login-title">Join Us</div>', unsafe_allow_html=True)
            st.markdown('<div class="login-subtitle">Create an account to start tracking waste metrics</div>', unsafe_allow_html=True)
            st.text_input("Full Name", placeholder="John Doe")
            st.text_input("Email", placeholder="john@example.com")
            st.text_input("Password", type="password", placeholder="Create a strong password")
            st.markdown('<div class="submit-btn">', unsafe_allow_html=True)
            if st.button("Create Account", key="reg_btn"):
                st.success("Account created! (Demo mode)")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="login-illustration">', unsafe_allow_html=True)
            st.markdown('<div class="feature-icon">✨</div>', unsafe_allow_html=True)
            st.markdown('### Get Started Today', unsafe_allow_html=True)
            st.markdown("Unlock advanced analytics and personalized waste management tips.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_forgot_password():
    render_sidebar()
    _, center_col, _ = st.columns([0.2, 0.6, 0.2])
    with center_col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-form-area">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Forgot Password?</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">No worries! Enter your details to receive reset instructions.</div>', unsafe_allow_html=True)
        
        st.text_input("Username", placeholder="Your username")
        st.text_input("Email ID", placeholder="your@email.com")
        
        st.markdown('<div class="submit-btn">', unsafe_allow_html=True)
        if st.button("Send Reset Link", key="forgot_btn"):
            st.info("Reset link sent! (Check your email)")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_reset_password():
    render_sidebar()
    _, center_col, _ = st.columns([0.2, 0.6, 0.2])
    with center_col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-form-area">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Reset Password</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Choose a new password for your account.</div>', unsafe_allow_html=True)
        
        st.text_input("Username", placeholder="Your username")
        st.text_input("Email ID", placeholder="your@email.com")
        st.text_input("New Password", type="password")
        st.text_input("Confirm New Password", type="password")
        
        st.markdown('<div class="submit-btn">', unsafe_allow_html=True)
        if st.button("Update Password", key="reset_btn"):
            st.success("Password updated successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_dashboard():
    render_sidebar()
    
    # Dashboard Styling Overrides for "Premium" look
    st.markdown("""
        <style>
        .stMetric { background: var(--glass) !important; border: 1px solid var(--glass-border) !important; padding: 1.5rem !important; border-radius: 15px !important; }
        [data-testid="stMetricValue"] { color: var(--primary) !important; font-weight: 700 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 20px; }
        .stTabs [data-baseweb="tab"] { background: transparent !important; color: #a0aec0 !important; border: none !important; }
        .stTabs [aria-selected="true"] { color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("♻️ Waste Intelligence Dashboard")
    st.markdown("---")

    # Sidebar: Global Stats
    with st.sidebar:
        st.header("📊 Global Statistics")
        stats = db.get_stats()
        st.metric("Total Scans", stats["total_scans"])
        st.metric("Avg. Confidence", f"{stats['avg_confidence']}%")
        
        st.info("System optimized for 10 waste categories.")

    tab1, tab2, tab3 = st.tabs(["🚀 Classification", "📈 Insights", "📝 Project Info"])

    model = load_my_model()
    if model is None:
        st.error(f"Model file not found or corrupted. Please ensure '{MODEL_PATH}' exists.")
        return

    with tab1:
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown("### 📤 Image Upload")
            uploaded_file = st.file_uploader("Upload waste image for analysis", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Target Image", use_container_width=True)

        with col2:
            st.markdown("### 🧠 AI Analysis")
            if uploaded_file:
                # Preprocess
                rgb_image = image.convert("RGB")
                resized_img = rgb_image.resize((224, 224))
                img_array = np.array(resized_img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                with st.spinner("Analyzing neural pathways..."):
                    preds = model.predict(img_array)
                    class_idx = np.argmax(preds[0])
                    confidence = preds[0][class_idx]
                    label = CLASSES[class_idx]

                # Result Metrics
                c1, c2 = st.columns(2)
                c1.metric("Category", label.capitalize())
                c2.metric("Confidence", f"{confidence*100:.1f}%")

                # Database Logging
                if 'last_prediction_id' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
                    pred_id = db.log_prediction(uploaded_file.name, label, float(confidence))
                    st.session_state.last_prediction_id = pred_id
                    st.session_state.last_file = uploaded_file.name

                st.markdown("---")
                # Explainability toggle
                if st.button("✨ Visualize Neural Attention (Grad-CAM)"):
                    try:
                        heatmap = get_gradcam(model, img_array)
                        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
                        heatmap = np.uint8(255 * heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        
                        original_cv = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                        superimposed = cv2.addWeighted(original_cv, 0.6, heatmap, 0.4, 0)
                        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                        
                        st.image(superimposed_rgb, caption="Regions of interest used by AI", use_container_width=True)
                    except Exception as e:
                        st.error(f"Visualization error: {e}")
            else:
                st.info("Please upload an image to start the analysis pipeline.")

    with tab2:
        st.markdown("### 📊 Database Insights")
        history = db.get_recent_history(limit=10)
        if history:
            import pandas as pd
            df = pd.DataFrame(history, columns=["ID", "Timestamp", "Filename", "Prediction", "Confidence"])
            st.dataframe(df.drop(columns=["ID"]), use_container_width=True)
            
            # Simple chart if enough data
            if len(df) > 1:
                st.line_chart(df.set_index("Timestamp")["Confidence"])
        else:
            st.write("No history available yet. Start classifying!")

    with tab3:
        st.markdown("### ℹ️ Project Context")
        st.write("""
        **Smart Waste Classifier** is a state-of-the-art waste management tool powered by deep learning.
        - **Model Architecture**: MobileNetV2 (Transfer Learning)
        - **Classes**: 10 distinct categories of common waste.
        - **Goal**: Facilitate automated sorting and improve recycling rates.
        """)
        st.markdown("Designed for performance and transparency.")

def main():
    if st.session_state.logged_in:
        render_dashboard()
    else:
        if st.session_state.current_page == "Login":
            render_login()
        elif st.session_state.current_page == "Register":
            render_register()
        elif st.session_state.current_page == "Forgot":
            render_forgot_password()
        elif st.session_state.current_page == "Reset":
            render_reset_password()

if __name__ == "__main__":
    main()
