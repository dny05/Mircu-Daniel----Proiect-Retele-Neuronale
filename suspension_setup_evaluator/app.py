
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Session State 

def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'features_extracted' not in st.session_state:
        st.session_state.features_extracted = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'page' not in st.session_state:
        st.session_state.page = "📁 Data Upload"

# Pages

def show_data_upload_page():
    st.header("📁 Upload Telemetry Data")
    st.write("Upload your telemetry CSV file here.")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload telemetry CSV from your logger"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"✅ Data loaded successfully! {len(df)} rows")
            
            st.subheader("Preview Data")
            st.dataframe(df.head())
            
            if st.button(" Process Data"):
                # Extrage features simple: media coloanelor numerice
                features = df.select_dtypes(include='number').mean().to_frame().T
                st.session_state.features = features
                st.session_state.features_extracted = True
                st.success("✅ Features extracted successfully!")
                
                st.subheader("Extracted Features")
                st.dataframe(features)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

def show_analysis_page():
    st.header("🔍 Telemetry Analysis")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please upload telemetry data first!")
        return
    
    if not st.session_state.get('features_extracted', False):
        st.warning("⚠️ Please process the data first!")
        if st.button("Go to Data Upload"):
            st.session_state.page = "📁 Data Upload"
            st.experimental_rerun()
        return
    
    st.success("✅ Data ready for analysis!")
    
    # Arată features extrase
    st.subheader("Extracted Features")
    st.dataframe(st.session_state.features)
    
    # Exemplu de grafic
    st.subheader("Feature Overview")
    fig, ax = plt.subplots(figsize=(8,4))
    st.session_state.features.T.plot(kind='bar', ax=ax)
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    st.pyplot(fig)

def show_results_page():
    st.header("📊 Results & Recommendations")
    
    if not st.session_state.get('features_extracted', False):
        st.warning("⚠️ Please process data first!")
        return
    
    st.success("✅ Evaluation ready!")
    
    # Placeholder rezultate
    st.subheader("Detected Behavior")
    behavior = np.random.choice(["Understeer", "Oversteer"])
    confidence = np.random.uniform(0.5, 1.0)
    
    st.markdown(f"- **Behavior:** {behavior}")
    st.markdown(f"- **Confidence:** {confidence*100:.1f}%")
    
    # Recomandări minimal
    st.subheader("Setup Recommendations")
    recs = [
        "Adjust front camber by +0.5°",
        "Reduce rear toe by -0.2°",
        "Check tire pressures"
    ]
    for i, r in enumerate(recs, 1):
        st.markdown(f"{i}. {r}")

def show_training_page():
    st.header("🎓 Model Training")
    
    st.info("Minimal training with synthetic data.")
    
    epochs = st.slider("Number of Epochs", 10, 200, 50, 10)
    
    if st.button("🚀 Start Training"):
        st.session_state.model_trained = True
        st.success(f"✅ Model trained for {epochs} epochs (synthetic)!")
        
        # Fake loss plot
        loss = np.exp(-np.linspace(0, 5, epochs)) + np.random.rand(epochs)*0.05
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(range(1, epochs+1), loss, label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training History")
        ax.legend()
        st.pyplot(fig)

def show_about_page():
    st.header("ℹ️ About")
    st.markdown("""
    **FS Suspension Setup Evaluator**  
    Minimal web interface for telemetry analysis and setup recommendations.

    - Upload telemetry CSV data
    - Process data to extract features
    - Analyze telemetry and visualize
    - Training & evaluation placeholders
    """)

# =========================
# Main
# =========================
def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🏎️ FS Suspension Evaluator")
        page = st.radio(
            "Navigation",
            ["📁 Data Upload", "🔍 Analysis", "📊 Results", "🎓 Model Training", "ℹ️ About"],
            index=["📁 Data Upload", "🔍 Analysis", "📊 Results", "🎓 Model Training", "ℹ️ About"].index(st.session_state.page)
        )
        st.session_state.page = page
        
        st.markdown("---")
        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
    
    # Pagini
    if st.session_state.page == "📁 Data Upload":
        show_data_upload_page()
    elif st.session_state.page == "🔍 Analysis":
        show_analysis_page()
    elif st.session_state.page == "📊 Results":
        show_results_page()
    elif st.session_state.page == "🎓 Model Training":
        show_training_page()
    elif st.session_state.page == "ℹ️ About":
        show_about_page()

# =========================
# Run App
# =========================
if __name__ == "__main__":
    main()
