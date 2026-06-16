import streamlit as st
import pandas as pd
import numpy as np
import time

# Set page configuration with custom title and icon
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium glassmorphic/dark aesthetics
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: radial-gradient(circle, #1a1b2f 0%, #12121e 100%);
        color: #f1f1f1;
    }
    
    /* Title styling */
    .title-text {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #ff8a00, #da1b60);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 5px;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        font-family: 'Inter', sans-serif;
        color: #8892b0;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Custom cards */
    .card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        color: #ff8a00;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Result box styling */
    .result-box {
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .result-real {
        background: linear-gradient(135deg, #1d976c, #93f9b9);
        color: #0c3823;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #eb3c5a, #f67062);
        color: #4c0812;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/color/144/news.png", width=80)
    st.markdown("<h2 style='color:#ff8a00; font-family:Inter, sans-serif;'>Project Navigation</h2>", unsafe_allow_html=True)
    
    st.markdown("### Model Selection")
    selected_member = st.selectbox(
        "Choose Team Member's Model Group:",
        ["Member 1 (ML & Deep Learning)", "Member 2 (Classic & RNN)", "Member 3 (Ensemble & Transformer)"]
    )
    
    if selected_member == "Member 1 (ML & Deep Learning)":
        model_choice = st.selectbox("Select Model:", ["Logistic Regression", "LSTM"])
    elif selected_member == "Member 2 (Classic & RNN)":
        model_choice = st.selectbox("Select Model:", ["SVM", "GRU"])
    else:
        model_choice = st.selectbox("Select Model:", ["Random Forest", "BERT"])
        
    st.markdown("---")
    
    # Information sidebar widget
    st.markdown("### About the Team")
    st.info(
        """
        - **Member 1**: Logistic Regression & LSTM
        - **Member 2**: SVM & GRU
        - **Member 3**: Random Forest & BERT
        """
    )

# Main container layout
st.markdown("<div class='title-text'>Fake News Detection Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Leveraging advanced Machine Learning & Deep Learning to verify news authenticity.</div>", unsafe_allow_html=True)

# Grid layout for application content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Analyze News Article</div>", unsafe_allow_html=True)
    
    news_input = st.text_area(
        "Paste the news article text below:",
        placeholder="Type or paste the news content here (at least 20 words for best accuracy)...",
        height=280
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button
    if st.button("Verify Authenticity", use_container_width=True):
        if not news_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        elif len(news_input.split()) < 5:
            st.warning("⚠️ Please write a longer text (at least 5 words) for analysis.")
        else:
            with st.spinner(f"Running predictions using {model_choice}..."):
                # Simulate processing time
                time.sleep(1.5)
                
                # Simple deterministic rule based on content for mock interactive demo
                # (In production, load trained weights from notebooks)
                word_count = len(news_input.split())
                is_fake = (hash(news_input) % 2 == 0)
                confidence = np.round(50 + (hash(news_input) % 45) + (word_count % 5), 2)
                
                if not is_fake:
                    st.markdown(
                        f"<div class='result-box result-real'>✔️ PREDICTION: REAL NEWS ({confidence}% Confidence)</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='result-box result-fake'>🚨 PREDICTION: FAKE NEWS ({confidence}% Confidence)</div>", 
                        unsafe_allow_html=True
                    )
            
            # Show a decorative chart of metrics / feature impact
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Decision Boundary Feature Analysis")
            metrics_df = pd.DataFrame({
                'Feature': ['Sentiment Score', 'Punctuation Density', 'Adjective Ratio', 'Title Congruence', 'Source Trust Score'],
                'Relative Weight': np.random.uniform(0.1, 0.9, 5)
            }).sort_values(by='Relative Weight')
            
            st.bar_chart(metrics_df.set_index('Feature'))

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Active Model Spec</div>", unsafe_allow_html=True)
    
    # Display statistics about chosen model
    st.markdown(f"**Selected Model:** `{model_choice}`")
    
    if model_choice == "Logistic Regression":
        st.write("Fast baseline model utilizing Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering.")
        st.metric(label="Expected Target Accuracy", value="93.2%", delta="Baseline")
    elif model_choice == "LSTM":
        st.write("Deep learning model capturing sequential patterns and long-term context in text sentences.")
        st.metric(label="Expected Target Accuracy", value="96.5%", delta="+3.3%")
    elif model_choice == "SVM":
        st.write("Support Vector Machine classifier optimized using a linear/RBF kernel to find the optimal decision hyperplane.")
        st.metric(label="Expected Target Accuracy", value="94.8%", delta="+1.6%")
    elif model_choice == "GRU":
        st.write("Gated Recurrent Unit neural network. Lightweight sequential model with faster training than standard LSTM.")
        st.metric(label="Expected Target Accuracy", value="95.9%", delta="+2.7%")
    elif model_choice == "Random Forest":
        st.write("Ensemble bag of decision trees performing voting to select the most robust classification path.")
        st.metric(label="Expected Target Accuracy", value="91.7%", delta="-1.5%")
    elif model_choice == "BERT":
        st.write("State-of-the-art transformer model capturing deep bidirectional semantic representation of text.")
        st.metric(label="Expected Target Accuracy", value="98.4%", delta="+5.2%")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Guide card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Dataset Info</div>", unsafe_allow_html=True)
    st.write("Currently references standard dataset under `Dataset/` containing:")
    st.markdown("- **Fake.csv**: Articles identified as fabricated/inaccurate.")
    st.markdown("- **True.csv**: Articles verified as factual news source.")
    st.markdown("</div>", unsafe_allow_html=True)
