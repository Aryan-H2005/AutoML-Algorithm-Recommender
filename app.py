import streamlit as st
import pandas as pd

from source.pipeline import run_recommendation

st.set_page_config(
    page_title="AutoML Recommender",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (UI Styling)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0b0f19;
}
.stApp {
    background: linear-gradient(135deg, #0b0f19, #111827);
    color: white;
}
.title {
    font-size: 48px;
    font-weight: 800;
    color: #5bffc8;
}
.subtitle {
    font-size: 18px;
    color: #9ca3af;
}
.upload-box {
    border: 2px dashed #5bffc8;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
}
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">AutoML Algorithm Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">📂Upload → Analyze → Get Best ML Model</div>', unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("🚀Run Recommendation"):

        problem_type, results, best_model = run_recommendation(df, target)

        st.subheader("🧠 Problem Type")
        st.write(problem_type)

        st.subheader("📊 Model Performance")
        st.write(results)

        st.subheader("🏆 Best Model")
        st.write(best_model)