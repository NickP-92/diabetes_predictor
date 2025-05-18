import streamlit as st
import matplotlib.pyplot as plt
from ml_models import load_and_preprocess, train_models

st.set_page_config(page_title="Diabetes Classifier")
st.title("🩺 Diabetes Prediction Comparison")

if st.button("Train Models"):
    X_train, X_test, y_train, y_test = load_and_preprocess()
    results = train_models(X_train, X_test, y_train, y_test)

    for model, metrics in results.items():
        st.subheader(model)
        st.text("Confusion Matrix:")
        st.text(metrics["confusion_matrix"])
        st.metric("F1 Score", f"{metrics['f1_score']:.2f}")

    st.subheader("F1 Score Chart")
    fig, ax = plt.subplots()
    ax.bar(results.keys(), [m["f1_score"] for m in results.values()], color=['skyblue', 'salmon'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.grid(axis='y')
    st.pyplot(fig)
