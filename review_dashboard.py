import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import string

# --- Page Configuration ---
st.set_page_config(page_title="📊 Customer Sentiment Dashboard", layout="wide")

st.title("📈 Customer Review Sentiment Dashboard")
st.markdown("""
Visualize sentiment trends, analyze customer confidence levels, 
and extract actionable insights from your processed reviews.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("analyzed_reviews.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    required_cols = ['sentiment', 'confidence', 'review']
    for col in required_cols:
        if col not in df.columns:
            st.error(
                f"❌ Missing required column: '{col}' — please upload the correct analyzed file.")
            st.stop()

    # --- KPI Metrics ---
    total_reviews = len(df)
    positive = (df['sentiment'].str.upper() == 'POSITIVE').sum()
    negative = (df['sentiment'].str.upper() == 'NEGATIVE').sum()
    positive_percent = (positive / total_reviews) * 100
    negative_percent = (negative / total_reviews) * 100
    avg_confidence = df['confidence'].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("🗂️ Total Reviews", total_reviews)
    col2.metric("😊 Positive Reviews", f"{positive_percent:.1f}%")
    col3.metric("😞 Negative Reviews", f"{negative_percent:.1f}%")
    st.markdown(f"**Average Model Confidence:** {avg_confidence:.1f}%")

    # --- Sidebar Filters ---
    st.sidebar.header("🔧 Filters")
    sentiment_filter = st.sidebar.multiselect(
        "Select Sentiment Type:",
        options=df['sentiment'].unique(),
        default=df['sentiment'].unique()
    )
    conf_range = st.sidebar.slider(
        "Select Confidence Range:",
        min_value=float(df['confidence'].min()),
        max_value=float(df['confidence'].max()),
        value=(float(df['confidence'].min()), float(df['confidence'].max()))
    )

    filtered_df = df[
        (df['sentiment'].isin(sentiment_filter)) &
        (df['confidence'].between(conf_range[0], conf_range[1]))
    ]

    st.subheader("📄 Filtered Reviews")
    st.dataframe(filtered_df.head(10))

    # --- Charts ---
    st.markdown("### 🎯 Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=filtered_df, x='sentiment', palette='coolwarm', ax=ax)
    plt.title("Sentiment Count Distribution")
    st.pyplot(fig)

    st.markdown("### 📊 Confidence Level Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df['confidence'], bins=20,
                 kde=True, ax=ax2, color='purple')
    plt.title("Model Confidence Distribution")
    st.pyplot(fig2)

    # --- Word Cloud Section ---
    st.markdown("## ☁️ Word Cloud Analysis")

    # Prepare cleaned text
    def clean_text(text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    pos_text = " ".join(filtered_df[filtered_df['sentiment'].str.upper(
    ) == 'POSITIVE']['review'].apply(clean_text))
    neg_text = " ".join(filtered_df[filtered_df['sentiment'].str.upper(
    ) == 'NEGATIVE']['review'].apply(clean_text))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 😊 Positive Reviews Word Cloud")
        if len(pos_text) > 0:
            pos_wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=STOPWORDS,
                colormap='Greens'
            ).generate(pos_text)
            plt.figure(figsize=(8, 4))
            plt.imshow(pos_wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("No positive reviews found for selected filters.")

    with col2:
        st.markdown("### 😞 Negative Reviews Word Cloud")
        if len(neg_text) > 0:
            neg_wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=STOPWORDS,
                colormap='Reds'
            ).generate(neg_text)
            plt.figure(figsize=(8, 4))
            plt.imshow(neg_wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.info("No negative reviews found for selected filters.")

    # --- Insights ---
    st.markdown("### 🧠 Business Insights")
    if positive_percent > 70:
        st.success(
            "💡 Customers are **very satisfied** overall with this product.")
    elif positive_percent > 50:
        st.info("💡 Customer sentiment is **moderately positive**.")
    else:
        st.error(
            "💡 Customers are **largely dissatisfied**. Consider investigating key complaints.")

    # --- Download Filtered Data ---
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Results",
        data=csv,
        file_name='filtered_analyzed_reviews.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Upload your **analyzed_reviews.csv** to begin.")
