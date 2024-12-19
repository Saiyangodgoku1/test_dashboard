# File: dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Constants
DEFAULT_DATA_PATH = "diabetes.csv"

# Dashboard title
st.title("Professional Dynamic Dashboard")
st.markdown("## Data Insights and Visualization")

# Sidebar
st.sidebar.header("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load data
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    st.info(f"Using default dataset: {DEFAULT_DATA_PATH}")
    data = load_data(DEFAULT_DATA_PATH)

# Display data
if data is not None:
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Sidebar Filters
    st.sidebar.subheader("Filter Options")
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns

    if not data.empty:
        selected_cols = st.sidebar.multiselect("Select columns to display", data.columns, default=data.columns[:5])
        st.subheader("Filtered Dataset Preview")
        st.dataframe(data[selected_cols])

        # Numeric Data Summary
        st.sidebar.subheader("Select Numeric Column for Analysis")
        numeric_col = st.sidebar.selectbox("Numeric Columns", numeric_cols)

        if numeric_col:
            st.subheader(f"Summary Statistics for {numeric_col}")
            st.write(data[numeric_col].describe())
            
            st.subheader(f"Histogram for {numeric_col}")
            fig, ax = plt.subplots()
            sns.histplot(data[numeric_col], kde=True, bins=20, ax=ax)
            st.pyplot(fig)

        # Categorical Data Summary
        if not categorical_cols.empty:
            st.sidebar.subheader("Select Categorical Column for Analysis")
            categorical_col = st.sidebar.selectbox("Categorical Columns", categorical_cols)

            if categorical_col:
                st.subheader(f"Value Counts for {categorical_col}")
                st.write(data[categorical_col].value_counts())
                
                st.subheader(f"Bar Plot for {categorical_col}")
                fig, ax = plt.subplots()
                data[categorical_col].value_counts().plot(kind="bar", color="skyblue", ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

    # Correlation Heatmap
    if not numeric_cols.empty:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
else:
    st.error("No data available. Please upload a valid CSV file or include the default dataset.")
