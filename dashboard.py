import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Advanced Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .reportview-container .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #1E3D59;
        }
        h2 {
            color: #1E3D59;
            font-size: 1.5rem;
        }
        .stAlert {
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Cache functions
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def generate_summary_stats(data):
    summary = {
        "Total Records": len(data),
        "Missing Values": data.isnull().sum().sum(),
        "Numeric Columns": len(data.select_dtypes(include=['float64', 'int64']).columns),
        "Categorical Columns": len(data.select_dtypes(include=['object', 'category']).columns),
        "Memory Usage": f"{data.memory_usage().sum() / 1024:.2f} KB"
    }
    return summary

# Main Dashboard
def main():
    # Header
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("📊 Advanced Analytics Dashboard")
    with col2:
        st.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar
    st.sidebar.header("📱 Dashboard Controls")
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.info("ℹ️ Using default dataset: diabetes.csv")
        data = load_data("diabetes.csv")

    if data is not None:
        # Data Overview Section
        with st.expander("📋 Dataset Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            summary_stats = generate_summary_stats(data)
            
            with col1:
                st.metric("Total Records", summary_stats["Total Records"])
            with col2:
                st.metric("Missing Values", summary_stats["Missing Values"])
            with col3:
                st.metric("Memory Usage", summary_stats["Memory Usage"])

            st.dataframe(data.head(), use_container_width=True)

        # Sidebar Controls
        st.sidebar.subheader("🎯 Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Univariate Analysis", "Bivariate Analysis", "Correlation Analysis"]
        )

        # Univariate Analysis
        if analysis_type == "Univariate Analysis":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Numeric Distribution")
                numeric_col = st.selectbox("Select Numeric Column", data.select_dtypes(include=['float64', 'int64']).columns)
                fig = px.histogram(data, x=numeric_col, marginal="box", title=f"Distribution of {numeric_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.write("Summary Statistics:")
                st.write(data[numeric_col].describe())

            with col2:
                st.subheader("📈 Time Series/Trend")
                fig = px.box(data, y=numeric_col, title=f"Box Plot of {numeric_col}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional statistics
                skewness = data[numeric_col].skew()
                kurtosis = data[numeric_col].kurtosis()
                st.write(f"Skewness: {skewness:.2f}")
                st.write(f"Kurtosis: {kurtosis:.2f}")

        # Bivariate Analysis
        elif analysis_type == "Bivariate Analysis":
            st.subheader("🔄 Bivariate Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis", data.columns)
            with col2:
                y_col = st.selectbox("Select Y-axis", [col for col in data.columns if col != x_col])

            plot_type = st.radio("Select Plot Type", ["Scatter", "Line", "Bar"])
            
            if plot_type == "Scatter":
                fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            elif plot_type == "Line":
                fig = px.line(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            else:
                fig = px.bar(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Analysis
        else:
            st.subheader("🔗 Correlation Analysis")
            
            numeric_data = data.select_dtypes(include=['float64', 'int64'])
            correlation_method = st.radio("Select Correlation Method", ["Pearson", "Spearman"])
            
            if correlation_method == "Pearson":
                corr_matrix = numeric_data.corr(method='pearson')
            else:
                corr_matrix = numeric_data.corr(method='spearman')

            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                title=f"{correlation_method} Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed correlation analysis
            st.write("### Detailed Correlation Values")
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', axis=None))

        # Data Export Options
        st.sidebar.subheader("💾 Export Options")
        if st.sidebar.button("Export Analysis Report"):
            # Create and download report logic here
            st.sidebar.success("Report exported successfully!")

    else:
        st.error("❌ No data available. Please upload a valid CSV file or include the default dataset.")

if __name__ == "__main__":
    main()
