import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Random Forest Classifier Dashboard",
    page_icon="🌲",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌲 Random Forest Classification Dashboard")
st.markdown("### Student Performance Prediction System")

# Sidebar
with st.sidebar:
    st.header("Model Configuration")
    
    uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx", "xls"])
    
    st.subheader("Model Parameters")
    n_estimators = st.slider("Number of estimators", 50, 500, 200, 50)
    max_depth = st.slider("Max depth", 5, 50, 20, 5)
    min_samples_split = st.slider("Min samples split", 2, 15, 5)
    min_samples_leaf = st.slider("Min samples leaf", 1, 10, 2)
    
    test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    
    train_model = st.button("Train Model")
    
    st.markdown("---")
    st.markdown("Developed as a student performance prediction system")

# Main function
def main():
    # Initialize session state variables if they don't exist
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feature_importances' not in st.session_state:
        st.session_state.feature_importances = None
    if 'accuracy' not in st.session_state:
        st.session_state.accuracy = None
    if 'precision' not in st.session_state:
        st.session_state.precision = None  
    if 'recall' not in st.session_state:
        st.session_state.recall = None
    if 'f1' not in st.session_state:
        st.session_state.f1 = None
    if 'confusion_matrix' not in st.session_state:
        st.session_state.confusion_matrix = None
    if 'x_test' not in st.session_state:
        st.session_state.x_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'df' not in st.session_state:
        st.session_state.df = None
        
    # Load and process data
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, header=1)
            st.session_state.df = df
            
            # Display dataset information
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total rows: {df.shape[0]}")
                st.write(f"Total columns: {df.shape[1]}")
            with col2:
                st.write(f"Null values: {df.isnull().sum().sum()}")
            
            # Display sample data
            st.dataframe(df.head())
            
            # Train model if button is clicked
            if train_model:
                with st.spinner("Training model..."):
                    # Preprocess data
                    df_clean = df.dropna()
                    
                    # Feature and target separation
                    X = df_clean.drop(['Sl No ', "USN ", "Name ", "Title ", "Grade"], axis=1)
                    st.session_state.features = X.columns.tolist()
                    
                    # Encode labels
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(df_clean['Grade'])
                    st.session_state.label_encoder = label_encoder
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Save test set for predictions
                    st.session_state.x_test = X_test
                    st.session_state.y_test = y_test
                    
                    # Train Random Forest
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42
                    )
                    
                    rf.fit(X_train, y_train)
                    st.session_state.model = rf
                    
                    # Feature importance
                    feature_importances = pd.Series(
                        rf.feature_importances_, index=X.columns
                    ).sort_values(ascending=False)
                    st.session_state.feature_importances = feature_importances
                    
                    # Make predictions
                    y_pred = rf.predict(X_test)
                    
                    # Calculate metrics
                    st.session_state.accuracy = accuracy_score(y_test, y_pred)
                    st.session_state.precision = precision_score(y_test, y_pred, average='weighted')
                    st.session_state.recall = recall_score(y_test, y_pred, average='weighted')
                    st.session_state.f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Confusion matrix
                    st.session_state.confusion_matrix = confusion_matrix(y_test, y_pred)
                    
                    st.success("Model trained successfully!")
                    
                    # Save model
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    joblib.dump(rf, 'models/random_forest_model.pkl')
                    st.success("Model saved to disk.")
        
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display results if model exists
    if st.session_state.model is not None:
        # Display metrics in a dashboard style
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <h2>{st.session_state.accuracy:.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Precision</h3>
                    <h2>{st.session_state.precision:.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Recall</h3>
                    <h2>{st.session_state.recall:.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <h2>{st.session_state.f1:.4f}</h2>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        # Feature importance plot
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        st.session_state.feature_importances.plot(kind='bar', ax=ax)
        plt.title('Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            st.session_state.confusion_matrix, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            ax=ax
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Prediction interface
        st.subheader("Make Predictions")
        st.markdown("Enter values to predict student grade:")
        
        # Create input fields for all features
        if st.session_state.features is not None:
            col1, col2 = st.columns(2)
            
            # Store feature inputs
            feature_inputs = {}
            
            for i, feature in enumerate(st.session_state.features):
                with col1 if i % 2 == 0 else col2:
                    # Get the mean of the feature from the dataset
                    mean_value = st.session_state.df[feature].mean()
                    feature_inputs[feature] = st.number_input(
                        f"{feature}", 
                        value=float(mean_value),
                        step=0.1
                    )
            
            if st.button("Predict Grade"):
                if st.session_state.model is not None:
                    # Create input array for prediction
                    input_data = pd.DataFrame([feature_inputs])
                    
                    # Make prediction
                    prediction = st.session_state.model.predict(input_data)
                    
                    # Decode the prediction
                    predicted_grade = st.session_state.label_encoder.inverse_transform(prediction)[0]
                    
                    # Display prediction
                    st.success(f"Predicted Grade: {predicted_grade}")
    else:
        # Display placeholder if no model is trained
        st.info("Upload a dataset and train the model to view results.")
        
        # Sample image or description
        st.markdown("""
        ### How to use this dashboard:
        1. Upload your Excel dataset using the sidebar uploader
        2. Adjust model parameters as needed
        3. Click "Train Model" to build the Random Forest classifier
        4. View model performance metrics and visualizations
        5. Make predictions with the interactive form
        """)

if __name__ == "__main__":
    main()
