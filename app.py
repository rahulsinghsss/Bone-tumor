import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import CancerDetectionModel
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Constants
MODEL_THRESHOLDS = {
    'mlp': 0.10,
    'svm': 0.90,
    'random_forest': 0.50
}

MODEL_INTERPRETATIONS = {
    'mlp': 'Early detection focused (high sensitivity)',
    'svm': 'High precision focused (high specificity)',
    'random_forest': 'Balanced detection approach'
}

@st.cache_resource
def load_models():
    """Load all models with caching"""
    models = {}
    model_types = ['random_forest', 'svm', 'mlp']
    
    for model_type in model_types:
        model_path = f"models/{model_type}_model.pth"
        if os.path.exists(os.path.splitext(model_path)[0] + '.joblib'):
            models[model_type] = CancerDetectionModel.load_model(model_path)
    
    return models

@st.cache_data
def load_results():
    """Load evaluation results with caching"""
    try:
        with open('results/model_evaluation.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def create_radar_chart(metrics, model_name):
    """Create radar chart with caching"""
    metrics_data = {
        'Confidence Threshold': MODEL_THRESHOLDS[model_name.lower()],
        'Accuracy': metrics['test']['accuracy'],
        'Precision': metrics['test']['precision'],
        'Recall': metrics['test']['recall'],
        'F1 Score': metrics['test']['f1'],
        'AUC': metrics['test']['auc']
    }
    
    fig = go.Figure(data=go.Scatterpolar(
        r=list(metrics_data.values()),
        theta=list(metrics_data.keys()),
        fill='toself',
        name=model_name
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    return fig

@st.cache_data
def create_comparison_chart(results):
    """Create comparison chart with caching"""
    models = list(results.keys())
    
    fig = go.Figure()
    
    # Add confidence threshold bars
    fig.add_trace(go.Bar(
        name='Confidence Threshold',
        x=models,
        y=[MODEL_THRESHOLDS[model.lower()] * 100 for model in models],
        text=[f'{MODEL_THRESHOLDS[model.lower()]*100:.0f}%' for model in models]
    ))
    
    # Add performance metrics
    for model in models:
        fig.add_trace(go.Bar(
            name=f'{model} Test Metrics',
            x=models,
            y=[results[model]['test']['accuracy'] * 100,
               results[model]['test']['f1'] * 100,
               results[model]['test']['auc'] * 100],
            text=[f'{v*100:.1f}%' for v in [
                results[model]['test']['accuracy'],
                results[model]['test']['f1'],
                results[model]['test']['auc']
            ]]
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score (%)',
        yaxis_range=[0, 100],
        barmode='group'
    )
    return fig

@st.cache_data
def create_confidence_gauge(probability, model_name):
    """Create confidence gauge with caching"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{model_name} Confidence"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    return fig

@st.cache_data
def resize_image(image_bytes, max_size=200):
    """Resize image while maintaining aspect ratio with caching"""
    image = Image.open(image_bytes).convert('RGB')
    ratio = max_size / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    return image.resize(new_size, Image.Resampling.LANCZOS)

@st.cache_data
def create_stats_dataframe(results):
    """Create statistics DataFrame with caching"""
    return pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'Total Images': [
            results['dataset_stats']['train']['total'],
            results['dataset_stats']['valid']['total'],
            results['dataset_stats']['test']['total']
        ],
        'Cancer Cases': [
            results['dataset_stats']['train']['cancer'],
            results['dataset_stats']['valid']['cancer'],
            results['dataset_stats']['test']['cancer']
        ],
        'Healthy Cases': [
            results['dataset_stats']['train']['healthy'],
            results['dataset_stats']['valid']['healthy'],
            results['dataset_stats']['test']['healthy']
        ]
    })

@st.cache_data
def create_threshold_dataframe(results):
    """Create threshold DataFrame with caching"""
    return pd.DataFrame({
        'Model': ['MLP', 'SVM', 'Random Forest'],
        'Confidence Threshold': ['10%', '90%', '50%'],
        'Best Parameters': [
            str(results['mlp']['best_parameters']),
            str(results['svm']['best_parameters']),
            str(results['random_forest']['best_parameters'])
        ],
        'Interpretation': [
            MODEL_INTERPRETATIONS['mlp'],
            MODEL_INTERPRETATIONS['svm'],
            MODEL_INTERPRETATIONS['random_forest']
        ]
    })

def main():
    st.set_page_config(layout="wide")
    
    # Initialize session state
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    st.title("Bone Tumor Detection System")
    
    # Load results once
    results = load_results()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Detection", "Model Performance"])
    
    with tab1:
        st.write("Upload an X-ray image for bone tumor detection")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Create placeholder for image display
            image_placeholder = st.empty()
            
            # Display uploaded image with smaller size
            resized_image = resize_image(uploaded_file)
            
            # Create two columns for better layout
            col1, col2 = st.columns([1, 2])
            with col1:
                image_placeholder.image(resized_image, caption='Uploaded Image')
            
            if not st.session_state.models:
                st.error("No trained models found. Please contact system administrator.")
                return
            
            # Create placeholder for results
            results_placeholder = st.empty()
            
            with results_placeholder.container():
                st.subheader("Detection Results")
                cols = st.columns(len(st.session_state.models))
                
                for idx, (model_type, model) in enumerate(st.session_state.models.items()):
                    with cols[idx]:
                        try:
                            features = model.extract_combined_features(resized_image)
                            prediction_prob = model.predict_proba(features.reshape(1, -1))[0][1]
                            
                            st.write(f"\n{model_type.upper()} Model:")
                            
                            gauge_fig = create_confidence_gauge(prediction_prob, model_type)
                            st.plotly_chart(gauge_fig, use_container_width=True)
                            
                            threshold = MODEL_THRESHOLDS[model_type]
                            if prediction_prob > threshold:
                                st.warning("Potential tumor detected")
                            else:
                                st.success("No tumor detected")
                        except Exception as e:
                            st.error(f"Error processing with {model_type} model: {str(e)}")
    
    with tab2:
        if results:
            st.subheader("Model Analysis")
            
            # Show dataset statistics
            st.write("Dataset Statistics:")
            stats_df = create_stats_dataframe(results)
            st.dataframe(stats_df, use_container_width=True)
            st.write("---")
            
            # Show model thresholds and interpretations
            st.write("Model Configuration:")
            threshold_df = create_threshold_dataframe(results)
            st.dataframe(threshold_df, use_container_width=True)
            st.write("---")
            
            # Display comparison chart
            comparison_fig = create_comparison_chart(
                {k: v for k, v in results.items() if k != 'dataset_stats'}
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Display individual model performance
            st.subheader("Individual Model Profiles")
            model_profiles = [(k, v) for k, v in results.items() if k != 'dataset_stats']
            cols = st.columns(len(model_profiles))
            
            for idx, (model_type, metrics) in enumerate(model_profiles):
                with cols[idx]:
                    st.write(f"{model_type.upper()} Model")
                    radar_fig = create_radar_chart(metrics, model_type)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.write("Performance Summary:")
                    metrics_container = st.container()
                    with metrics_container:
                        st.write(f"CV Score: {metrics['best_cv_score']:.4f}")
                        st.write(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
                        st.write(f"Test F1 Score: {metrics['test']['f1']:.4f}")
                        st.write(f"Test AUC: {metrics['test']['auc']:.4f}")
            
            # Display detailed metrics table
            st.subheader("Detailed Test Metrics")
            metrics_df = pd.DataFrame({
                model_type: data['test'] 
                for model_type, data in results.items() 
                if model_type != 'dataset_stats'
            }).round(4)
            
            st.dataframe(metrics_df)
            
            # Last training time
            if os.path.exists('results'):
                training_dirs = [d for d in os.listdir('results') if d.startswith('training_')]
                if training_dirs:
                    latest_training = max(training_dirs)
                    try:
                        training_time = datetime.strptime(latest_training.split('_')[1], "%Y%m%d%H%M%S")
                        st.write(f"Last trained: {training_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        pass
        else:
            st.error("No model evaluation results found. Please contact system administrator.")

if __name__ == "__main__":
    main()
