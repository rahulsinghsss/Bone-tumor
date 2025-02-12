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

def load_models():
    models = {}
    model_types = ['random_forest', 'svm', 'mlp']
    
    for model_type in model_types:
        model_path = f"models/{model_type}_model.pth"
        if os.path.exists(os.path.splitext(model_path)[0] + '.joblib'):
            models[model_type] = CancerDetectionModel.load_model(model_path)
    
    return models

def load_results():
    try:
        with open('results/model_evaluation.json', 'r') as f:
            return json.load(f)
    except:
        return None

def create_radar_chart(metrics, model_name):
    # Modified to show both confidence and performance metrics
    thresholds = {
        'mlp': 0.10,
        'svm': 0.90,
        'random_forest': 0.50
    }
    
    metrics_data = {
        'Confidence Threshold': thresholds[model_name.lower()],
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

def create_comparison_chart(results):
    models = list(results.keys())
    metrics = ['Confidence Threshold', 'Accuracy', 'F1 Score', 'AUC']
    
    thresholds = {
        'mlp': 0.10,
        'svm': 0.90,
        'random_forest': 0.50
    }
    
    fig = go.Figure()
    
    # Add confidence threshold bars
    fig.add_trace(go.Bar(
        name='Confidence Threshold',
        x=models,
        y=[thresholds[model.lower()] * 100 for model in models],
        text=[f'{thresholds[model.lower()]*100:.0f}%' for model in models]
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

def create_confidence_gauge(probability, model_name):
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

def resize_image(image, max_size=200):
    """Resize image while maintaining aspect ratio"""
    ratio = max_size / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    return image.resize(new_size, Image.Resampling.LANCZOS)

def main():
    st.set_page_config(layout="wide")
    st.title("Bone Tumor Detection System")
    
    # Load results
    results = load_results()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Detection", "Model Performance"])
    
    with tab1:
        st.write("Upload an X-ray image for bone tumor detection")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Display uploaded image with smaller size
            original_image = Image.open(uploaded_file).convert('RGB')
            resized_image = resize_image(original_image, max_size=200)
            
            # Create two columns for better layout
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(resized_image, caption='Uploaded Image')
            
            # Load models and use resized image for prediction
            models = load_models()
            
            if not models:
                st.error("No trained models found. Please contact system administrator.")
                return
            
            # Make predictions using resized image
            st.subheader("Detection Results")
            
            cols = st.columns(len(models))
            
            for idx, (model_type, model) in enumerate(models.items()):
                with cols[idx]:
                    try:
                        features = model.extract_combined_features(resized_image)
                        prediction_prob = model.predict_proba(features.reshape(1, -1))[0][1]
                        
                        st.write(f"\n{model_type.upper()} Model:")
                        
                        # Display confidence gauge
                        gauge_fig = create_confidence_gauge(prediction_prob, model_type)
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        # Classification result with different thresholds
                        if model_type == 'mlp':
                            threshold = 0.10
                        elif model_type == 'svm':
                            threshold = 0.90
                        else:
                            threshold = 0.50
                            
                        if prediction_prob > threshold:
                            st.warning(f"Potential tumor detected")
                        else:
                            st.success(f"No tumor detected")
                    except Exception as e:
                        st.error(f"Error processing with {model_type} model: {str(e)}")
    
    with tab2:
        if results:
            st.subheader("Model Analysis")
            
            # Show dataset statistics
            st.write("Dataset Statistics:")
            stats_df = pd.DataFrame({
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
            st.dataframe(stats_df, use_container_width=True)
            st.write("---")
            
            # Show model thresholds and interpretations
            st.write("Model Configuration:")
            threshold_df = pd.DataFrame({
                'Model': ['MLP', 'SVM', 'Random Forest'],
                'Confidence Threshold': ['10%', '90%', '50%'],
                'Best Parameters': [
                    str(results['mlp']['best_parameters']),
                    str(results['svm']['best_parameters']),
                    str(results['random_forest']['best_parameters'])
                ],
                'Interpretation': [
                    'Early detection focused (high sensitivity)',
                    'High precision focused (high specificity)',
                    'Balanced detection approach'
                ]
            })
            st.dataframe(threshold_df, use_container_width=True)
            st.write("---")
            
            # Display comparison chart
            comparison_fig = create_comparison_chart(
                {k: v for k, v in results.items() if k != 'dataset_stats'}
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Display individual model performance
            st.subheader("Individual Model Profiles")
            cols = st.columns(len([k for k in results.keys() if k != 'dataset_stats']))
            
            for idx, (model_type, metrics) in enumerate(
                [(k, v) for k, v in results.items() if k != 'dataset_stats']
            ):
                with cols[idx]:
                    st.write(f"{model_type.upper()} Model")
                    radar_fig = create_radar_chart(metrics, model_type)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    # Show model performance summary
                    st.write("Performance Summary:")
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
