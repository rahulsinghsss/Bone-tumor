import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from model import CancerDetectionModel
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torchvision import transforms

def load_and_preprocess_data(data_dir, model_instance, augment=False):
    features = []
    labels = []
    failed_images = []
    
    total_images = sum([len(os.listdir(os.path.join(data_dir, label))) 
                       for label in ['healthy', 'cancer']])
    
    # Define data augmentation transforms (without normalization)
    if augment:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224))
        ])
    
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for label in ['healthy', 'cancer']:
            dir_path = os.path.join(data_dir, label)
            if not os.path.exists(dir_path):
                print(f"Warning: Directory not found - {dir_path}")
                continue
                
            for img_name in os.listdir(dir_path):
                try:
                    img_path = os.path.join(dir_path, img_name)
                    # Load and convert image to RGB
                    image = Image.open(img_path).convert('RGB')
                    
                    # Apply transforms (without normalization)
                    if augment:
                        image = data_transforms(image)
                    
                    # Extract features using the model's feature extractor
                    combined_features = model_instance.extract_combined_features(image)
                    
                    if combined_features is not None:
                        features.append(combined_features)
                        labels.append(1 if label == 'cancer' else 0)
                except Exception as e:
                    failed_images.append((img_path, str(e)))
                pbar.update(1)
    
    if failed_images:
        print("\nWarning: Failed to process the following images:")
        for img_path, error in failed_images:
            print(f"- {img_path}: {error}")
    
    if not features:
        print(f"\nError: No features were extracted from the images in {data_dir}")
        return np.array([]), np.array([])
    
    return np.array(features), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, model_type, phase, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_type} ({phase})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'{model_type}_{phase}_confusion_matrix.png'))
    plt.close()

def evaluate_model(model, X, y, phase):
    predictions = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='weighted')
    auc = roc_auc_score(y, proba)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }, predictions

def train_models():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/training_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize model for feature extraction
    init_model = CancerDetectionModel()
    
    # Load all data first
    print("Loading training data...")
    TRAIN_DIR = "data/train/train"
    TEST_DIR = "data/test"
    
    # Load and preprocess data with augmentation for training set
    X, y = load_and_preprocess_data(TRAIN_DIR, init_model, augment=True)
    X_test, y_test = load_and_preprocess_data(TEST_DIR, init_model, augment=False)
    
    # Check if data was loaded
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data loaded. Check TRAIN_DIR and image files.")
    
    # Split training data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save dataset statistics
    dataset_stats = {
        'train': {'total': len(y_train), 'cancer': int(sum(y_train)), 'healthy': int(len(y_train) - sum(y_train))},
        'valid': {'total': len(y_valid), 'cancer': int(sum(y_valid)), 'healthy': int(len(y_valid) - sum(y_valid))},
        'test': {'total': len(y_test), 'cancer': int(sum(y_test)), 'healthy': int(len(y_test) - sum(y_test))}
    }
    
    model_types = ['random_forest', 'svm', 'mlp']
    results = {'dataset_stats': dataset_stats}
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        model = CancerDetectionModel(model_type=model_type)
        
        # Train model
        best_params, best_cv_score = model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics, val_predictions = evaluate_model(model, X_valid, y_valid, 'validation')
        
        # Evaluate on test set
        test_metrics, test_predictions = evaluate_model(model, X_test, y_test, 'test')
        
        # Generate confusion matrices
        plot_confusion_matrix(y_valid, val_predictions, model_type, 'validation', results_dir)
        plot_confusion_matrix(y_test, test_predictions, model_type, 'test', results_dir)
        
        # Save results
        results[model_type] = {
            'best_parameters': best_params,
            'best_cv_score': float(best_cv_score),
            'validation': val_metrics,
            'test': test_metrics
        }
        
        # Save model
        model.save_model(f"models/{model_type}_model.pth")
        
        # Print results
        print(f"\n{model_type} Results:")
        print(f"Best Parameters: {best_params}")
        print(f"Best CV Score: {best_cv_score:.4f}")
        print(f"Validation Metrics: {val_metrics}")
        print(f"Test Metrics: {test_metrics}")
    
    # Save results
    with open(os.path.join(results_dir, 'model_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    with open('results/model_evaluation.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Generate summary report
    summary_df = pd.DataFrame({
        model_type: {
            'Best CV Score': results[model_type]['best_cv_score'],
            'Test Accuracy': results[model_type]['test']['accuracy'],
            'Test F1': results[model_type]['test']['f1'],
            'Test AUC': results[model_type]['test']['auc']
        } for model_type in model_types
    }).round(4)
    
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'))
    print("\nTraining complete! Results saved to:", results_dir)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    train_models()