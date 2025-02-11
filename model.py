import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import joblib
import os

warnings.filterwarnings('ignore', category=UserWarning)

class ImageSegmentationModel:
    def __init__(self):
        self.image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        self.segmentation_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")
        self.segmentation_model.eval()
        
    def segment_image(self, image):
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.segmentation_model(**inputs)
            logits = outputs.logits
            return logits.detach()

class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        with torch.no_grad():
            image = self.transform(image).unsqueeze(0)
            features = self.model(image)
            return features.squeeze().detach().numpy()

class CancerDetectionModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15],
                'classifier__min_samples_split': [5, 10],
                'classifier__min_samples_leaf': [2, 4]
            }
        elif model_type == 'svm':
            base_model = SVC(
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
            self.param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__kernel': ['rbf'],
                'classifier__gamma': ['scale', 'auto']
            }
        elif model_type == 'mlp':
            base_model = MLPClassifier(
                max_iter=300,
                random_state=42
            )
            self.param_grid = {
                'classifier__hidden_layer_sizes': [(64,), (128,), (64, 32)],
                'classifier__alpha': [0.0001, 0.001, 0.01],
                'classifier__learning_rate_init': [0.001, 0.01]
            }
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', base_model)
        ])
        
        self.segmentation_model = ImageSegmentationModel()
        self.feature_extractor = FeatureExtractor()
    
    def extract_combined_features(self, image):
        seg_logits = self.segmentation_model.segment_image(image)
        img_features = self.feature_extractor.extract_features(image)
        seg_features = seg_logits.mean(dim=-1).mean(dim=-1).squeeze().numpy()
        
        combined_features = np.concatenate([
            img_features.flatten(),
            seg_features
        ])
        return combined_features
    
    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=cv,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.pipeline = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def save_model(self, path):
        model_dir = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        pipeline_path = os.path.join(model_dir, f"{base_name}.joblib")
        joblib.dump(self.pipeline, pipeline_path)
    
    @classmethod
    def load_model(cls, path):
        model_dir = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        
        pipeline_path = os.path.join(model_dir, f"{base_name}.joblib")
        
        instance = cls()
        instance.pipeline = joblib.load(pipeline_path)
        return instance