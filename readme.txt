access--https://bone-tumor-ypeffyukmcychnn2bfds38.streamlit.app/


The bone tumor detection system is built using a comprehensive tech stack where Streamlit provides the web interface, while PyTorch and Hugging Face Transformers handle deep learning operations. The system employs three distinct classifiers: Random Forest, SVM, and MLP, each optimized for different aspects of tumor detection (balanced, high specificity, and high sensitivity respectively). Feature extraction combines ResNet50's pretrained capabilities with SegFormer's semantic segmentation, creating a robust feature set for analysis. To ensure model reliability, the training process incorporates extensive data augmentation techniques including image rotation, flipping, and color adjustments. The system uses a rigorous validation approach with stratified data splitting and 5-fold cross-validation, while GridSearchCV optimizes model hyperparameters. This architecture balances sophisticated deep learning techniques with traditional machine learning approaches to create a reliable medical imaging analysis tool.


Core Technologies and Frameworks:

Streamlit: Used for creating the web interface
PyTorch: Deep learning framework for feature extraction and segmentation
Scikit-learn: For traditional machine learning models and metrics
Transformers (Hugging Face): For image segmentation
Plotly: For interactive visualizations
Pandas: For data manipulation and analysis
PIL (Python Imaging Library): For image processing
NumPy: For numerical computations


Model Architecture:

Three different classifiers implemented:

Random Forest (balanced detection)
Support Vector Machine (high specificity focus)
Multi-Layer Perceptron (high sensitivity focus)


Each model uses a pipeline with StandardScaler
Features are extracted using both ResNet50 and SegFormer


Feature Extraction Technique:

ResNet50 pretrained on ImageNet: Used for general image feature extraction
SegFormer (nvidia/mit-b0): Used for semantic segmentation features
Combined features approach: Concatenates both ResNet and SegFormer features
Image preprocessing includes resizing to 224x224 and normalization


Training Process:

Data augmentation techniques:

Random rotation (15 degrees)
Random horizontal flip
Random vertical flip
Color jitter (brightness, contrast, saturation, hue)


Train-validation-test split with stratification
5-fold cross-validation during training
GridSearchCV for hyperparameter optimization
