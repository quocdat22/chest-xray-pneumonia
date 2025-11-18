# 🫁 Pneumonia Detection from Chest X-Ray Images

This project develops a deep learning CNN model to classify chest X-ray images into 2 classes: **NORMAL** (healthy lungs) and **PNEUMONIA** (pneumonia-affected lungs).

## 🎯 Project Objectives

- ✅ Build and train an efficient baseline CNN model
- ✅ Achieve high accuracy in pneumonia detection
- ✅ Implement Grad-CAM to explain model decisions
- ✅ Perform in-depth analysis of Precision, Recall, and F1-Score metrics
- ✅ Handle data imbalance using Class Weights

## 📊 Dataset

### Data Source
- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: ~5,800 X-ray images
- **Format**: JPEG grayscale, 224×224 pixels size
- **Classes**: 2 classes (NORMAL vs PNEUMONIA)
- **Dataset Author**: Paul Mooney

### Data Distribution

#### Original Dataset (Before Train/Val Split)

| Split/Category | Train | Val | Test | **Total** |
|:---|:---|:---|:---|:---|
| **NORMAL** | 1,341 | 8 | 234 | **1,583** |
| **PNEUMONIA** | 3,875 | 8 | 390 | **4,273** |
| **Total** | **5,216** | **16** | **624** | **5,856** |

#### After Train/Val Re-split (Final Distribution)

| Split/Category | Train | Val | Test | **Total** |
|:---|:---|:---|:---|:---|
| **NORMAL** | 1,214 | 135 | 234 | **1,583** |
| **PNEUMONIA** | 3,494 | 389 | 390 | **4,273** |
| **Total** | **4,708** | **524** | **624** | **5,856** |

**Key Changes:**
- ✅ Increased validation set from 16 to 524 images for better model evaluation
- ✅ Maintained test set at 624 images for consistent performance assessment
- ✅ Redistributed training set to 4,708 images with balanced representation
- ✅ Better validation set size helps detect overfitting more reliably

![Data Distribution](asset/phan_bo.png)

### Handling Data Imbalance

Using **Class Weights** to balance the 2 classes:
- **NORMAL (Class 0)**: 1.939
- **PNEUMONIA (Class 1)**: 0.674

This method automatically balances the influence of each class during training without losing data.

## 🧠 Model Architecture

### Baseline CNN
The model consists of:

**4 Conv Blocks** (each block):
- 2 × Conv2D layers (32 → 64 → 128 → 256 filters)
- BatchNormalization (output normalization)
- MaxPooling2D (2×2) - dimension reduction
- Dropout (0.25) - prevent overfitting

**Dense Layers**:
- Flatten - convert from 2D to 1D
- Dense(512, relu) + BatchNorm + Dropout(0.5)
- Dense(256, relu) + BatchNorm + Dropout(0.5)
- Dense(1, sigmoid) → Output (0 = NORMAL, 1 = PNEUMONIA)

### Model Parameters

| Attribute | Value |
|-----------|-------|
| **Input Shape** | 224 × 224 × 1 (grayscale) |
| **Total Parameters** | 27,000,801 |
| **Batch Size** | 32 |
| **Epochs Trained** | 42 |
| **Optimizer** | Adam (learning rate = 0.001) |
| **Loss Function** | Binary Crossentropy |
| **Early Stopping** | Yes (patience=10 on val_auc) |
| **Regularization** | Dropout + BatchNormalization |

## 📈 Training Results & Evaluation

### Performance on Test Set

| Metric | Value |
|--------|-------|
| **Accuracy** | 85.74% |
| **Precision** | 82.65% |
| **Recall** | 97.69% |
| **AUC** | 0.9516 |
| **F1-Score** | 0.8954 |

### Confusion Matrix

![Confusion Matrix](asset/confusion_matrix.png)

Confusion matrix shows:
- **True Negatives (TN)**: Number of NORMAL images correctly predicted
- **True Positives (TP)**: Number of PNEUMONIA images correctly predicted
- **False Positives (FP)**: Number of NORMAL images incorrectly predicted as PNEUMONIA
- **False Negatives (FN)**: Number of PNEUMONIA images incorrectly predicted as NORMAL (very few - only 2.31%)

### Detailed Metric Explanations

**📊 Accuracy (Overall Accuracy)**
- Ratio of correct predictions to total predictions
- **85.74%** = Model correctly predicts 85.74% of test cases

**✅ Precision (Positive Predictive Value)**
- Among images the model predicts as "PNEUMONIA", **82.65%** truly have pneumonia
- **Meaning**: When the model alerts "pneumonia", you can trust it 82.65%
- **Application**: Avoids excessive false alarms

**🔍 Recall (Sensitivity)**
- Among images truly with "PNEUMONIA", the model detects **97.69%**
- **Meaning**: The model rarely misses actual cases (only misses ~2.31%)
- **Important in Healthcare**: High recall reduces the risk of missing diseases
- **Trade-off**: To achieve high recall, the model must be more "lenient", resulting in some false alerts (lower precision)

**🎯 AUC (Area Under Curve)**
- **0.9516** indicates the model has excellent ability to distinguish between 2 classes
- Values closer to 1.0 are better

**⚖️ F1-Score**
- **0.8954** is the harmonic mean of Precision and Recall
- Provides a balanced assessment of model performance
- Suitable when considering both metrics equally

## 📈 ROC Curve & AUC Analysis

### ROC Curve (Receiver Operating Characteristic)
The ROC curve displays the balance between **True Positive Rate (Recall)** and **False Positive Rate** as the prediction threshold changes.

![ROC Curve - AUC = 0.9516](asset/ROC_curve.png)

### ROC Curve Explanation

**📊 AUC (Area Under Curve) = 0.9516**
- **Meaning**: Model has a **95.16%** probability of ranking a PNEUMONIA image higher than a NORMAL image
- **Excellent Value**: 
  - 0.5 = Random (no better than chance)
  - 0.7 - 0.8 = Good
  - 0.8 - 0.9 = Very Good
  - 0.9 - 1.0 = Excellent ✓

**🎯 Optimal Point**
- Optimal point is marked on the curve (optimal threshold ≈ 0.946)
- At this point, the model achieves the best balance between:
  - TPR (True Positive Rate) = High Recall
  - FPR (False Positive Rate) = Low False Alerts

**📍 Diagonal Line (Random Classifier)**
- The red dashed diagonal represents a random classifier (AUC = 0.5)
- Our model lies **well above the diagonal** ✓ → Superior performance

### Healthcare Application
- **High AUC** → Model distinguishes NORMAL and PNEUMONIA excellently
- **Disregards False Positive Rate** → Can be used when high recall is needed
- **Suitable for imbalanced data** → Not affected by class imbalance

## 📦 Training Techniques

### Early Stopping & Learning Rate Reduction
- **Early Stopping**: Stop training when `val_auc` doesn't improve for 10 consecutive epochs
- **ReduceLROnPlateau**: Reduce learning rate when loss plateaus
- **ModelCheckpoint**: Automatically save best model based on highest val_auc

### Data Augmentation
- Rotation ±10 degrees
- Width/Height shift: ±10%
- Shear: ±10%
- Zoom: ±20%
- Horizontal flip: Disabled (don't flip, medical X-rays must maintain orientation)

This technique helps the model generalize better and prevents overfitting on small training datasets.

## 🔍 Grad-CAM: Explaining Model Decisions

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is a technique to visualize regions of an image that the model focuses on to make decisions.

### Significance
- Helps understand where the model "looks"
- Identifies important medical indicators
- Increases confidence when applying model in practice

### Results
The `Grad_CAM.ipynb` notebook displays:
- Heatmap of important regions on PNEUMONIA images
- Helps doctors confirm model decisions
- Model focuses on areas showing disease signs

![Example Prediction](asset/grad-cam.png)

## 📊 Precision vs Recall Analysis

### Trade-off Between 2 Metrics

**Precision ↑ (High Accuracy)**
- Model is "conservative" → predicts PNEUMONIA only when very confident
- Few false alarms ✓
- But misses many disease cases ✗

**Recall ↑ (High Sensitivity)**
- Model is "lenient" → predicts PNEUMONIA if there's possibility
- Detects most disease cases ✓
- But produces many false alarms ✗

### Choice in Healthcare

**In disease detection applications, Recall is prioritized over Precision**

Why?
- **Cost of missing disease**: Very high (patient doesn't receive treatment)
- **Cost of false alert**: Lower (patient can get additional tests)

**This model achieves:**
- Recall = 97.69% ✓ (Detects nearly all disease cases)
- Precision = 82.65% ✓ (Controlled false alerts)
- F1-Score = 0.8954 ✓ (Good balance)


## 🚀 Quick Start Guide

### 1️⃣ Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Web Application

```bash
# Start Streamlit app
streamlit run app.py
```
The application will open at `http://localhost:8501`

### 3️⃣ Explore Notebooks

Open Jupyter Notebooks in the `notebooks/` folder:
- **`notebook.ipynb`** - Train CNN model from scratch
- **`Grad_CAM.ipynb`** - Visualize Grad-CAM (explain decisions)
- **`AUC.ipynb`** - Analyze ROC Curve & AUC
- **`pre_rec.ipynb`** - Analyze Precision vs Recall
- **`push_model2hf.ipynb`** - Push model to Hugging Face

## 💡 Key Points & Conclusion

### 1. Model Performance
✅ **Very High Recall (97.69%)** → Detects nearly all disease cases  
✅ **Good Precision (82.65%)** → Controlled false alerts  
✅ **Superior AUC (0.9516)** → Excellent class discrimination ability  
✅ **Balanced sensitivity & specificity** → Suitable for healthcare

### 2. Handling Data Imbalance
✅ **Class Weights effective** → Automatically balances 2 classes  
✅ **Preserves data** → No information loss  
✅ **Suitable for healthcare context** → Uses all clinical cases

### 3. Regularization & Overfitting Prevention
✅ **Dropout + BatchNormalization** → Prevents overfitting  
✅ **Early Stopping** → Stops at optimal point (epoch 42)  
✅ **Data Augmentation** → Improves generalization  

### 4. Model Explainability
✅ **Grad-CAM visualization** → Explains model decisions  
✅ **Precision-Recall analysis** → Understands trade-offs  
✅ **Transparency** → Trust model in healthcare

## 🔄 Project Workflow

**Data Preparation** → **Model Building** → **Training** → **Evaluation** → **Analysis** → **Deployment**

1. **Data Preparation** (notebook.ipynb)
   - Load Kaggle dataset
   - Split train/val 9:1
   - Analyze and visualize

2. **Building & Training** (notebook.ipynb)
   - Design CNN architecture
   - Compile with healthcare metrics
   - Training with class weights

3. **Evaluation & Analysis** (notebook.ipynb, Grad_CAM.ipynb, pre_rec.ipynb)
   - Test set evaluation
   - Confusion matrix
   - Grad-CAM visualization
   - Precision/Recall trade-off

## 📚 References

### Dataset
- [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Original Research Paper](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)

### CNN & Deep Learning
- [Convolutional Neural Networks: Architectures, Mechanisms, and Applications](https://arxiv.org/abs/2010.07468)
- [A Guide to Convolutional Neural Networks](https://arxiv.org/abs/1808.04752)
- [VGG Networks: Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### Model Interpretability
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02055)
- [Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://arxiv.org/abs/1506.02390)

### Framework & Tools
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Keras API Reference - Class Weights](https://keras.io/api/models/sequential/#fit)
- [Scikit-learn: Machine Learning Library](https://scikit-learn.org/)

## ⚠️ Important Disclaimer

### 🔴 Disclaimer
This model is developed for **educational and research purposes only**.  
**Should NOT be used directly for real medical diagnosis**.  
Any medical decision must be confirmed by trained medical professionals.

### 📌 Model Limitations
- Only trained on Kaggle dataset
- Fixed image size 224×224 pixels
- Only binary classification (NORMAL vs PNEUMONIA)
- May not generalize well to data from other hospitals

### ✅ Safe Usage Guidelines
- **Use as a decision support tool**, not a replacement for doctors
- **Always combine** with expert clinical diagnosis
- **Check Confidence Score** before application
- **Specially focus on** False Negatives (missed diseases)

### 🏥 Usage Recommendations
1. Treat model as a "second opinion" tool
2. When model predicts "NORMAL" with Confidence < 80% → Recommend re-examination
3. When model predicts "PNEUMONIA" → Require doctor confirmation
4. Record all results in patient records

## 📝 Project Information

- **Creation Date**: November 18, 2025
- **Model Timestamp**: 20251118_091549
- **Purpose**: Education & Research
- **Dataset**: Kaggle Chest X-Ray Images (Pneumonia)
- **Framework**: TensorFlow/Keras
- **GPU**: NVIDIA P100 (if available)

---

**"Prevention is better than cure" - This model is a support tool, not a doctor replacement** 🏥
