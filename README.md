Comparative Analysis of Ensemble Learning Architectures and Explainable AI for Early CVD Prediction                                                                                                         
Authors: Siya Bhagat, Aryan Walia | Chitkara University, Patiala                       
Paper ID: IJRASET80940 (Submitted for Publication)
  
**Overview** 

Compares ML models (Logistic Regression, SVM, Random Forest, XGBoost) for early cardiovascular disease prediction using the Cleveland Heart Disease dataset. Integrates SHAP for explainability and SMOTE for   
class balancing. 

**Results**                                                                                                                                                                                                    
  | Model | Accuracy | ROC-AUC | F1-Score | 
  |-------|----------|---------|----------|
  | XGBoost | **91.4%** | **0.94** | **0.92** |
  | Random Forest | 88.7% | 0.91 | 0.88 |
  | SVM | 85.5% | 0.89 | 0.86 |
  | Logistic Regression | 81.2% | — | — |
  
 **Pipeline**                                                                                                                                                                                                    
  Data Loading → KNN Imputation → EDA → Z-Score Normalization → Lasso/RFE Feature Selection → SMOTE → 10-Fold CV Training → SHAP Explainability 
  
**Repository Structure** 
  ├── cvd_prediction_pipeline_executed.ipynb   # Main notebook                                                          ├── heart_disease.csv             # Cleveland Heart Disease dataset
  ├── Submission_screenshot.jpeg    # Proof of paper submission
  └── README.md 
  
**Setup & Run** 
 bash 
 pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn                                jupyter notebook cvd_prediction_pipeline.ipynb  
 
 **Dataset**
 
 Cleveland Heart Disease dataset — 303 patients, 13 features, binary target (0 = no disease, 1 = disease).             Missing values in ca (4) and thal (2) handled via KNN Imputation (k=5). 
 
 **Key Findings**
  - XGBoost achieves 91.4% accuracy and 0.94 ROC-AUC 
  - Top SHAP features: age, thalach (max heart rate), oldpeak (ST depression)
  - SHAP explanations increase clinician Weight of Advice from 0.50 → 0.73                                            
