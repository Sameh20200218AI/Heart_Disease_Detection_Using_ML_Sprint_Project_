
# 💓 Heart Disease Prediction using Machine Learning

A complete end-to-end machine learning project that predicts the presence of heart disease using clinical features from the UCI Cleveland dataset. This project demonstrates the full ML pipeline: from data preprocessing and visualization to feature selection, model training, evaluation, clustering, hyperparameter tuning, and deployment using Gradio.

---



## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Features:** 13 clinical attributes (e.g., age, cholesterol, blood pressure)
- **Target:** Presence of heart disease (0 = No, 1 = Yes)

---

## 🧠 Machine Learning Techniques

- **📦 Preprocessing:** 
  - Handle missing values
  - One-hot encoding for categorical features
  - Feature scaling (standardization)

- **📊 Exploratory Data Analysis:**
  - Histograms, KDE plots, box/violin plots
  - Correlation heatmap
  - Target distribution

- **🧬 Dimensionality Reduction:**
  - Principal Component Analysis (PCA)
  - 2D and 3D interactive visualizations

- **🎯 Feature Selection:**
  - Random Forest feature importance
  - Recursive Feature Elimination (RFE)
  - Chi-square test

- **🤖 Classification Models:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)

- **📈 Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score
  - ROC AUC
  - Confusion Matrix
  - Interactive bar plots for comparison

- **🧪 Hyperparameter Tuning:**
  - Grid Search (Random Forest)
  - Random Search (Logistic Regression, SVM)

- **🧩 Clustering:**
  - K-Means (with elbow method)
  - Agglomerative Clustering (with dendrograms)
  - Silhouette Score analysis

- **🌐 Deployment:**
  - Gradio web app for real-time prediction

---

## 📷 Visualizations

All visualizations are stored in the `plots/` folder:

- Distribution plots of features
- Heatmaps of correlations
- PCA 2D and 3D projections
- Confusion matrices
- Clustering results
- ROC curve comparisons

---

## 🚀 Gradio App

An interactive web app for predicting heart disease risk based on clinical inputs.

### 🟢 Run the app:

```bash
pip install gradio joblib numpy pandas scikit-learn matplotlib seaborn plotly
python 7_gradio_app.py
```

Or use:  
```python
demo.launch(share=True)  # In notebook
```

---

## 📌 Highlights

- Modular Jupyter notebooks for each step of the ML pipeline
- Visual, interpretable feature selection and model comparisons
- Interactive, user-friendly deployment with Gradio
- Clean, production-ready code and saved models

---

## 🔮 Future Improvements

- Add SHAP model explainability
- Deploy app using Hugging Face Spaces or Streamlit Cloud
- Build backend API for wider integration

---

## 🙌 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- Scikit-learn, Pandas, Seaborn, Matplotlib, Plotly, Gradio

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
