# ⚖️ COMPAS Fairness, Bias and Explainability Analysis

## 📌 Project Overview
This project analyzes fairness, bias, and explainability of a machine learning model trained on the COMPAS recidivism dataset.

The project focuses on:
- Fairness evaluation
- Bias detection
- Explainable AI techniques
- Ethical machine learning analysis

---

# 🤖 Model Used
- CatBoost Classifier

---

# 🧠 Techniques Used
- SHAP (Global and Local Explainability)
- LIME (Local Explainability)
- Fairlearn (Bias and Fairness Evaluation)

---

# 📊 Key Results
- Accuracy: ~67%
- Important Features:
  - `priors_count`
  - `decile_score`
  - `age`
- Bias observed across race and gender groups

---

# ⚠️ Ethical Considerations
The inclusion of COMPAS-generated risk scores improved prediction performance; however, these variables may already contain historical or societal bias, potentially propagating unfairness into the machine learning model.

---

# 🚀 Project Features
✔ Feature Importance Analysis  
✔ SHAP Explainability Plots  
✔ LIME Local Explanations  
✔ Fairness Metrics  
✔ Bias Analysis Across Sensitive Groups  
✔ ROC Curve Analysis  
✔ Correlation Heatmap  
✔ Ethical AI Discussion  

---

# 📂 Project Structure

```text
COMPAS-Fairness-Analysis/
│
├── COMPAS_Fairness_Bias_Explainability_Analysis.ipynb
├── compas_model.cbm
├── README.md
├── requirements.txt
```

---

# ▶️ How to Run

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Open Notebook

Run:

```text
COMPAS_Fairness_Bias_Explainability_Analysis.ipynb
```

using:
- Jupyter Notebook
- Google Colab

---

## 3️⃣ Run Inference

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()

model.load_model("compas_model.cbm")

prediction = model.predict(sample_data)

probability = model.predict_proba(sample_data)
```

---

# 📁 Dataset
COMPAS Recidivism Dataset (ProPublica)

---

# ✅ Conclusion
This project demonstrates that in sensitive real-world applications such as criminal justice systems, fairness, transparency, explainability, and ethical AI practices are as important as predictive accuracy.
