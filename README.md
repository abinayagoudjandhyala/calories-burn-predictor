# 🔥 Calories Burn Predictor

A **Streamlit web application** that predicts calories burned during exercise using a machine learning regression model (XGBoost). The model is trained on a fitness dataset that includes metrics like gender, age, height, weight, heart rate, and more.

🔗 **Live App:** [https://calories-burn-predictor.streamlit.app/](https://calories-burn-predictor.streamlit.app/)

---

## 📌 Project Overview

This app helps estimate the number of calories burned based on user-provided input features. It was built to demonstrate machine learning model deployment with an interactive interface using Streamlit.

**Dataset Source:**  
[Kaggle - Exercise and Health Dataset](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)

---

## 🚀 Features

- Clean Streamlit UI for user interaction
- Predicts calories burned using:
  - Gender
  - Age
  - Height (cm)
  - Weight (kg)
  - Duration of exercise (minutes)
  - Heart Rate (bpm)
  - Body Temperature (°C)
- Model trained using XGBoost Regression

---

## 🛠️ Technologies Used

- Python
- Streamlit
- XGBoost
- NumPy
- Pandas
- Scikit-learn
- Google Colab (for model training)

---

## ⚙️ Local Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/abinayagoudjandhyala/calories-burn-predictor.git
cd calories-burn-predictor
````

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 💻 Sample Prediction Code

If you want to test the model directly in Python:

```python
import numpy as np
import pickle

# Load model
model = pickle.load(open('trained_model.sav', 'rb'))

# Sample input
input_data = (1, 68, 190, 94, 29, 105, 40)
input_data_reshaped = np.asarray(input_data).reshape(1, -1)

# Prediction
prediction = model.predict(input_data_reshaped)
print(f'Calories Burned: {prediction[0]:.2f}')
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Abinaya Goud Jandhyala**
📎 [GitHub](https://github.com/abinayagoudjandhyala)
