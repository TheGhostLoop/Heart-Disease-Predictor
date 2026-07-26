import torch
import joblib
import torch.nn as nn
preprocessor = joblib.load("preprocessor.pkl")
# model defining


num_of_features = 19

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_of_features,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.net(x)



model = ANN()
model.load_state_dict(
    torch.load(
        "heart_disease_model.pth",
        map_location=torch.device("cpu")
    )
)
model.eval()

import streamlit as st
import torch
import joblib
import pandas as pd

st.title("❤️ Heart Disease Prediction")
st.header("Artificial Neural Network (PyTorch)")
st.subheader("Enter the patient's details below")

# -------------------------
# Numerical Inputs
# -------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=40,
    step=1
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120,
    step=1
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200,
    step=1
)

max_heart_rate = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150,
    step=1
)

oldpeak = st.number_input(
    "Old Peak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

# -------------------------
# Categorical Inputs
# -------------------------

sex = st.selectbox(
    "Sex",
    [
        "male",
        "female"
    ]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    [
        "asymptomatic",
        "non-anginal pain",
        "atypical angina",
        "typical angina"
    ]
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [
        "no",
        "yes"
    ]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    [
        "normal",
        "hypertrophy",
        "abnormality"
    ]
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    [
        "no",
        "yes"
    ]
)

st_slope = st.selectbox(
    "ST Slope",
    [
        "flat",
        "upsloping",
        "downsloping"
    ]
)

# -------------------------
# Predict
# -------------------------

if st.button("Predict"):

    sample = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "chest pain type":[chest_pain],
        "resting bp s":[resting_bp],
        "cholesterol":[cholesterol],
        "fasting blood sugar":[fasting_bs],
        "resting ecr":[resting_ecg],
        "max heart rate":[max_heart_rate],
        "exercise angina":[exercise_angina],
        "oldpeak":[oldpeak],
        "ST slope":[st_slope]
    })

    # preprocessing
    sample = preprocessor.transform(sample)

    sample = torch.tensor(
        sample,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        logits = model(sample)

        probability = torch.sigmoid(logits).item()

        prediction = 1 if probability >= 0.5 else 0

    if prediction == 1:
        st.error(
            f"⚠️ Heart Disease Detected\n\nConfidence: {probability*100:.2f}%"
        )
    else:
        st.success(
            f"✅ Normal\n\nConfidence: {(1-probability)*100:.2f}%"
        )


# ==========================
# Sidebar
# ==========================

st.sidebar.title("❤️ Heart Disease ANN")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Model Information")

st.sidebar.write("**Model:** Artificial Neural Network")
st.sidebar.write("**Framework:** PyTorch")
st.sidebar.write("**Task:** Binary Classification")

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Performance")

st.sidebar.metric(
    label="Test Accuracy",
    value="85.29%"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Network Architecture")

st.sidebar.code("""
Input Layer
      ↓
Linear (64)
      ↓
ReLU
      ↓
Linear (32)
      ↓
ReLU
      ↓
Linear (1)
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Training Loss")

st.sidebar.image(
    "loss.png",
    use_container_width=True,
    caption="Training Loss Curve"
)

st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Technologies")

st.sidebar.write("""
- Python
- PyTorch
- Streamlit
- Scikit-learn
- Pandas
- NumPy
""")