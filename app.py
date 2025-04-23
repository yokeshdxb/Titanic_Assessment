import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model
with open('titanic_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor ")

# User inputs
age = st.slider("Age", 0, 100, 25)
fare = st.number_input("Fare", min_value=0.0)
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Other"])

# One-hot encoding for Title
title_mr = title == "Mr"
title_mrs = title == "Mrs"
title_miss = title == "Miss"
title_rare = title in ["Master", "Dr", "Other"]  # Treat these as 'Rare'

# One-hot encoding for Embarked
embarked_c = embarked == "C"
embarked_q = embarked == "Q"
embarked_s = embarked == "S"

# Construct input data for prediction
input_df = pd.DataFrame({
    "Age": [age],
    "Fare": [fare],
    "Pclass": [pclass],
    "Sex": [1 if sex == "male" else 0],  # Assuming male=1, female=0
    "SibSp": [sibsp],
    "Parch": [parch],
    "Title_Mr": [title_mr],
    "Title_Mrs": [title_mrs],
    "Title_Miss": [title_miss],
    "Title_Rare": [title_rare],
    "Embarked_C": [embarked_c],
    "Embarked_Q": [embarked_q],
    "Embarked_S": [embarked_s],
    # Add any other engineered features here
})

# Show for confirmation
#st.write("Input for Prediction:")
#st.dataframe(input_df)

# Predict (assuming model is loaded as `model`)
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")


