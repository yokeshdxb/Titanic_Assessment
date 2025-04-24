import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model
with open('titanic_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor ")

# 👇 Collect user inputs
age = st.slider("Age", 0, 100, 25)
fare = st.number_input("Fare", min_value=0.0)
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Other"])

# 👇 Feature Engineering (derived from inputs)
sex = 1 if sex == "male" else 0
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0
fare_per_class = fare / pclass if pclass != 0 else 0

# Title one-hot
title_mr = 1 if title == "Mr" else 0
title_mrs = 1 if title == "Mrs" else 0
title_miss = 1 if title == "Miss" else 0
title_rare = 1 if title in ["Dr", "Master", "Other"] else 0

# Embarked one-hot
embarked_c = 1 if embarked == "C" else 0
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

# AgeBand one-hot (example logic, adjust bins as per your training)
ageband_teen = 1 if 13 <= age < 20 else 0
ageband_adult = 1 if 20 <= age < 60 else 0
ageband_senior = 1 if age >= 60 else 0
ageband_unknown = 1 if age is None else 0  # Optional, depending on how model was trained

# Deck one-hot: You may not get this from user input, so default to 0
deck_b = deck_c = deck_d = deck_e = deck_f = deck_g = deck_t = 0

# 👇 Create Input DataFrame
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "FamilySize": family_size,
    "IsAlone": is_alone,
    "FarePerClass": fare_per_class,
    "Title_Miss": title_miss,
    "Title_Mr": title_mr,
    "Title_Mrs": title_mrs,
    "Title_Rare": title_rare,
    "Embarked_C": embarked_c,
    "Embarked_Q": embarked_q,
    "Embarked_S": embarked_s,
    "AgeBand_Teen": ageband_teen,
    "AgeBand_Adult": ageband_adult,
    "AgeBand_Senior": ageband_senior,
    "AgeBand_Unknown": ageband_unknown,
    "Deck_B": deck_b,
    "Deck_C": deck_c,
    "Deck_D": deck_d,
    "Deck_E": deck_e,
    "Deck_F": deck_f,
    "Deck_G": deck_g,
    "Deck_T": deck_t
}])

# 👇 Make sure column order matches training
input_df = input_df[model.feature_names_in_]

# 👇 Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
    # Show the predicted class (0 = Not Survived, 1 = Survived)
    prediction = model.predict(input_df)[0]
    st.subheader("Prediction:")
    st.success("Survived" if prediction == 1 else "Did Not Survive")

    # Show model prediction confidence
    st.subheader("Prediction Confidence:")
    probabilities = model.predict_proba(input_df)[0]
    st.write(f"Not Survived (0): {probabilities[0]:.2f}")
    st.write(f"Survived (1): {probabilities[1]:.2f}")

    # Optional: show model classes just for verification/debugging
    st.caption(f"Model Classes: {model.classes_}")


