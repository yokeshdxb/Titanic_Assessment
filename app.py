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




'''
st.title("Titanic Survival Predictor ")

age = st.slider("Age", 0, 100, 25)
fare = st.number_input("Fare", min_value=0.0)
pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
sibsp = st.number_input("SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Dr", "Other"])
# (Add rest of features as needed...)

# Convert to model input format
# Make sure encoding matches your model training!
# You can construct dummy dataframe as required

if st.button("Predict"):
    prediction = model.predict(np.array([[pclass, age, fare, sex,]]))  # Add rest features
    st.success("Survived" if prediction[0] == 1 else "Did not survive")



# Streamlit UI
st.title("Iris Prediction Web App")
st.write("🔍 This app uses a Logistic Regression to predict type of iris.")

# Collect user input
SepalLengthCm = st.number_input("SepalLengthCm", min_value=4.3, max_value=7.9, value=5.8)
SepalWidthCm = st.number_input("SepalWidthCm", min_value=2.0, max_value=4.4, value=3.0)
PetalLengthCm = st.number_input("PetalLengthCm", min_value=1.0, max_value=6.9, value=4.35)
PetalWidthCm = st.number_input("PetalWidthCm", min_value=0.1, max_value=2.5, value=1.3)

# Button to predict
if st.button("Predict Follow-up Requirement"):
    import pandas as pd
    input_df = pd.DataFrame([{
        'Id': 0,  # Dummy ID, since model expects it
        'SepalLengthCm': SepalLengthCm,
        'SepalWidthCm': SepalWidthCm,
        'PetalLengthCm': PetalLengthCm,
        'PetalWidthCm': PetalWidthCm
    }])

    try:
        prediction = model.predict(input_df)[0]
        if prediction == 0:
            st.success("🟢 Iris Setosa !!!")
        elif prediction == 1:
            st.success("🟢 Iris Versicolor !!!")
        elif prediction == 2:
            st.success("🟢 Iris Virginica !!!")
        else:
            st.error("🔴 Unknown Prediction")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
