import streamlit as st
import pandas as pd
import pickle

with open("models/mlp_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


st.title("UCLA Admission Prediction App")
st.write("Enter student information to predict admission chance.")

gre_score = st.number_input("GRE Score", min_value=0, max_value=340, value=320)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)
university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.number_input("SOP", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
lor = st.number_input("LOR", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
research = st.selectbox("Research", [0, 1])

def prepare_input():
    input_data = {
        "GRE_Score": gre_score,
        "TOEFL_Score": toefl_score,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "University_Rating_2": 0,
        "University_Rating_3": 0,
        "University_Rating_4": 0,
        "University_Rating_5": 0,
        "Research_1": 0
    }

    if university_rating == 2:
        input_data["University_Rating_2"] = 1
    elif university_rating == 3:
        input_data["University_Rating_3"] = 1
    elif university_rating == 4:
        input_data["University_Rating_4"] = 1
    elif university_rating == 5:
        input_data["University_Rating_5"] = 1

    if research == 1:
        input_data["Research_1"] = 1

    return pd.DataFrame([input_data])


if st.button("Predict"):
    try:
        input_df = prepare_input()
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("Prediction: Admitted")
        else:
            st.error("Prediction: Not Admitted")

    except Exception as e:
        st.error(f"Error: {e}")