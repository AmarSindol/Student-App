import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

# method for loading the file
def load_model():
    with open("student_lr_model.pkl",'rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

# passing the data in the format user should give as inputs
def preprocessing_input_data(data,scalar,le):
    data['Extracurricular Activities'] = le.transform([data["Extracurricular Activities"]])
    df =pd.DataFrame([data])
    df_transformed = scalar.transform(df)
    return df_transformed

#predicting the output 
def predict_data(data):
    model,scalar,le = load_model()
    processed_data = preprocessing_input_data(data,scalar,le)
    prediction = model.predict(processed_data)
    return prediction

# create an UI
def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")
    
    hours_studied = st.number_input('Hours Studied',min_value=1,max_value=10,value=5)
    previous_score = st.number_input('Previous Score',min_value=35,max_value=100,value=70)
    extra = st.selectbox('Extracurricular Activities' , ['Yes', 'No'])
    sleep_hours = st.number_input('Sleep Hours',min_value=4,max_value=10,value=7)
    number_of_paper_solved = st.number_input('Number of Question Papers Solved',min_value=0,max_value=10,value=5)
    
    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleep_hours,
            "Sample Question Papers Practiced":number_of_paper_solved
            
        }
        
        prediction = predict_data(user_data)
        st.success(f"Your Predicted Result is {prediction} ")
    
if __name__ == "__main__":
    main()