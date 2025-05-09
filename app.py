import streamlit as st
import pickle
import numpy as np

# Load the trained calories prediction model
loaded_model = pickle.load(open(r'C:\Users\shara\OneDrive\Desktop\calories burnt\trained_model.sav', 'rb'))


# Function for prediction
def calories_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return f'Estimated Calories Burned: {prediction[0]:.2f} kcal'

def main():
    st.title('Calories Burned Prediction App')

    # Input fields matching the model's expected feature order
    gender = st.selectbox('Gender', ['Male', 'Female'])  # 1 for Male, 0 for Female
    age = st.text_input('Age')
    height = st.text_input('Height (cm)')
    weight = st.text_input('Weight (kg)')
    duration = st.text_input('Exercise Duration (minutes)')
    heart_rate = st.text_input('Heart Rate (bpm)')
    body_temp = st.text_input('Body Temperature (°C)')

    # Encode gender
    gender_encoded = 1 if gender == 'Male' else 0

    result = ''
    if st.button('Predict Calories Burned'):
        try:
            input_data = [
                gender_encoded,
                float(age),
                float(height),
                float(weight),
                float(duration),
                float(heart_rate),
                float(body_temp)
            ]
            result = calories_prediction(input_data)
        except ValueError:
            result = "Please enter valid numeric values for all fields."

    st.success(result)

if __name__ == '__main__':
    main()
