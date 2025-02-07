import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the trained Random Forest model and its R² score
model_file = "rf_model.pkl"
r_squared = 0.9862 # Model performance score

# Dataset file path (ensure this is correct)
data_file = "C:/Users/Piyusha/Car Price Prediction/cleandataset.csv"
car_data = pd.read_csv(data_file)

# Streamlit App
st.title("Car Price Prediction")

# Dropdown inputs for categorical data mappings
fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3}
seller_map = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
transmission_map = {'Manual': 0, 'Automatic': 1}
owner_map = {'First Owner': 0, 'Second Owner': 1,
             'Third Owner': 2, 'Fourth & Above Owner': 3,
             'Test Drive Car': 4}

# Input fields for user interaction
selected_model = st.selectbox("Select Car Model", sorted(car_data['name'].unique()))
year = st.selectbox("Select Year", sorted(car_data['year'].unique(), reverse=True))
km_driven = st.number_input("Enter Kilometers Driven", min_value=0)
fuel_type = st.selectbox("Select Fuel Type", list(fuel_map.keys()))
seller_type = st.selectbox("Select Seller Type", list(seller_map.keys()))
transmission = st.selectbox("Select Transmission", list(transmission_map.keys()))
owner = st.selectbox("Select Owner Type", list(owner_map.keys()))

# Retrieve features dynamically based on selected car model
selected_car_data = car_data[car_data['name'] == selected_model]
if not selected_car_data.empty:
    mileage = float(selected_car_data['mileage'].values[0])
    engine = float(selected_car_data['engine'].values[0])
    max_power = float(selected_car_data['max_power'].values[0])
else:
    st.error("Car model details not found in the dataset!")
    mileage, engine, max_power = 0.0, 0.0, 0.0

# Prepare input data for prediction
input_data = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel_map[fuel_type]],
    'seller_type': [seller_map[seller_type]],
    'transmission': [transmission_map[transmission]],
    'owner': [owner_map[owner]],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
})

# Ensure input columns match training data order
columns_order = ['year', 'km_driven', 'fuel', 
                 'seller_type', 'transmission', 
                 'owner', 'mileage', 
                 'engine', 'max_power']
encoded_data = input_data[columns_order]

# Display predictions and model performance
st.subheader("Prediction Result")

if st.button("Predict Price"):
    try:
        # Load the Random Forest model and make predictions
        model = pickle.load(open(model_file, 'rb'))
        prediction = model.predict(encoded_data)[0]
        final_prediction = np.round(prediction)
        
        # Display results with accuracy percentage based on R² score
        accuracy_percentage = r_squared * 100  
        st.success(f"Predicted Car Price: ₹ {final_prediction:,}")
        st.info(f"Model Performance (R²): {accuracy_percentage:.2f}%")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
