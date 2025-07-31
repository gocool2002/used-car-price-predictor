
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load saved model and transformer
with open('ridge_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('poly.pkl', 'rb') as f:
    poly = pickle.load(f)

# --- Brand Bucketing Function ---
super_luxury = ['Rolls-Royce', 'Bentley', 'Ferrari', 'Lamborghini', 'Bugatti', 'McLaren', 'Aston']

luxury = ['Audi', 'BMW', 'Mercedes-Benz', 'Lexus', 'Jaguar', 'Porsche', 'Genesis', 'Land',
    'Maserati', 'INFINITI', 'Alfa', 'Volvo']
premium = ['Volkswagen', 'MINI', 'Subaru', 'Cadillac', 'Lincoln', 'Acura', 'Buick']
economy = ['Toyota', 'Mazda', 'Hyundai', 'Kia', 'Honda', 'Ford', 'Nissan', 'Chevrolet',
    'Mitsubishi', 'Jeep', 'Chrysler', 'RAM', 'GMC', 'Dodge', 'Scion', 'FIAT',
    'Pontiac', 'Saturn', 'Saab', 'Hummer']

def classify_brand(brand):
    brand = brand.strip().lower()
    
    if brand in super_luxury:
        return 'Super Luxury'
    elif brand in luxury:
        return 'Luxury'
    elif brand in premium:
        return 'Premium'
    else:
        return 'Economy'

# --- Fuel Type Bucketing Function ---
def classify_fuel(fuel):
    fuel = fuel.lower()
    if 'diesel' in fuel:
        return 'Diesel'
    elif 'hybrid' in fuel or 'plug' in fuel:
        return 'Hybrid'
    elif 'e85' in fuel or 'flex' in fuel:
        return 'E85/Flex Fuel'
    elif 'gasoline' in fuel or 'petrol' in fuel:
        return 'Petrol'
    else:
        return 'Unknown'

# --- Sidebar UI ---
st.title("🚗 Car Price Prediction Dashboard")

brand = st.selectbox("Select Brand", ['Ford', 'INFINITI', 'Audi', 'Lexus', 'Aston', 'Toyota', 'Lincoln',
       'Land', 'Mercedes-Benz', 'Dodge', 'Jaguar', 'Chevrolet', 'Hyundai',
       'BMW', 'Kia', 'Jeep', 'Bentley', 'MINI', 'Porsche', 'Hummer',
       'Chrysler', 'Acura', 'Volvo', 'Cadillac', 'Maserati', 'Genesis',
       'Volkswagen', 'GMC', 'RAM', 'Nissan', 'Subaru', 'Alfa', 'Ferrari',
       'Scion', 'Mitsubishi', 'Mazda', 'Saturn', 'Honda', 'Bugatti',
       'Lamborghini', 'Rolls-Royce', 'McLaren', 'Buick', 'Lotus', 'FIAT',
       'Pontiac', 'smart', 'Saab'])
year = st.number_input("Manufacturing Year", min_value=1990, max_value=datetime.now().year, value=2018)
mileage = st.number_input("Mileage (in km)", min_value=0, max_value=300000, value=60000, step=1000)
fuel_type = st.selectbox("Fuel Type", ['Gasoline/Petrol', 'Diesel', 'Hybrid', 'EV', 'E85 Flex Fuel'])
accident_flag = st.selectbox("Accident History", ['None', 'Accident Occured'])
transmission = st.selectbox("Transmission Type", ['Automatic', 'Manual'])
color = st.selectbox("Color", ['White', 'Black', 'Silver', 'Red', 'Blue', 'Gray', 'Other'])

# --- Feature Engineering ---
brand_bucket = classify_brand(brand)
fuel_cat = classify_fuel(fuel_type)
current_year = datetime.now().year
age = current_year - year
mileage_per_yr = mileage / age if age > 0 else mileage
log_mileage = np.log1p(mileage)
age_squared = age ** 2
accident_flag_val = 1 if accident_flag == 'Accident Occured' else 0
transmission_flag = 1 if transmission == 'Automatic' else 0

# --- One-hot for brand, fuel, color ---
features = {
    'accident_flag': accident_flag_val,
    'transmission_flag': transmission_flag,
    'log_mileage': log_mileage,
    'mileage_per_yr': mileage_per_yr,
    'age_squared': age_squared,
    
    'brand_cat_Super Luxury': 1 if brand_bucket == 'Super Luxury' else 0,
    'brand_cat_Luxury': 1 if brand_bucket == 'Luxury' else 0,
    'brand_cat_Premium': 1 if brand_bucket == 'Premium' else 0,

    'fuel_cat_Diesel': 1 if fuel_cat == 'Diesel' else 0,
    'fuel_cat_Hybrid': 1 if fuel_cat == 'Hybrid' else 0,
    'fuel_cat_EV': 1 if fuel_cat == 'EV' else 0,
    'fuel_cat_Petrol': 1 if fuel_cat == 'Gasoline/Petrol' else 0
}

# One-hot color columns
for clr in ['blue', 'brown', 'yellow', 'green', 'red', 'silver', 'gray', 'white']:
    features[f'color__{clr}'] = 1 if clr in color.lower() else 0

# Create DataFrame for model input
input_df = pd.DataFrame([features])

# --- Prediction ---
if st.button("Predict Price"):
	try:
		transformed_input = poly.transform(input_df)
		predicted_price = model.predict(transformed_input)[0]
		st.success(f"Estimated Car Price: **${predicted_price:,.2f}**")

	except Exception as e:
        	st.error(f"Error in prediction: {e}")

