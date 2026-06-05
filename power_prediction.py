import streamlit as st
import pandas as pd
import pickle

scaler = pickle.load(open("scaler.pkl", "rb"))
rf_model = pickle.load(open("rf-model.pkl", "rb"))

def main ():
    st.title("Household power prediction")
    st.write("Enter the required information to predict the Global active power")

    date_time = st.datetime_input("Date and time")
    reactive_power = st.number_input("Global reactive power")
    voltage = st.number_input("Voltage", min_value=223, max_value=260)
    intensity = st.number_input("Global intensity")
    sub_metering_1 = st.number_input("Sub metering 1")
    sub_metering_2 = st.number_input("Sub metering 2")
    sub_metering_3 = st.number_input("Sub metering 3")

    dico = [{
        "DateTime": date_time,
        "Global_reactive_power": reactive_power,
        "Voltage": voltage,
        "Global_intensity": intensity,
        "Sub_metering_1": sub_metering_1,
        "Sub_metering_2": sub_metering_2,
        "Sub_metering_3": sub_metering_3
    }]

    data = pd.DataFrame(dico)

    data["DateTime"] = pd.to_datetime(data["DateTime"], dayfirst=True)
    data['day'] = data['DateTime'].dt.day
    data['year'] = data['DateTime'].dt.year
    data['Day_of_week'] = data['DateTime'].dt.dayofweek
    data['Week_of_year'] = data['DateTime'].dt.isocalendar().week
    data['Hour'] = data['DateTime'].dt.hour
    data['Minute'] = data['DateTime'].dt.minute
    data.drop(columns=["DateTime"], inplace=True)

    if st.button("Predict"):
        scaled_data = scaler.transform(data)
        predicted_power = rf_model.predict(scaled_data)

        success_msg = "Predicted global active power = " + str(predicted_power) + " W"
        st.success(success_msg)

if __name__ == '__main__':
    main()
