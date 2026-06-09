import streamlit as st
import pandas as pd
import time
import pickle

scaler = pickle.load(open("components/scaler.pkl", "rb"))
rf_model = pickle.load(open("components/rf-model.pkl", "rb"))

def stream_response (predicted_power):
    """
        Callback function for the write_stream function that makes a live typing animation
        Args:
            predicted_power: (float) the predicted value by the model
        Return:
            the sequence of each word in response_msg
    """

    rounded_value = round(predicted_power[0], 2) # predicted_power[0] because the predicted value is a numpy array

    response_msg = "The predicted global active power is " + str(rounded_value) + " Watt.\n"

    for letter in response_msg:
        yield letter
        time.sleep(0.02)

def main ():
    """
        The user interface on Streamlit
    """

    st.title("🔋 Household power consumption estimation")
    st.markdown(
        ":violet-badge[Artificial Intelligence] :orange-badge[Supervised Learning] :gray-badge[Regression] :blue-badge[Machine Learning]"
    )
    st.subheader("About")
    st.write("The machine learning model that is used to predict the electric power in this app" \
    " was trained with the Individual Household Electric Power Consumption" \
    " dataset from the UCI Machine Learning repository.")
    st.divider()
    st.write("Enter the required information to predict the electric power consumption.")

    date_time = st.datetime_input("Date and time")
    reactive_power = st.number_input("Household global reactive power (in kilowatt)", value=0.436)
    voltage = st.number_input("Voltage (in volt)", value=233.63)
    intensity = st.number_input("Global intensity (in ampere)", value=23)
    st.write("Sub metering 1:")
    sub_metering_1 = st.number_input(
        "Energy sub-metering No. 1"
        "(in watt-hour of active energy)." \
        "It corresponds to the kitchen, containing mainly a dishwasher," \
        "an oven and a microwave (hot plates are not electric but gas powered)"
    )
    sub_metering_2 = st.number_input("Sub metering 2: energy sub-metering No. 2 "
    "(in watt-hour of active energy). " \
    "It corresponds to the laundry room, containing a washing-machine, " \
    "a tumble-drier, a refrigerator and a light.", value=1)
    sub_metering_3 = st.number_input("Sub metering 3: energy sub-metering No. 3 "
    "(in watt-hour of active energy). " \
    "It corresponds to an electric water-heater and an air-conditioner.", value=16)

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

        st.divider()
        st.write_stream(stream_response (predicted_power), cursor="⚡")
        st.divider()

if __name__ == '__main__':
    main()
