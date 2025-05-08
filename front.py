import streamlit as st
import requests
import pandas as pd


st.title("stocks prediction and trading app")


if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    

option = st.selectbox(
    'Select an option',
    ('Upload', 'Prediction','Table','Chart','Agent')
)

if option == 'Upload':
    uploaded_file = st.file_uploader("please upload a file")
    if uploaded_file is not None:
        response = requests.post("http://127.0.0.1:8000/upload", files={"file": uploaded_file})
        if response.status_code == 200:
            st.success("File uploaded successfully!")
            st.write(response.json())
        else:
            st.error("Failed to upload file. Please try again.")
            
elif option == 'Prediction':
    response = requests.post("http://127.0.0.1:8000/predict")
    if response.status_code == 200:
        prediction = response.json()
        st.session_state.prediction = prediction.get('predictions') 
        st.write("Prediction:", prediction)
    else:
        st.error("Failed to get prediction. Please try again.")
elif option == 'Table':
    if st.session_state.prediction is not None:
        try:
            # Extract the numeric values from the list of dicts
            values = [item["predicted_price"] for item in st.session_state.prediction]
            df = pd.DataFrame({'Predicted Price': values})
            st.table(df)
        except Exception as e:
            st.error(f"Failed to display prediction table: {e}")
    else:
        st.warning("No prediction data found. Please run prediction first.")
elif option == 'Chart':
    if st.session_state.prediction is not None:
        try:
            # Extract the numeric values from the list of dicts
            values = [item["predicted_price"] for item in st.session_state.prediction]
            df = pd.DataFrame({'Predicted Price': values})
            max_rows = len(df)
            num_rows = st.slider("Select number of rows to plot", min_value=1, max_value=max_rows, value=min(10, max_rows), key="chart_slider")
            st.line_chart(df[:num_rows])
        except Exception as e:
            st.error(f"Failed to display prediction table: {e}")
    else:
        st.warning("No prediction data found. Please run prediction first.")
elif option == 'Agent':
    prompt = st.text_input("Enter your prompt")
    if st.button("Submit"):
        response = requests.post("http://127.0.0.1:8000/agent", params={"prompt": prompt})
        if response.status_code == 200:
            agent_response = response.json()
            st.write(agent_response.get('response'))
        else:
            st.error(f"Failed to get agent response: {response.status_code} - {response.text}")
    


