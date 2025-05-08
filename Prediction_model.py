import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model



def predict_stock_price(model,data):
    test_predictions = model.predict(data).flatten()
    test_results = pd.DataFrame(data={'test Predictions':test_predictions})
    return test_results

def postprocess_data(data,scaler):
    inverse_scaled = scaler.inverse_transform(data.values.reshape(-1, 1)).flatten()
    df = pd.DataFrame(inverse_scaled, columns=["predicted_price"])
    return df

def scaled(data, scaler):
    # Ensure data is 2D for scaling
    data_reshaped = data.values.reshape(-1, 1)  # Reshape into (n_samples, 1)
    
    # Apply the scaler
    scaled_data = scaler.transform(data_reshaped)
    
   
    scaled_data = scaled_data.reshape(693, 60, 1)
    return scaled_data



    
