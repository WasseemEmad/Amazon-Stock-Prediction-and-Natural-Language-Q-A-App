from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from io import BytesIO
from Prediction_model import predict_stock_price, postprocess_data, scaled
from Prediction_model import load_model
import tempfile
import os
import pickle
from llm import create_agent, llm

app = FastAPI()

temp_file_path = None

uploaded_df_or_array = None

data = None

model = load_model('model2.keras')

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

llm = llm

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global temp_file_path, uploaded_df_or_array

    # Remove old temp file
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        temp_file_path = None
        uploaded_df_or_array = None

    # Determine file type
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        suffix = ".csv"
    elif filename.endswith('.npy'):
        suffix = ".npy"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload .csv or .npy")

    # Save uploaded file to a temporary file
    content = await file.read()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(content)
    temp.close()
    temp_file_path = temp.name

    # Load content
    try:
        if suffix == ".csv":
            df = pd.read_csv(temp_file_path)
            uploaded_df_or_array = df
            if "Unnamed: 0" in uploaded_df_or_array.columns:
                uploaded_df_or_array = uploaded_df_or_array.drop(columns=["Unnamed: 0"])
            print(uploaded_df_or_array.shape) 
            return JSONResponse(content={"message": "CSV uploaded successfully", "columns": df.columns.tolist()})
        else:  # .npy
            arr = np.load(temp_file_path, allow_pickle=True)
            uploaded_df_or_array = arr
            return JSONResponse(content={"message": "NumPy file uploaded successfully", "shape": arr.shape})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")



@app.post("/predict")
async def predict_stock_price_endpoint():
    global uploaded_df_or_array

    if uploaded_df_or_array is None:
        raise HTTPException(status_code=400, detail="No file uploaded. Please upload a .csv or .npy file first.")

    try:
        if isinstance(uploaded_df_or_array, pd.DataFrame):
            data = uploaded_df_or_array
            print("Data shape:", data.shape)
            data = scaled(data,scaler)
            print("Scaled data shape:", data.shape)
            
            predictions = predict_stock_price(model, data)
            df = postprocess_data(predictions, scaler)
            data = df
            with open("data.pkl", "wb") as f:
                pickle.dump(df, f)
            return JSONResponse(content={"message": "Prediction successful","predictions": df.to_dict(orient='records')})
        else:
            data = uploaded_df_or_array      
            predictions = predict_stock_price(model, data)
            df = postprocess_data(predictions, scaler)
            return JSONResponse(content={"message": "Prediction successful","predictions": df.to_dict(orient='records')})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.post("/agent")
async def agent_endpoint(prompt: str):
    global data, llm
    if not os.path.exists("data.pkl"):
        raise HTTPException(status_code=400, detail="No prediction data available. Please run prediction first.")
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)

    try:
        # Create the agent with the LLM and the DataFrame
        agent = create_agent(llm, data)
        # Use the agent to process the prompt
        response = agent.run(prompt)
        return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")
        