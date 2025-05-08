# 📈 Amazon Stock Prediction and Natural Language Q&A App
This project uses a deep learning LSTM model to predict Amazon stock prices. It features a FastAPI backend, a Streamlit frontend, and an LLM-powered agent to answer natural language questions about the predicted stock prices.

## 🔧 Features
🧠 LSTM-based stock price prediction trained on historical Amazon stock data.

📁 Upload support for both .csv and .npy formats.

🧮 Prediction view in both table and line chart format.

💬 LLM Agent Q&A: Ask questions like:

    "What is the maximum predicted price in the first 20 days?"

    "Which day had the lowest predicted price?"

    "Plot the trend of predicted prices."

## 🚀 Tech Stack
  Frontend: Streamlit

  Backend: FastAPI

  Machine Learning: TensorFlow LSTM model

  LLM Agent: LangChain with OpenAI's GPT-3.5 Turbo

  Others: Pandas, NumPy, Pickle, Matplotlib

## 🧪 Example Prompts for Agent
Use the "Agent" tab in the Streamlit app to ask natural language questions like:
  
    "What is the average predicted price?"
    
    "Which 5 days had the highest predicted prices?"
    
    "Plot the predicted stock prices."
    
    "Find the percent change day over day."

## 📊 Model Training
The model was trained on the historical closing prices of Amazon using a 2-layer LSTM network, with MinMax scaling. It was saved in .keras format and integrated into the backend.

You can retrain the model using the provided Colab notebook.

## 📹 Demo
https://github.com/user-attachments/assets/7a95eb85-58c7-4576-aa52-178d0e5adb86





