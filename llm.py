from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def create_agent(llm, df):
    agent=create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    return agent