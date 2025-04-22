# Import necessary libraries
import os
import json
import requests
import time
import logging
import base64
import re
import matplotlib.pyplot as plt
import pandas as pd
import pydantic
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Dict, Any, Tuple, Annotated, TypedDict, Sequence
from enum import Enum
from uuid import uuid4

# LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool, BaseTool, StructuredTool, Tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolExecutor, ToolInvocation, tools_to_para_tools --> Check this once
from langchain.agents.format_scratchpad import format_to_openai_function_messages

# LangSmith for monitoring
from langsmith import traceable
from langchain_core.tracers.context import tracing_v2_enabled

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINANCIAL_MODELLING_PREP_API_KEY = os.getenv("FINANCIAL_MODELLING_PREP_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
EVENT_REGISTRY_API_KEY = os.getenv("EVENT_REGISTRY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Define the LLM model to use for each agent
def get_llm(model_name = "gemini-1.5-flash", with_tools = None):
    """Gets an LLM Instance --> Can be configured with tools depending on which agent is calling the function"""
    llm = ChatGoogleGenerativeAI(
        model = model_name,
        api_key = GOOGLE_API_KEY,
        temperature = 0.1, # Lower Temperature for a more factual answer (Conversely, higher temperature for more creativity)
        convert_system_message_to_human = True
    )
    if(with_tools == None): 
        return llm
    
    # Bind tools to LLM if they are passed in to the function
    llm_with_tools = llm.bind_tools(with_tools)
    return llm_with_tools


# Shared state that all agents access and modify
class AgentState(TypedDict):
    messages: List[Dict]
    next_agent: str
    current_agent_scratchpad: List[Dict] # Used for monitoring the current agent's tool usage history
    industry: str
    companies: List[str]
    time_period: str
    task_status: str
    sentiment_data: Dict[str, Any]
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any]
    market_data: Dict[str, Any]
    report: Dict[str, Any]

# ----- Tools ----- #

@tool
def get_news_sentiment(tickers = None, topics = None):
    """For a specified ticker / topic --> Get news and sentiment data"""
    
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function" : "NEWS_SENTIMENT",
        "apikey" : ALPHA_VANTAGE_API_KEY
    }

    if tickers:
        params["tickers"] = ",".join(tickers)
    if topics:
        params["topics"] = ",".join(topics)

    response = requests.get(base_url, params = params)
    return response.json

@tool
def get_company_overview(symbol):
    """Get Company Overview data"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function" : "OVERVIEW",
        "symbol" : symbol,
        "apikey" : ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, parmas = params)
    return response.json()

# Check this tool once
@tool 
def get_news_by_ticker(ticker = None, from_date = None, to_date = None):
    """Get news articles using the stock ticker"""
    base_url = "https://api.polygon.io/v2/reference/news"
    params = {
        "apiKey": POLYGON_API_KEY
    }
    if ticker:
        params["ticker"] = ticker
    if from_date:
        params["published_utc.gte"] = from_date
    if to_date:
        params["published_utc.lte"] = to_date
    response = requests.get(base_url, params = params)
    return response.json()

@tool
def get_income_statement(symbol):
    """Get income statement data"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function" : "INCOME_STATEMENT",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params = params)
    return response.json()

@tool
def get_balance_sheet(symbol):
    """Get balance sheet data"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function" : "BALANCE_SHEET",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params = params)
    return response.json()

@tool
def get_cash_flow_statement(symbol):
    """Get cash flow  statement"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "CASH_FLOW",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params = params)
    return response.json()

@tool
def get_earnings_history(symbol):
    """Get earnings history"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "EARNINGS",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params = params)
    return response.json()

@tool
def get_stock_price_and_volume(symbol):
    """Get Stock Price and Volume Data"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FINANCIAL_MODELLING_PREP_API_KEY}"
    response = requests.get(api_endpoint)
    return response.json()

@tool
def get_company_financials(symbol):
    """Get Company Financials"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FINANCIAL_MODELLING_PREP_API_KEY}"
    response = requests.get(api_endpoint)
    return response.json()

@tool
def get_key_metrics(symbol, period = "annual"):
    """Get Key Metrics for a company"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/key_metrics/{symbol}"
    params = {
        "apikey": FINANCIAL_MODELLING_PREP_API_KEY,
        "period": period
    }
    response = requests.get(api_endpoint, params = params)
    return response.json()

@tool
def get_financial_ratios(symbol):
    """Get the Financial Ratios for a company"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
    params = {
        "apikey": FINANCIAL_MODELLING_PREP_API_KEY
    }
    response = requests.get(api_endpoint, params = params)
    return response.json()

@tool
def get_industry_data(industry):
    """Get companies in a specific industry"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        "apikey": FINANCIAL_MODELLING_PREP_API_KEY,
        "sector": industry,
        "limit": 3
    }
    response = requests.get(api_endpoint, params = params)
    return response.json()

@tool 
def get_company_data(company_name, days_back = 10):
    """Get news article for a specific company"""
    api_endpoint = f"https://eventregistry.org/api/v1/article/getArticles"

    end_date = datetime.now()
    start_date = end_date - timedelta(days = days_back)

    payload = {
        "action": "getArticles",
        "keyword": company_name,
        "dateStart": start_date.strftime("%Y-%m-%d"),
        "dateEnd": end_date.strftime("%Y-%m-%d"),
        "articlesCount": 1,
        "resultType": "articles",
        "apikey": EVENT_REGISTRY_API_KEY
    }

    response = requests.post(api_endpoint, json = payload)
    return response.json()

# ----- Prompts (TO DO) ----- #

SUPERVISOR_PROMPT = """Supervisor"""
SENTIMENTS_AGENT_PROMPT = """Sentiments"""
METRICS_AGENT_PROMPT = """Metrics"""
VISUALIZATION__AGENT_PROMPT = """Visualizer"""
SUMMARIZER_PROMPT = """Summarizer"""


# ----- Agent Implementation (TO DO) ----- #

class SupervisorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        ("human", "{input}"),
        ])
    
class SentimentsAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", SENTIMENTS_AGENT_PROMPT),
        ("human", "{input}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()

class MetricsAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", METRICS_AGENT_PROMPT),
        ("human", "{input}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()

class VisualizationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", VISUALIZATION__AGENT_PROMPT),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()

class SummarizerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZER_PROMPT),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | llm | StrOutputParser()

# ----- LangGraph Workflow  ----- #

def create_business_intelligence_graph():
    # Initialize Agents
    supervisor = SupervisorAgent(get_llm())
    market_sentiments = SentimentsAgent(get_llm(with_tools = [get_company_data, get_industry_data, get_news_sentiment, get_company_overview, get_news_by_ticker]))
    financial_metrics = MetricsAgent(get_llm(with_tools = [get_balance_sheet, get_cash_flow_statement, get_income_statement, get_company_financials, get_earnings_history, get_key_metrics])) # Add get_key_metrics() if needed
    visualizer = VisualizationAgent(get_llm())
    summarizer = SummarizerAgent(get_llm())

    # Define Workflow
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("market_sentiments", market_sentiments)
    workflow.add_node("financial_metrics", financial_metrics)
    workflow.add_node("visualizer", visualizer)
    workflow.add_node("summarizer", summarizer)

    # Add Edges
    # Routing function that decides the next node based on `next_agent`
    # Always takes current state of the graph as the input
    def supervisor_router(AgentState):
        return AgentState["next_agent"]

    # Add conditional edges from the supervisor node
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,  # This function returns the name of the next node
        # Now, we provide a routing dictionary that maps the output of the routing function with the next node
        # This is done to ensure no conditional edges from supervisor to END
        {
            "market_sentiments": "market_sentiments",
            "financial_metrics": "financial_metrics",
            "visualizer": "visualizer",
            "summarizer": "summarizer"
        }
    )

    # Return edges from the child nodes back to the supervisor
    workflow.add_edge("market_sentiments", "supervisor")
    workflow.add_edge("financial_metrics", "supervisor")
    workflow.add_edge("visualizer", "supervisor")
    workflow.add_edge("summarizer", END)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Compile The Graph
    return workflow.compile()

# ----- Streamlit UI ----- #