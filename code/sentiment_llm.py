import os
import json
import requests
from langchain.tools import StructuredTool
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List

# Import helper function to parse data
from sentiment_helper_functions import parse_retrieved_data

# Store the retrieved sentiment data in a new directory
os.makedirs("retrieved_sentiment_data", exist_ok=True)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINANCIAL_MODELLING_PREP_API_KEY = os.getenv("FINANCIAL_MODELLING_PREP_API_KEY")
EVENT_REGISTRY_API_KEY = os.getenv("EVENT_REGISTRY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_llm(model_name="gemini-2.0-flash"):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )
    return llm

class CompanyOverviewInput(BaseModel):
    symbol: str

class CompanyNewsInput(BaseModel):
    company_name: str
    days_back: int = 10

class IndustryDataInput(BaseModel):
    industry: str

class NewsSentimentInput(BaseModel):
    tickers: List[str]

def get_company_overview(symbol: str) -> dict:
    print(f"[TOOL CALL] get_company_overview({symbol})")
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    print("[DATA RECEIVED] Company Overview:", json.dumps(data, indent=2)[:1000])

    filename = f"retrieved_sentiment_data/{symbol}_get_company_overview.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return data


def get_company_news(company_name: str, days_back: int = 10) -> dict:
    print(f"[TOOL CALL] get_company_news({company_name}, days_back={days_back})")
    api_endpoint = "https://eventregistry.org/api/v1/article/getArticles"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    payload = {
        "action": "getArticles",
        "keyword": company_name,
        "dateStart": start_date.strftime("%Y-%m-%d"),
        "dateEnd": end_date.strftime("%Y-%m-%d"),
        "articlesCount": 5,
        "resultType": "articles",
        "apiKey": EVENT_REGISTRY_API_KEY
    }
    try:
        response = requests.post(api_endpoint, json=payload)
        if response.text and response.status_code == 200:
            data = response.json()
            print("[DATA RECEIVED] Company News:", json.dumps(data, indent=2)[:1000])
            
            filename = f"retrieved_sentiment_data/{company_name}_get_company_news.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            
            return data
        else:
            print(f"[ERROR] News API returned status {response.status_code}")
            return {"error": True, "content": response.text[:500]}
    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
        return {"error": True, "exception": str(e)}


def get_industry_data(industry: str) -> list:
    print(f"[TOOL CALL] get_industry_data({industry})")
    api_endpoint = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        "apikey": FINANCIAL_MODELLING_PREP_API_KEY,
        "sector": industry,
        "limit": 10
    }
    response = requests.get(api_endpoint, params=params)
    data = response.json()
    print("[DATA RECEIVED] Industry Data:", json.dumps(data, indent=2)[:1000])

    filename = f"retrieved_sentiment_data/{industry}_get_industry_data.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return data


def get_news_sentiment(tickers: list[str] = None) -> dict:
    print(f"[TOOL CALL] get_news_sentiment({tickers})")
    base_url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
    if tickers:
        base_url += "&tickers=" + ','.join(tickers)
    base_url += '&apikey=' + ALPHA_VANTAGE_API_KEY
    response = requests.get(base_url)
    data = response.json()
    print("[DATA RECEIVED] News Sentiment:", json.dumps(data, indent=2)[:1000])

    with open("retrieved_sentiment_data/get_news_sentiment.json", "w") as f:
        json.dump(data, f, indent=2)

    return data


# Convert function into a LangChain tool
company_overview_tool = StructuredTool.from_function(
    name="get_company_overview",
    func=get_company_overview,
    args_schema=CompanyOverviewInput,
    description="Get overview data for a company by ticker symbol."
)

company_news_tool = StructuredTool.from_function(
    name="get_company_news",
    func=get_company_news,
    args_schema=CompanyNewsInput,
    description="Get recent news articles and sentiment for a company."
)

industry_data_tool = StructuredTool.from_function(
    name="get_industry_data",
    func=get_industry_data,
    args_schema=IndustryDataInput,
    description="Get a list of companies and stats in a given industry."
)

news_sentiment_tool = StructuredTool.from_function(
    name="get_news_sentiment",
    func=get_news_sentiment,
    args_schema=NewsSentimentInput,
    description="Get news and sentiment data for a list of tickers."
)

def retrieve_data(symbols, industry, time_period):

    # Initialize Gemini model with tools
    llm = get_llm()
    tools = [company_overview_tool, company_news_tool, industry_data_tool, news_sentiment_tool]
    llm_with_tools = llm.bind_tools(tools)

    for symbol in symbols:
        query = f"""You have the following four tools at your disposal :- company_overview_tool, company_news_tool, industry_data_tool, news_sentiment_tool. 
                    Using the company_overview_tool, Get the company overview of : {symbol}. 
                    The companies all belong to {industry} industry. Use the industry_data_tool to get industry data for {industry}.
                    Use the company_news_tool to extract recent news for each company denoted by {symbol} and for the last {time_period} days.
                    Use the news_sentiment_tool to extract news sentiments for each of the following companies {symbols}.
                 """
        print(f"\nQuery: {query}")
        response = llm_with_tools.invoke(query)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args")

                if tool_name == "get_company_overview":
                    result = get_company_overview(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_industry_data":
                    result = get_industry_data(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_company_news":
                    result = get_company_news(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_news_sentiment":
                    result = get_news_sentiment(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
        else:
            print("No tool was used. LLM response:", getattr(response, "text", response), "\n")

def generate_sentiment_report(parsed_data, symbols, industry, time_period):
    """
    Generate a comprehensive sentiment analysis report based on the parsed data.
    """

    llm = get_llm()
    
    data_summary = {
        "industry": industry,
        "time_period": time_period,
        "companies": {},
        "industry_overview": parsed_data["industry"]
    }
    
    # Summarize company data
    for symbol in symbols:
        if symbol in parsed_data["companies"]:
            company_data = parsed_data["companies"][symbol]
            news_sentiment = parsed_data["news_sentiment"]["ticker_averages"].get(symbol, {})
            
            data_summary["companies"][symbol] = {
                "name": company_data.get("overview", {}).get("name", symbol),
                "financial_metrics": {
                    "market_cap": company_data.get("overview", {}).get("market_cap", ""),
                    "pe_ratio": company_data.get("overview", {}).get("pe_ratio", ""),
                    "quarterly_earnings_growth": company_data.get("overview", {}).get("quarterly_earnings_growth", ""),
                    "quarterly_revenue_growth": company_data.get("overview", {}).get("quarterly_revenue_growth", "")
                },
                "news_sentiment": {
                    "avg_sentiment": news_sentiment.get("avg_sentiment", 0),
                    "sentiment_label": news_sentiment.get("sentiment_label", "Neutral"),
                    "news_count": news_sentiment.get("news_count", 0)
                },
                "recent_events": [
                    {
                        "title": article.get("title", ""),
                        "sentiment": article.get("sentiment", 0)
                    } for article in parsed_data["news_sentiment"]["ticker_specific"].get(symbol, [])
                ]
            }
    
    with open("retrieved_sentiment_data/data_summary.json", "w") as f:
        json.dump(data_summary, f, indent=2)

    # Read the financial analysis report
    try:
        with open("financial_analysis_report.md", "r") as f:
            financial_analysis_report = f.read()
    except FileNotFoundError:
        financial_analysis_report = "Financial metrics report not found."

    # prompt for sentiment report generation
    prompt = f"""
    You are a financial analyst specializing in sentiment analysis. Generate a comprehensive, insightful sentiment analysis report
    based on the following data for {len(symbols)} companies in the {industry} industry over the past {time_period} days.
    You can use the following data summary to get an overview of the collected sentiment data for each firm. 
    You can also use the financial analysis report which gives an overview of the financial details of each firm.
    DATA SUMMARY:
    {json.dumps(data_summary, indent=2)}

    FINANCIAL METRICS REPORT:
    {financial_analysis_report}

    Your report should include:
    1. An executive summary highlighting key findings
    2. Individual company analysis with relevant metrics and sentiment scores
    3. Industry-wide analysis and trends
    4. Comparative analysis between companies
    5. Actionable insights and recommendations based on the sentiment data

    Use a professional, analytical tone, and focus on extracting meaningful insights from the data.
    Include relevant metrics but focus on what's most important and impactful.
    Organize the report in a clear, structured format with appropriate Markdown headings and sections.

    The report should be titled "Sentiment Analysis Report: {industry} Industry" and include today's date in the header.
    """

    response = llm.invoke(prompt)
    report_content = response.content
    
    # Save report to sentiment_report.md
    report_file = "sentiment_report.md"
    with open(report_file, "w") as f:
        f.write(report_content)
    
    print(f"Sentiment analysis report generated and saved to {report_file}")
    
    # print a part of the report for verification
    return report_content[:500] + "..."


def run_sent(state):

    symbols = state["companies"]
    industry = state["industry"]
    time_period = state["time_period"]

    retrieve_data(symbols, industry, time_period)

    parsed_data = parse_retrieved_data(symbols, industry)

    generate_sentiment_report(parsed_data, symbols, industry, time_period)

    return state