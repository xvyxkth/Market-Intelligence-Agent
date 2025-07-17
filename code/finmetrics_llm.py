import os
import json
import requests
import matplotlib.pyplot as plt
from langchain.tools import StructuredTool, tool
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Import helper functions
from finmetrics_helper_functions import *

# Store the retrieved metrics data in a new directory
os.makedirs("retrieved_metrics_data", exist_ok=True)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FINANCIAL_MODELLING_PREP_API_KEY = os.getenv("FINANCIAL_MODELLING_PREP_API_KEY")
EVENT_REGISTRY_API_KEY = os.getenv("EVENT_REGISTRY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class IncomeStatementInput(BaseModel):
    symbol: str

class BalanceSheetInput(BaseModel):
    symbol: str

class StockPriceInput(BaseModel):
    symbol: str

class FinancialRatiosInput(BaseModel):
    symbol: str

class CompanyFinancialsInput(BaseModel):
    symbol: str

def get_llm(model_name="gemini-2.0-flash"):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )
    return llm

def get_income_statement(symbol : str) -> dict:
    """Get income statement data"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function" : "INCOME_STATEMENT",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params = params)
    data =  response.json()
    filename = f"retrieved_metrics_data/{symbol}_get_income_statement.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return data

def get_balance_sheet(symbol : str) -> dict:
    """Get balance sheet data"""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function" : "BALANCE_SHEET",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params = params)
    data =  response.json()
    filename = f"retrieved_metrics_data/{symbol}_get_balance_sheet.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return data

def get_stock_price_weekly(symbol : str) -> dict:
    """Get Weekly Stock Price for a Company"""
    base_url = "https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={sym}&apikey={apikey}".format(sym = symbol, apikey = ALPHA_VANTAGE_API_KEY)
    response = requests.get(base_url)
    data =  response.json()
    filename = f"retrieved_metrics_data/{symbol}_get_stock_price_weekly.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return data

def get_financial_ratios(symbol : str) -> dict:
    """Get the Financial Ratios for a company"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
    params = {
        "apikey": FINANCIAL_MODELLING_PREP_API_KEY
    }
    response = requests.get(api_endpoint, params = params)
    data =  response.json()
    filename = f"retrieved_metrics_data/{symbol}_get_financial_ratios.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return data

@tool
def get_company_financials(symbol: str) -> dict:
    """Get Company Financials"""
    api_endpoint = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    params = {
        "apikey": FINANCIAL_MODELLING_PREP_API_KEY
    }
    response = requests.get(api_endpoint, params=params)
    data = response.json()
    filename = f"retrieved_metrics_data/{symbol}_get_company_financials.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    return data

# Convert function into a LangChain tool
income_statement_tool = StructuredTool.from_function(
    name="get_income_statement",
    func=get_income_statement,
    args_schema=IncomeStatementInput,
    description="Get the Income Statement of a company by ticker symbol."
)

balance_sheet_tool = StructuredTool.from_function(
    name="get_balance_sheet",
    func=get_balance_sheet,
    args_schema=BalanceSheetInput,
    description="Get the balance sheet of a company by stock ticker symbol."
)

stock_price_weekly_tool = StructuredTool.from_function(
    name="get_stock_price_weekly",
    func=get_stock_price_weekly,
    args_schema=StockPriceInput,
    description="Get the weekly stock price of a company by stock ticker symbol."
)

company_financials_tool = StructuredTool.from_function(
    name="get_company_financials",
    func=get_company_financials,
    args_schema=CompanyFinancialsInput,
    description="Get the company financials of a company by stock ticker symbol."
)

financial_ratios_tool = StructuredTool.from_function(
    name="get_financial_ratios",
    func=get_financial_ratios,
    args_schema=FinancialRatiosInput,
    description="Get the financial ratios of a company by stock ticker symbol."
)

def retrieve_data(symbols, industry, time_period):
    tools = [income_statement_tool, balance_sheet_tool, stock_price_weekly_tool, company_financials_tool, financial_ratios_tool]
    llm_with_tools = get_llm().bind_tools(tools)

    for symbol in symbols:
        query = f"""You have the following five tools at your disposal :- income_statement_tool, balance_sheet_tool, stock_price_tool, company_financials_tool, financial_ratios_tool.
                    Using the income_statement_tool, Get the Income Statement of : {symbol}. 
                    Using the balance_sheet_tool, Get the Balance Sheet of : {symbol}. 
                    Using the stock_price_weekly_tool, Get the Weekly Stock Price of : {symbol}. 
                    Using the company_financials_tool, Get the Company Financials of : {symbol}. 
                    Using the financial_ratios_tool, Get the Financial Ratios of : {symbol}. 
                 """
    
        print(f"\nQuery: {query}")
        response = llm_with_tools.invoke(query)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args")

                if tool_name == "get_income_statement":
                    result = get_income_statement(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_balance_sheet":
                    result = get_balance_sheet(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_stock_price_weekly":
                    result = get_stock_price_weekly(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_company_financials":
                    result = get_company_financials.invoke(tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
                if tool_name == "get_financial_ratios":
                    result = get_financial_ratios(**tool_args)
                    print(f"Tool `{tool_name}` used with arguments {tool_args}.")
                    print(f"Answer: {json.dumps(result, indent=2)}\n")
        else:
            print("No tool was used. LLM response:", getattr(response, "text", response), "\n")

# Function for comparative analysis
def generate_comparative_analysis(companies_data, industry):
    """Generate comparative analysis between companies using LLM"""
    llm = get_llm()
    
    companies_summary = []
    for symbol, data in companies_data.items():
        if not data.get('summary'):
            continue
            
        company_info = {
            'symbol': symbol,
            'name': data['summary'].get('name', symbol),
            'sector': data['summary'].get('sector', 'N/A'),
            'industry': data['summary'].get('industry', 'N/A')
        }
        
        if data.get('income_statement') and len(data['income_statement']) > 0:
            latest = data['income_statement'][0]
            company_info['revenue'] = latest.get('total_revenue', 0)
            company_info['net_income'] = latest.get('net_income', 0)
            company_info['gross_margin'] = latest.get('gross_margin', 0)
            company_info['operating_margin'] = latest.get('operating_margin', 0)
            company_info['net_margin'] = latest.get('net_margin', 0)
            if len(data['income_statement']) >= 2:
                company_info['revenue_growth'] = latest.get('revenue_growth', 0)
                company_info['net_income_growth'] = latest.get('net_income_growth', 0)
        
        if data.get('financial_ratios') and len(data['financial_ratios']) > 0:
            latest = data['financial_ratios'][0]
            company_info['pe_ratio'] = latest.get('price_to_earnings', 0)
            company_info['pb_ratio'] = latest.get('price_to_book', 0)
            company_info['ps_ratio'] = latest.get('price_to_sales', 0)
            company_info['ev_ebitda'] = latest.get('enterprise_value_multiple', 0)
        
        if data.get('stock_price') and data['stock_price'].get('summary'):
            summary = data['stock_price']['summary']
            company_info['price_change'] = summary.get('percent_change', 0)
            company_info['volatility'] = summary.get('volatility', 0)
        
        companies_summary.append(company_info)
    
    # Prompt
    prompt = f"""
    You are a financial analyst tasked with comparing companies in the {industry} industry.
    
    Here is the data for the companies:
    {companies_summary}
    
    Please provide a detailed comparative analysis of these companies focusing on:
    1. Revenue and profitability comparison
    2. Valuation metrics comparison (P/E, P/B, P/S, EV/EBITDA)
    3. Growth metrics comparison
    4. Relative strengths and weaknesses
    5. Competitive positioning within the industry
    
    Format your response as a markdown section that can be directly included in a financial report.
    """
    
    response = llm.invoke(prompt)
    analysis = getattr(response, "content", str(response))
    
    return analysis

# Function for investment insights
def generate_investment_insights(companies_data, industry_averages, industry):
    """Generate investment insights using LLM"""
    llm = get_llm()
    
    companies_summary = []
    for symbol, data in companies_data.items():
        if not data.get('summary'):
            continue
        companies_summary.append(data['summary'])
    
    # Prompt
    prompt = f"""
    You are a financial advisor tasked with providing investment insights for companies in the {industry} industry.
    
    Here is the data for the companies:
    {companies_summary}
    
    Here are the industry averages:
    {industry_averages}
    
    Please provide detailed investment insights focusing on:
    1. Companies that appear undervalued or overvalued
    2. Companies with strong growth prospects
    3. Companies with potential financial risks
    4. Overall industry outlook
    5. Specific investment recommendations (buy/hold/sell) with rationale
    
    Format your response as a markdown section that can be directly included in a financial report.
    """
    
    response = llm.invoke(prompt)
    insights = getattr(response, "content", str(response))
    
    return insights

# Function to generate executive summary
def generate_executive_summary(companies_data, industry_averages, industry, time_period):
    """Generate executive summary using LLM"""
    llm = get_llm()
    
    prompt = f"""
    You are a financial analyst tasked with creating an executive summary for a financial analysis report on the {industry} industry.
    
    Here is the industry average data:
    {industry_averages}
    
    The analysis covers {len(companies_data)} companies over a {time_period}-day period.
    
    Please provide a concise executive summary focusing on:
    1. Overall financial health of the industry
    2. Key financial metrics and trends
    3. Major financial strengths and challenges in the industry
    4. Brief outlook for the industry
    
    Format your response as a markdown section (2-3 paragraphs) that can be directly included in a financial report.
    """

    response = llm.invoke(prompt)
    summary = getattr(response, "content", str(response))
    
    return summary

# For final report
def generate_financial_report(companies_data, industry_averages, industry, time_period):
    """Generate comprehensive financial analysis report"""
    
    report = []
    
    report.append(f"# Financial Analysis Report: {industry} Industry")
    report.append(f"## Analysis Period: {time_period} days")
    
    report.append("\n## Executive Summary")
    executive_summary = generate_executive_summary(companies_data, industry_averages, industry, time_period)
    report.append(executive_summary)
    
    report.append("\n### Industry Average Metrics")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    
    if industry_averages.get('profit_margins'):
        report.append(f"| Gross Margin | {industry_averages['profit_margins'].get('gross_margin', 0):.2f}% |")
        report.append(f"| Operating Margin | {industry_averages['profit_margins'].get('operating_margin', 0):.2f}% |")
        report.append(f"| Net Margin | {industry_averages['profit_margins'].get('net_margin', 0):.2f}% |")
    
    if industry_averages.get('returns'):
        report.append(f"| Return on Assets (ROA) | {industry_averages['returns'].get('return_on_assets', 0):.2f}% |")
        report.append(f"| Return on Equity (ROE) | {industry_averages['returns'].get('return_on_equity', 0):.2f}% |")
    
    if industry_averages.get('liquidity'):
        report.append(f"| Current Ratio | {industry_averages['liquidity'].get('current_ratio', 0):.2f} |")
        report.append(f"| Debt-to-Equity | {industry_averages['liquidity'].get('debt_to_equity', 0):.2f} |")
    
    if industry_averages.get('valuation'):
        report.append(f"| Price-to-Earnings (P/E) | {industry_averages['valuation'].get('price_to_earnings', 0):.2f} |")
        report.append(f"| Price-to-Book (P/B) | {industry_averages['valuation'].get('price_to_book', 0):.2f} |")
        report.append(f"| Price-to-Sales (P/S) | {industry_averages['valuation'].get('price_to_sales', 0):.2f} |")
        report.append(f"| EV/EBITDA | {industry_averages['valuation'].get('enterprise_value_multiple', 0):.2f} |")
    
    report.append("\n## Company Analysis")
    
    for symbol, data in companies_data.items():
        if not data.get('summary'):
            continue
            
        company_summary = data['summary']
        report.append(f"\n### {company_summary.get('name', symbol)} ({symbol})")
        
        report.append(f"**Sector:** {company_summary.get('sector', 'N/A')}  ")
        report.append(f"**Industry:** {company_summary.get('industry', 'N/A')}  ")
        if company_summary.get('company_info') and company_summary['company_info'].get('market_cap'):
            market_cap = company_summary['company_info'].get('market_cap', 0)
            if market_cap >= 1_000_000_000:
                market_cap_str = f"${market_cap / 1_000_000_000:.2f}B"
            else:
                market_cap_str = f"${market_cap / 1_000_000:.2f}M"
            report.append(f"**Market Cap:** {market_cap_str}  ")
        
        report.append("\n#### Financial Performance")
        report.append("| Metric | Value | vs. Industry Avg |")
        report.append("|--------|-------|-----------------|")
        
        if company_summary.get('financials'):
            fin = company_summary['financials']
            revenue = fin.get('revenue', 0)
            if revenue >= 1_000_000_000:
                revenue_str = f"${revenue / 1_000_000_000:.2f}B"
            else:
                revenue_str = f"${revenue / 1_000_000:.2f}M"
            
            report.append(f"| Revenue | {revenue_str} | N/A |")
            report.append(f"| Gross Margin | {fin.get('gross_margin', 0):.2f}% | {fin.get('gross_margin_vs_industry', 0):.2f}% |")
            report.append(f"| Operating Margin | {fin.get('operating_margin', 0):.2f}% | {fin.get('operating_margin_vs_industry', 0):.2f}% |")
            report.append(f"| Net Margin | {fin.get('net_margin', 0):.2f}% | {fin.get('net_margin_vs_industry', 0):.2f}% |")
        
        report.append("\n#### Financial Health")
        report.append("| Metric | Value | vs. Industry Avg |")
        report.append("|--------|-------|-----------------|")
        
        if company_summary.get('health'):
            health = company_summary['health']
            report.append(f"| Current Ratio | {health.get('current_ratio', 0):.2f} | {health.get('current_ratio_vs_industry', 0):.2f} |")
            report.append(f"| Debt-to-Equity | {health.get('debt_to_equity', 0):.2f} | {health.get('debt_to_equity_vs_industry', 0):.2f} |")
        
        if company_summary.get('returns'):
            report.append("\n#### Return Metrics")
            report.append("| Metric | Value | vs. Industry Avg |")
            report.append("|--------|-------|-----------------|")
            
            returns = company_summary['returns']
            report.append(f"| Return on Assets (ROA) | {returns.get('roa', 0):.2f}% | {returns.get('roa_vs_industry', 0):.2f}% |")
            report.append(f"| Return on Equity (ROE) | {returns.get('roe', 0):.2f}% | {returns.get('roe_vs_industry', 0):.2f}% |")
        
        if company_summary.get('valuation'):
            report.append("\n#### Valuation Metrics")
            report.append("| Metric | Value | vs. Industry Avg |")
            report.append("|--------|-------|-----------------|")
            
            valuation = company_summary['valuation']
            report.append(f"| Price-to-Earnings (P/E) | {valuation.get('pe_ratio', 0):.2f} | {valuation.get('pe_ratio_vs_industry', 0):.2f} |")
            report.append(f"| Price-to-Book (P/B) | {valuation.get('pb_ratio', 0):.2f} | {valuation.get('pb_ratio_vs_industry', 0):.2f} |")
            report.append(f"| Price-to-Sales (P/S) | {valuation.get('ps_ratio', 0):.2f} | {valuation.get('ps_ratio_vs_industry', 0):.2f} |")
            report.append(f"| EV/EBITDA | {valuation.get('ev_ebitda', 0):.2f} | {valuation.get('ev_ebitda_vs_industry', 0):.2f} |")
        
        if company_summary.get('stock'):
            report.append("\n#### Stock Performance")
            stock = company_summary['stock']
            report.append(f"- Current Price: ${stock.get('latest_price', 0):.2f}")
            report.append(f"- {time_period}-day Change: {stock.get('price_change', 0):.2f}% ({'+' if stock.get('price_change_vs_industry', 0) > 0 else ''}{stock.get('price_change_vs_industry', 0):.2f}% vs industry)")
            report.append(f"- Volatility: {stock.get('volatility', 0):.2f}")
        
        report.append("\n#### SWOT Analysis")
        
        if company_summary.get('strengths'):
            report.append("\n**Strengths:**")
            for strength in company_summary['strengths']:
                report.append(f"- {strength}")
        
        if company_summary.get('weaknesses'):
            report.append("\n**Weaknesses:**")
            for weakness in company_summary['weaknesses']:
                report.append(f"- {weakness}")
        
        if company_summary.get('opportunities'):
            report.append("\n**Opportunities:**")
            for opportunity in company_summary['opportunities']:
                report.append(f"- {opportunity}")
        
        if company_summary.get('threats'):
            report.append("\n**Threats:**")
            for threat in company_summary['threats']:
                report.append(f"- {threat}")
    
    report.append("\n## Comparative Analysis")
    comparative_analysis = generate_comparative_analysis(companies_data, industry)
    report.append(comparative_analysis)
    
    report.append("\n## Investment Insights")
    investment_insights = generate_investment_insights(companies_data, industry_averages, industry)
    report.append(investment_insights)
    
    report.append("\n## Disclaimer")
    report.append("*This financial analysis is based on historical data and should not be considered as financial advice. Investment decisions should be made in consultation with a qualified financial advisor. Past performance is not indicative of future results.*")
    
    return "\n".join(report)

def run_fin(state):
    symbols = state["companies"]
    industry = state["industry"]
    time_period = state["time_period"]

    print("STARTING FINANCIAL ANALYSIS ")

    retrieve_data(symbols, industry, time_period)

    companies_data = {}
    
    for symbol in symbols:
        print(f"\nParsing data for {symbol}...")
        
        income_statement_data = parse_income_statement(symbol)
        balance_sheet_data = parse_balance_sheet(symbol)
        stock_price_data = parse_stock_price(symbol)
        company_financials_data = parse_company_financials(symbol)
        financial_ratios_data = parse_financial_ratios(symbol)
        
        companies_data[symbol] = {
            'income_statement': income_statement_data,
            'balance_sheet': balance_sheet_data,
            'stock_price': stock_price_data,
            'company_info': company_financials_data,
            'financial_ratios': financial_ratios_data
        }
    
    industry_averages = calculate_industry_averages(companies_data)
    
    for symbol, data in companies_data.items():
        companies_data[symbol]['summary'] = generate_company_summary(data, industry_averages)
    
    report = generate_financial_report(companies_data, industry_averages, industry, time_period)
    
    with open("financial_analysis_report.md", "w") as f:
        f.write(report)
    
    print("\nFinancial analysis report generated successfully. See 'financial_analysis_report.md'")
    
    # Generate visualizations
    try:
        plt.figure(figsize=(12, 6))
        
        for symbol, data in companies_data.items():
            if data.get('stock_price') and data['stock_price'].get('weekly_data'):
                weekly_data = data['stock_price']['weekly_data']
                dates = [item['index'] for item in weekly_data[-12:]]
                prices = [item['close'] for item in weekly_data[-12:]]
                
                plt.plot(dates, prices, label=symbol)
        
        plt.title(f"{industry} Stock Price Trends - Last 12 Weeks")
        plt.xlabel("Date")
        plt.ylabel("Stock Price (USD)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("stock_price_trends.png")
        
        plt.figure(figsize=(12, 6))
        
        companies = []
        net_margins = []
        
        for symbol, data in companies_data.items():
            if data.get('income_statement') and len(data['income_statement']) > 0:
                companies.append(symbol)
                net_margins.append(data['income_statement'][0].get('net_margin', 0))
        
        if companies and net_margins:
            plt.bar(companies, net_margins)
            plt.axhline(y=industry_averages['profit_margins'].get('net_margin', 0), color='r', linestyle='-', label="Industry Average")
            plt.title(f"{industry} Net Profit Margin Comparison")
            plt.xlabel("Company")
            plt.ylabel("Net Profit Margin (%)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("profit_margin_comparison.png")
        
        print("Visualizations generated successfully.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

    return state