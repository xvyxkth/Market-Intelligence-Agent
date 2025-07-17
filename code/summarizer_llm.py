import os
import base64
import re
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool

# Import Helper Functions
from summarizer_helper_functions import FileReadingTools, read_report_file, save_report_to_file, SUMMARIZER_AGENT_PROMPT

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_llm(model_name = "gemini-2.0-flash"):
    """Gets an LLM Instance --> Can be configured with tools depending on which agent is calling the function"""
    llm = ChatGoogleGenerativeAI(
        model = model_name,
        api_key = GOOGLE_API_KEY
    )
    return llm

def generate_summary(state):
    """
    Enhanced Summarizer Agent that creates the final report by reading and analyzing
    contents from financial_analysis_report.md and sentiment_report.md along with
    incorporating profit_margin_comparison.png and stock_price_trends.png
    """
    messages = state["messages"]
    
    # file reading tools
    file_tools = [
        Tool.from_function(
            func=FileReadingTools.read_file,
            name="read_file",
            description="Read a file from disk given its path"
        ),
        Tool.from_function(
            func=FileReadingTools.read_image,
            name="read_image",
            description="Read an image file and return its base64 encoding for embedding in reports"
        )
    ]
    
    llm_with_tools = get_llm()
    llm_with_tools.bind_tools(file_tools)
    
    financial_report_content = ""
    sentiment_report_content = ""
    profit_margin_image_data = ""
    stock_price_image_data = ""
    
    try:
        # read text files directly
        financial_report_content = read_report_file("financial_analysis_report.md")
        sentiment_report_content = read_report_file("sentiment_report.md")
        
        # read images directly
        with open("profit_margin_comparison.png", 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            profit_margin_image_data = f"data:image/png;base64,{encoded_image}"
            
        with open("stock_price_trends.png", 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            stock_price_image_data = f"data:image/png;base64,{encoded_image}"
            
        print("Successfully read all input files directly")
    except Exception as e:
        print(f"Error reading input files directly: {str(e)}")
    
    # chat history for the LLM with detailed instructions
    chat_history = [SystemMessage(content=SUMMARIZER_AGENT_PROMPT)]
    
    # tool execution prompt to read both required files and images
    tool_exec_prompt = """You need to read four important files to complete your task:
    1. financial_analysis_report.md - contains financial metrics and analysis
    2. sentiment_report.md - contains sentiment analysis for the companies
    3. profit_margin_comparison.png - image showing profit margin comparisons
    4. stock_price_trends.png - image showing stock price trends
    
    Please first read both text files to understand their contents, then read the image files to incorporate them at appropriate places in your report.
    """
    
    chat_history.append(HumanMessage(content=tool_exec_prompt))
    
    report_summary = f"""
    I've analyzed the following files:
    
    1. Financial Analysis Report:
    {financial_report_content[:500]}... (content truncated for brevity)
    
    2. Sentiment Analysis Report:
    {sentiment_report_content[:500]}... (content truncated for brevity)
    
    3. I've also processed the profit margin comparison and stock price trend images, which are available to include in the final report.
    """

    chat_history.append(HumanMessage(content=report_summary))
    
    # prompt for the actual analysis and report generation
    analysis_context = f"""
    Now that you have read the financial metrics report, sentiment analysis report, and have access to the visualization images, please generate a comprehensive business intelligence report for:
    - Companies: {', '.join(state['companies']) if 'companies' in state and state['companies'] else 'the analyzed companies'}
    - Industry: {state['industry'] if 'industry' in state else 'the analyzed industry'}
    - Time Period: {state['time_period'] if 'time_period' in state else 'the analyzed period'}
    
    Your report should:
    1. Establish clear correlations between sentiment data and financial metrics for each company
    2. Provide investment-focused insights and recommendations
    3. Analyze trends and patterns across both datasets
    4. Include the visualization images at appropriate places with thoughtful captions and analysis
    5. Offer actionable investment strategies based on the combined analysis
    
    Remember to:
    - Provide elaborate reasoning for all results
    - Establish meaningful connections between sentiment and financial performance
    - Make the report investment-centric, highlighting opportunities, risks, and strategic considerations
    - Format the report professionally using markdown
    - Include the images at relevant sections with proper captions and analysis
    
    Please ensure the report is comprehensive yet coherent, with a logical flow from the executive summary through to the future outlook.
    """
    
    # create a prompt for the report generation
    final_prompt = [
        SystemMessage(content=SUMMARIZER_AGENT_PROMPT),
        HumanMessage(content=f"""
        Here are the contents of the financial analysis report:
        
        {financial_report_content}
        
        Here are the contents of the sentiment analysis report:
        
        {sentiment_report_content}
        
        I've also analyzed two important images:
        1. Profit Margin Comparison - This shows the profit margins for the companies being analyzed
        2. Stock Price Trends - This shows the stock price trends over time
        
        {analysis_context}
        """)
    ]
    
    try:
        print("Generating final business intelligence report")
        response = llm_with_tools.invoke(final_prompt)
        print("Successfully generated report")
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        response_content = f"""
        # Business Intelligence Report
        
        ## Executive Summary
        
        Unable to generate a complete report due to API limitations. The financial and sentiment analysis reports were successfully read, but the final report generation encountered an error.
        
        Error details: {str(e)}
        
        Please try again with a simplified request or check API credentials.
        """
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
                
        response = MockResponse(response_content)
    
    # process the report content to embed images
    report_text = response.content
    
    # replace image placeholders with actual base64 images if they exist
    if profit_margin_image_data:
        report_text = report_text.replace(
            "![Profit Margin Comparison](profit_margin_comparison.png)", 
            f"![Profit Margin Comparison]({profit_margin_image_data})"
        )
    
    if stock_price_image_data:
        report_text = report_text.replace(
            "![Stock Price Trends](stock_price_trends.png)", 
            f"![Stock Price Trends]({stock_price_image_data})"
        )
    
    # store the final report
    state["report"] = {
        "content": report_text,
        "generated_at": datetime.now().isoformat(),
        "completed": True
    }
    
    report_intro = "## Investment-Focused Business Intelligence Report Complete\n\n"
    exec_summary_match = re.search(r"Executive Summary.*?\n(.*?)(?=##|\n\n\d\.)", report_text, re.DOTALL)
    if exec_summary_match:
        report_intro += exec_summary_match.group(1).strip() + "\n\n"
    else:
        report_intro += "Report generation complete. The full report contains detailed financial and sentiment analysis with actionable investment recommendations.\n\n"
    
    # add completion message
    company_list = ', '.join(state['companies']) if 'companies' in state and state['companies'] else 'the analyzed companies'
    industry = state['industry'] if 'industry' in state else 'the analyzed industry'
    time_period = state['time_period'] if 'time_period' in state else 'the analyzed period'
    
    report_intro += f"Full investment analysis generated for {company_list} in the {industry} industry covering {time_period}."
    
    messages.append({
        "role": "ai", 
        "content": report_intro
    })
    
    return state

def run_summ(state):
    """
    Main function to test the functionality of the business intelligence summarizer
    """

    print("Starting Business Intelligence Report Generator")
    
    required_files = ["financial_analysis_report.md", "sentiment_report.md", 
                       "profit_margin_comparison.png", "stock_price_trends.png"]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required file not found: {file_path}")
            return state
    
    print("All required input files found")
    
    print("Running summarizer agent")
    updated_state = generate_summary(state)
    
    # save the report
    if updated_state["report"].get("completed", False):
        report_filename = f"summarizer_llm_report.md" 
        result = save_report_to_file(updated_state["report"]["content"], report_filename)
        print(result)
        
        print("\n" + "="*50)
        print(f"Business Intelligence Report Generated: {report_filename}")
        print("="*50 + "\n")
        
        # prints a small part of the summary
        report_content = updated_state["report"]["content"]
        exec_summary_match = re.search(r"## Executive Summary\s*(.*?)(?=##)", report_content, re.DOTALL)
        if exec_summary_match:
            print("EXECUTIVE SUMMARY EXCERPT:")
            print("-"*50)
            print(exec_summary_match.group(1).strip()[:500] + "...")
            print("-"*50)
    else:
        print("Report generation failed or incomplete")
    
    return state