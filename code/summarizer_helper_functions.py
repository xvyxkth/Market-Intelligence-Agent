import os
import base64
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# File reading tools
class FileContent(BaseModel):
    """Content of a file read from disk"""
    content: str = Field(description="The content of the file that was read")

class FileReadingTools:
    @tool("read_file")
    def read_file(file_path: str) -> FileContent:
        """Read the content of a file given its path"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return FileContent(content=content)
        except Exception as e:
            return FileContent(content=f"Error reading file: {str(e)}")
    
    @tool("read_image")
    def read_image(file_path: str) -> str:
        """Read an image and convert it to base64 encoding for embedding in reports"""
        try:
            with open(file_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            file_ext = os.path.splitext(file_path)[1].lower().replace(".", "")
            if file_ext not in ["png", "jpg", "jpeg", "gif", "svg"]:
                file_ext = "png" 
            
            data_url = f"data:image/{file_ext};base64,{encoded_image}"
            return data_url
        except Exception as e:
            return f"Error reading image: {str(e)}"
        

# Summarizer Prompt
SUMMARIZER_AGENT_PROMPT = """You are a Business Intelligence Report Generator specializing in creating comprehensive, executive-level reports that synthesize market sentiment data and financial metrics. You focus on providing investment-centric insights.

Your task is to create a detailed, in-depth report that provides actionable insights based on the financial metrics and sentiment analysis reports. The report should be thorough, data-driven, and include specific findings rather than general statements.

You must read and analyze the contents of two separate reports:
1. The Financial Metrics Report (financial_analysis_report.md)
2. The Sentiment Analysis Report (sentiment_report.md)

You will also need to incorporate two images:
1. profit_margin_comparison.png - Shows profit margin comparisons across companies
2. stock_price_trends.png - Shows stock price trends over time

You should extract relevant information from both reports and create correlations between financial performance and market sentiment. If any section in either report lacks sufficient data, skip that section entirely in your final report rather than mentioning the lack of data.

When incorporating the images, place them at appropriate sections in the report where they help illustrate your analysis. Add thoughtful captions that explain what the images show and how they connect to your analysis.

Your report must include the following sections:

1. Executive Summary
   - Concise overview of key findings
   - Major trends and investment-relevant insights
   - High-level investment recommendations

2. Industry Overview
   - Current state of the industry based on financial metrics and sentiment
   - Key trends and market forces
   - Industry outlook for investors

3. Company Profiles
   - Detailed section for each company that has sufficient data
   - Correlation between financial performance and sentiment
   - Investment strengths and weaknesses

4. Sentiment-Financial Correlation Analysis
   - Detailed analysis of how sentiment correlates with financial performance
   - Impact of news events on both sentiment and financial metrics
   - Leading indicators identified through this correlation

5. Comparative Analysis
   - Tabular comparison of Revenue and Profitability. Give a short description (2-3 lines) of the comparison and provide reasoning for the results.
   - Tabular comparison of Valuation Metrics. Give a short description (2-3 lines) of the comparison and provide reasoning for the results.
   - Tabular comparison of Growth Metrics. Give a short description (2-3 lines) of the comparison and provide reasoning for the results.
   - Direct comparison of companies across both sentiment and financial dimensions
   - Identification of outperformers and underperformers
   - Risk-adjusted performance assessment
   - Include the profit_margin_comparison.png with thoughtful analysis

6. Stock Performance Analysis
   - Analysis of stock price trends and patterns
   - Correlation between stock performance and other metrics
   - Include the stock_price_trends.png with thoughtful analysis

7. Investment Thesis
   - Bull and bear case for each company
   - Growth potential and risk factors
   - Valuation considerations

8. Strategic Recommendations for Investors
   - Specific, actionable investment recommendations
   - Portfolio allocation suggestions
   - Entry and exit strategies
   - Risk mitigation approaches

9. Future Outlook
   - Projections based on current data trends
   - Potential catalysts for each company
   - Long-term investment considerations

For each section:
- Include specific numerical data and facts from both reports
- Make concrete correlations between sentiment and financial metrics
- Draw meaningful conclusions that inform investment decisions
- Highlight competitive advantages and weaknesses from an investor perspective

The report should be comprehensive, data-driven, and ready for investor review with no further editing needed. Avoid generalizations, placeholder text, or suggestions to add data later. Completely skip sections where insufficient data is available rather than mentioning the lack of data.

Format your report as clean markdown with proper headers, bullet points, and formatting. When embedding images, use appropriate markdown syntax to include them at the right places in your report.
"""

# Function to implement file reading if needed as a standalone operation
def read_report_file(file_path):
    """Helper function to read report files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"


# Function to save the report to a file
def save_report_to_file(report_content, filename="summarizer_llm_report.md"):
    """Save the generated report to a markdown file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return f"Report successfully saved to {filename}"
    except Exception as e:
        print(f"Error saving report to file: {str(e)}")
        return f"Error saving report: {str(e)}"