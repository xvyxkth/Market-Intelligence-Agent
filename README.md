# Market-Intelligence-Agent
This project focuses on building a multi-agent workflow for a market intelligence agent which provides a comprehensive business report including financials and sentiment analysis for a list of companies in a particular industry over a specified time period. 

The project uses 4 agents :
Supervisor Agent : Parses the input query and orchestrates the analysis workflow.
Sentiment Analysis Agent : Monitors real-time news of companies and industry with sentiment analysis.
Financial Analysis Agent : Obtains and analyses the metrics of companies in a particular industry. It also generates visualizations of the financial data.
Summarizer Agent : Responsible for consolidating all data collected by other agents and their responses to generate a comprehensive and unified answer for the user.

The workflow is as follows :-
The supervisor agent receives the input user query and parses it to retrieve the list of companies, industry and time period of analysis. It then passes control to the financial analysis agent which retrieves a comprehensive list of metrics for the specified companies. It analyzes these metrics and stores the analysis in a file. Then the control is passed to the sentiment analysis agent which retrieves news articles and other financial and sentiment data pertaining to the industry. It analyzes the received information and stores the analysis in another file. The control is passed on to the summarizer agent which summarizes the financial analysis and sentiment analysis stored in the 2 files and generates a comprehensive business intelligence report. The report is displayed on a web application using streamlit and can also be downloaded as a PDF.
