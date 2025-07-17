import os
import streamlit as st
from typing import List, Dict, Any, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the helper functions
from supervisor_helper_functions import render_markdown_with_images, markdown_to_pdf, parse_user_query

# Import the sub-agents
from sentiment_llm import run_sent  # sentiment_agent
from finmetrics_llm import run_fin  # finmetrics agent
from summarizer_llm import run_summ # summarizer agent 

# Shared State
class AgentState(TypedDict):
    messages: List[Dict]
    next_agent: str
    industry: str
    companies: List[str]
    time_period: int
    task_status: str
    report: Dict[str, Any]

# Function to get an LLM instance
def get_llm(model_name="gemini-2.0-flash"):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )
    return llm

# Workflow Orchestration
def supervisor_agent(user_query: str, state: AgentState) -> AgentState:
    
    # Step 1: Parse user query for companies, industry, and time period
    state = parse_user_query(user_query, state)

    # Step 2: Agent Orchestration Loop
    while state["next_agent"] != "end":
        if state["next_agent"] == "finmetrics_agent":
            st.session_state['status'] = "Running Financial analysis..."
            state = run_fin(state)
            state["task_status"] = "metrics_complete"
            state["next_agent"] = "sentiment_agent"
            st.session_state['progress'] += 0.33
        if state["next_agent"] == "sentiment_agent":
            st.session_state['status'] = "Performing Sentiment analysis..."
            state = run_sent(state)
            state["task_status"] = "sentiment_complete"
            state["next_agent"] = "summariser_agent"
            st.session_state['progress'] += 0.33
        if state["next_agent"] == "summariser_agent":
            st.session_state['status'] = "Generating comprehensive report..."
            state = run_summ(state)
            state["task_status"] = "summary_complete"
            state["next_agent"] = "end"
            st.session_state['progress'] += 0.34
        else:
            raise Exception(f"Unknown agent: {state['next_agent']}")

    return state
    
# Streamlit UI

def main():
    load_dotenv()

    st.set_page_config(
        page_title="Competitive Market Intelligence Agent",
        page_icon="📊",
        layout="wide",
    )

    # CSS
    st.markdown("""
        <style>
            html, body, [class*="css"]  {
                font-family: 'Segoe UI', sans-serif;
                color: #222;
                background-color: #f7f9fc;
            }
            .report-container {
                padding: 1.5rem;
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                margin-bottom: 2rem;
            }
            .center-text {
                text-align: center;
            }
            .status-bar {
                color: #1f77b4;
                font-weight: 500;
            }
            .stProgress > div > div > div > div {
                background-color: #1f77b4;
            }
            .stButton>button {
                background-color: #1f77b4;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #125e91;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header
    with st.container():
        st.markdown("<div class='center-text'><h1>🔍 Competitive Market Intelligence Agent</h1></div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='center-text' style='font-size:18px; margin-top:-10px;'>
                This Agent analyzes the financials and sentiments of companies to help you make informed investing decisions. Simply enter the list of companies, industry and time period that you wish to analyze and receive a concise business report with just the information you need for your next trade. Additionally, we have provided a finance report and sentiment report if you want to focus on those areas specifically.
                <br><br>
                <i>Example:</i> <b>Analyze Apple, Microsoft, and Google in the technology sector for the last quarter</b>
            </div>
        """, unsafe_allow_html=True)

    # Session State Initialization
    for key, default in {
        'report_generated': False,
        'report_content': "",
        'companies': [],
        'industry': "",
        'time_period': "",
        'status': "Ready",
        'progress': 0.0
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown("---")

    # Input section
    st.markdown("<h3 class='center-text'>📝 Enter Your Query</h3>", unsafe_allow_html=True)
    with st.container():
        with st.form("query_form"):
            user_query = st.text_area(
                "",
                placeholder="E.g., Analyze Apple, Microsoft, and Google in the technology sector for the last quarter",
                height=100
            )
            submitted = st.form_submit_button("🚀 Generate Report")

    if submitted and user_query:
        st.session_state.update({
            'report_generated': False,
            'progress': 0.0,
            'status': "Starting analysis..."
        })

        # Init a random state --> parse_user_query will update it anyways
        state: AgentState = {
            "messages": [],
            "next_agent": "",
            "industry": "",
            "companies": [],
            "time_period": 30,
            "task_status": "",
            "report": {}
        }

        try:
            with st.spinner("🧠 Analyzing..."):
                final_state = supervisor_agent(user_query, state)

                st.session_state.update({
                    'companies': final_state['companies'],
                    'industry': final_state['industry'],
                    'time_period': final_state['time_period'],
                    'report_content': final_state.get("report", {}).get("content", ""),
                    'report_generated': True,
                    'status': "Complete"
                })

                # Summarizer report is stored in BI_Report_{companies}.md
                filename = "BI_Report_" + "_".join(final_state["companies"]) + ".md"
                with open(filename, "w") as f:
                    f.write(st.session_state['report_content'])

                st.success("✅ Report generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            st.session_state.update({'status': "Ready", 'progress': 0.0})

    # Progress bar --> work on this later if poss
    if st.session_state['status'] not in ["Ready", "Complete"]:
        st.markdown(f"<p class='status-bar'>🔄 {st.session_state['status']}</p>", unsafe_allow_html=True)
        st.progress(st.session_state['progress'])

    # Report display
    if st.session_state['report_generated']:
        st.markdown("---")
        st.subheader("📋 Analysis Parameters")

        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Companies:** {', '.join(st.session_state['companies'])}")
            with col2:
                st.markdown(f"**Industry:** {st.session_state['industry']}")
            with col3:
                st.markdown(f"**Time Period:** {st.session_state['time_period']}")

        st.markdown("---")
        st.subheader("📊 Business Intelligence Report")

        # Function to display .md files with images on streamlit
        render_markdown_with_images(st.session_state['report_content'])

        try:
            # Convert .md file to pdf for downloading
            pdf_data = markdown_to_pdf(st.session_state['report_content'])

            base_name = "_".join(st.session_state['companies'])
            pdf_file = f"BI_Report_{base_name}.pdf"
            md_file = f"BI_Report_{base_name}.md"

            with open("sentiment_report.md", "r") as file:
                sentiment_report = file.read()

            sentiment_pdf = markdown_to_pdf(sentiment_report)
            sentiment_file = f"Sentiment_Report_{base_name}.pdf"
            sentiment_md = f"Sentiment_Report_{base_name}.md"

            with open("financial_analysis_report.md", "r") as file:
                finance_report = file.read()

            finance_pdf = markdown_to_pdf(finance_report)
            finance_file = f"Financial_Analysis_Report_{base_name}.pdf"
            finance_md = f"Financial_Analysis_Report_{base_name}.md"

            st.download_button(
                label="📥 Download Executive Report as PDF",
                data=pdf_data,
                file_name=pdf_file,
                mime="application/pdf"
            )

            st.download_button(
                label="📥 Download Sentiment Report as PDF",
                data= sentiment_pdf,
                file_name= sentiment_file,
                mime = "application/pdf"
            )

            st.download_button(
                label="📥 Download Finance Report as PDF",
                data= finance_pdf,
                file_name= finance_file,
                mime = "application/pdf"
            )

        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {str(e)}")
            st.download_button(
                label="📥 Download Executive Report as Markdown",
                data=st.session_state['report_content'],
                file_name=md_file,
                mime="text/markdown"
            )
            st.download_button(
                label="📥 Download Sentiment Report as Markdown",
                data=sentiment_report,
                file_name=sentiment_md,
                mime="text/markdown"
            )
            st.download_button(
                label="📥 Download Finance Report as Markdown",
                data=finance_report,
                file_name=finance_md,
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()