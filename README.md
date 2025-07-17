# 📊 Market-Intelligence-Agent

**Market-Intelligence-Agent** is a multi-agent system that generates a detailed business intelligence report based on both financial and sentiment analysis for a list of companies in a specific industry over a defined time period. The final output is a comprehensive, data-rich report presented via a Streamlit web interface and available for download in three separate PDF files.

## 🚀 Features

* ✅ Multi-agent architecture with distinct responsibilities
* ✅ Real-time sentiment analysis from news sources
* ✅ In-depth financial metric analysis and visualizations
* ✅ Consolidated business intelligence reporting
* ✅ Streamlit-based interactive web interface
* ✅ Downloadable reports in PDF format

## 📁 Downloadable Reports

Upon completing the analysis, the system generates **three** downloadable PDF reports:

1. **📈 Financial Analysis Report**
   Detailed breakdown of financial metrics for each company with visualizations.

2. **📰 Sentiment Analysis Report**
   Real-time sentiment insights from industry and company-specific news sources.

3. **📊 Comprehensive Business Intelligence Report**
   Merges insights from financial and sentiment analyses with additional summaries, charts, and business intelligence recommendations.

---

## 🧠 Multi-Agent System Architecture

The project is powered by four specialized agents working together:

### 🔹 Supervisor Agent

* Parses user input (industry, company list, time range)
* Orchestrates the analysis pipeline across agents

### 🔹 Financial Analysis Agent

* Fetches and analyzes financial metrics for companies
* Generates plots and tabular insights
* Produces the Financial Analysis PDF

### 🔹 Sentiment Analysis Agent

* Gathers and evaluates news and sentiment data from various sources
* Produces the Sentiment Analysis PDF

### 🔹 Summarizer Agent

* Synthesizes data from both financial and sentiment analyses
* Produces the final Business Intelligence PDF

---

## ⚙️ Workflow Overview

1. **User Input**
   The user enters a query specifying the industry, companies, and date range.

2. **Financial Analysis**
   Financial data is collected and analyzed for each company.

3. **Sentiment Analysis**
   Industry-related news is analyzed for sentiment trends.

4. **Report Generation**
   All data is compiled into separate PDF reports and a combined business intelligence dashboard viewable in Streamlit.

---

## 🖥️ Streamlit Web App

An interactive Streamlit interface displays the reports and provides PDF download buttons for:

* Financial Analysis
* Sentiment Analysis
* Business Intelligence Summary
