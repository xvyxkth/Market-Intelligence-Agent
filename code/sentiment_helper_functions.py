import os
import json
import statistics
def parse_retrieved_data(symbols, industry):
    """
    Parse the retrieved JSON files and extract relevant information for analysis.
    Returns a structured dictionary with all the parsed data.
    """
    parsed_data = {
        "companies": {},
        "industry": {},
        "news_sentiment": {}
    }

    for symbol in symbols:
        parsed_data["companies"][symbol] = {}
        
        # get ompany overview data
        overview_file = f"retrieved_sentiment_data/{symbol}_get_company_overview.json"
        if os.path.exists(overview_file):
            with open(overview_file, 'r') as f:
                overview_data = json.load(f)
                
            parsed_data["companies"][symbol]["overview"] = {
                "name": overview_data.get("Name", symbol),
                "description": overview_data.get("Description", ""),
                "sector": overview_data.get("Sector", ""),
                "industry": overview_data.get("Industry", ""),
                "market_cap": overview_data.get("MarketCapitalization", ""),
                "pe_ratio": overview_data.get("PERatio", ""),
                "eps": overview_data.get("EPS", ""),
                "profit_margin": overview_data.get("ProfitMargin", ""),
                "quarterly_earnings_growth": overview_data.get("QuarterlyEarningsGrowthYOY", ""),
                "quarterly_revenue_growth": overview_data.get("QuarterlyRevenueGrowthYOY", ""),
                "analyst_target_price": overview_data.get("AnalystTargetPrice", ""),
                "beta": overview_data.get("Beta", ""),
                "52_week_high": overview_data.get("52WeekHigh", ""),
                "52_week_low": overview_data.get("52WeekLow", "")
            }
        
        # get company news data
        company_name = parsed_data["companies"][symbol].get("overview", {}).get("name", symbol)
        news_file = f"retrieved_sentiment_data/{company_name}_get_company_news.json"
        
        if os.path.exists(news_file):
            with open(news_file, 'r') as f:
                news_data = json.load(f)
            
            articles = []
            sentiment_scores = []
            
            if "articles" in news_data and "results" in news_data["articles"]:
                for article in news_data["articles"]["results"]:
                    articles.append({
                        "title": article.get("title", ""),
                        "date": article.get("dateTime", ""),
                        "url": article.get("url", ""),
                        "sentiment": article.get("sentiment", 0),
                        "source": article.get("source", {}).get("title", ""),
                        "summary": article.get("body", "")[:200] + "..." if article.get("body") else ""
                    })
                    
                    if article.get("sentiment") is not None:
                        sentiment_scores.append(article.get("sentiment"))

            avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            
            parsed_data["companies"][symbol]["news"] = {
                "articles": articles,
                "avg_sentiment": avg_sentiment,
                "sentiment_label": sentiment_label,
                "article_count": len(articles)
            }
    
    # get industry data
    industry_file = f"retrieved_sentiment_data/{industry}_get_industry_data.json"
    if os.path.exists(industry_file):
        with open(industry_file, 'r') as f:
            industry_data = json.load(f)
        
        industry_companies = []
        market_caps = []
        betas = []
        
        for company in industry_data:
            industry_companies.append({
                "symbol": company.get("symbol", ""),
                "name": company.get("companyName", ""),
                "market_cap": company.get("marketCap", 0),
                "beta": company.get("beta", 0),
                "price": company.get("price", 0)
            })
            
            if company.get("marketCap"):
                market_caps.append(company.get("marketCap"))
            
            if company.get("beta"):
                betas.append(company.get("beta"))
        
        parsed_data["industry"] = {
            "companies": industry_companies,
            "avg_market_cap": statistics.mean(market_caps) if market_caps else 0,
            "avg_beta": statistics.mean(betas) if betas else 0,
            "company_count": len(industry_companies)
        }
    
    # get news sentiment data
    sentiment_file = "retrieved_sentiment_data/get_news_sentiment.json"
    if os.path.exists(sentiment_file):
        with open(sentiment_file, 'r') as f:
            sentiment_data = json.load(f)
        
        ticker_sentiments = {symbol: [] for symbol in symbols}
        general_news = []
        
        if "feed" in sentiment_data:
            for news_item in sentiment_data["feed"]:
                if "ticker_sentiment" in news_item:
                    for ticker_data in news_item["ticker_sentiment"]:
                        ticker = ticker_data.get("ticker", "")
                        if ticker in symbols:
                            ticker_sentiments[ticker].append({
                                "title": news_item.get("title", ""),
                                "date": news_item.get("time_published", ""),
                                "sentiment_score": float(ticker_data.get("ticker_sentiment_score", 0)),
                                "sentiment_label": ticker_data.get("ticker_sentiment_label", ""),
                                "relevance_score": float(ticker_data.get("relevance_score", 0)),
                                "source": news_item.get("source", ""),
                                "summary": news_item.get("summary", "")[:200] + "..." if news_item.get("summary") else ""
                            })
                
                general_news.append({
                    "title": news_item.get("title", ""),
                    "date": news_item.get("time_published", ""),
                    "sentiment_score": float(news_item.get("overall_sentiment_score", 0)),
                    "sentiment_label": news_item.get("overall_sentiment_label", ""),
                    "source": news_item.get("source", ""),
                    "summary": news_item.get("summary", "")[:200] + "..." if news_item.get("summary") else ""
                })
        
        ticker_avg_sentiments = {}
        for ticker, items in ticker_sentiments.items():
            sentiment_scores = [item["sentiment_score"] for item in items if "sentiment_score" in item]
            avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            
            ticker_avg_sentiments[ticker] = {
                "avg_sentiment": avg_sentiment,
                "sentiment_label": sentiment_label,
                "news_count": len(items)
            }
        
        parsed_data["news_sentiment"] = {
            "ticker_specific": ticker_sentiments,
            "ticker_averages": ticker_avg_sentiments,
            "general_news": general_news
        }
    
    return parsed_data