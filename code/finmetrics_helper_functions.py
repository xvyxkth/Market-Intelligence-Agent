import json as json
import pandas as pd

# New functions for parsing and analyzing data

def load_json_data(symbol, file_type):
    """Load JSON data from file"""
    filename = f"retrieved_metrics_data/{symbol}_{file_type}.json"
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON in {filename}.")
        return None

def parse_income_statement(symbol):
    """Parse income statement data for a company"""
    data = load_json_data(symbol, "get_income_statement")
    if not data or "annualReports" not in data:
        print(f"No valid income statement data for {symbol}")
        return None
        
    results = []
    for report in data.get("annualReports", [])[:20]: 
        fiscal_date = report.get("fiscalDateEnding", "N/A")
        try:
            total_revenue = float(report.get("totalRevenue", 0))
            gross_profit = float(report.get("grossProfit", 0))
            operating_income = float(report.get("operatingIncome", 0))
            net_income = float(report.get("netIncome", 0))
            ebitda = float(report.get("ebitda", 0))
            
            gross_margin = (gross_profit / total_revenue) * 100 if total_revenue else 0
            operating_margin = (operating_income / total_revenue) * 100 if total_revenue else 0
            net_margin = (net_income / total_revenue) * 100 if total_revenue else 0
            
            results.append({
                "fiscal_date": fiscal_date,
                "total_revenue": total_revenue,
                "gross_profit": gross_profit,
                "operating_income": operating_income,
                "net_income": net_income,
                "ebitda": ebitda,
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "net_margin": net_margin
            })
        except (ValueError, TypeError) as e:
            print(f"Error processing income statement for {symbol} on {fiscal_date}: {e}")
            continue
    
    if len(results) >= 2:
        for i in range(len(results)-1):
            current_year = results[i]
            prev_year = results[i+1]
            
            revenue_growth = ((current_year["total_revenue"] - prev_year["total_revenue"]) / prev_year["total_revenue"]) * 100 if prev_year["total_revenue"] else 0
            net_income_growth = ((current_year["net_income"] - prev_year["net_income"]) / prev_year["net_income"]) * 100 if prev_year["net_income"] else 0
            
            results[i]["revenue_growth"] = revenue_growth
            results[i]["net_income_growth"] = net_income_growth
    
    return results

def parse_balance_sheet(symbol):
    """Parse balance sheet data for a company"""
    data = load_json_data(symbol, "get_balance_sheet")
    if not data or "annualReports" not in data:
        print(f"No valid balance sheet data for {symbol}")
        return None
        
    results = []
    for report in data.get("annualReports", [])[:20]: 
        fiscal_date = report.get("fiscalDateEnding", "N/A")
        try:
            total_assets = float(report.get("totalAssets", 0))
            total_liabilities = float(report.get("totalLiabilities", 0))
            total_equity = float(report.get("totalShareholderEquity", 0))
            current_assets = float(report.get("totalCurrentAssets", 0))
            current_liabilities = float(report.get("totalCurrentLiabilities", 0))
            cash_and_equivalents = float(report.get("cashAndCashEquivalentsAtCarryingValue", 0))
            short_term_debt = float(report.get("shortTermDebt", 0)) if report.get("shortTermDebt") else 0
            long_term_debt = float(report.get("longTermDebt", 0)) if report.get("longTermDebt") else 0
            
            current_ratio = current_assets / current_liabilities if current_liabilities else 0
            debt_to_equity = (short_term_debt + long_term_debt) / total_equity if total_equity else 0
            
            results.append({
                "fiscal_date": fiscal_date,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "total_equity": total_equity,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "cash_and_equivalents": cash_and_equivalents,
                "short_term_debt": short_term_debt,
                "long_term_debt": long_term_debt,
                "current_ratio": current_ratio,
                "debt_to_equity": debt_to_equity
            })
        except (ValueError, TypeError) as e:
            print(f"Error processing balance sheet for {symbol} on {fiscal_date}: {e}")
            continue
    
    return results

def parse_stock_price(symbol):
    """Parse weekly stock price data for a company"""
    data = load_json_data(symbol, "get_stock_price_weekly")
    if not data or "Weekly Time Series" not in data:
        print(f"No valid stock price data for {symbol}")
        return None
    
    time_series = data.get("Weekly Time Series", {})
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    df = df.tail(52)
    
    df['weekly_return'] = df['close'].pct_change() * 100
    df['volatility'] = df['weekly_return'].rolling(window=4).std()
    
    start_price = df['close'].iloc[0] if not df.empty else 0
    end_price = df['close'].iloc[-1] if not df.empty else 0
    price_change = end_price - start_price
    percent_change = (price_change / start_price) * 100 if start_price else 0
    
    min_price = df['low'].min() if not df.empty else 0
    max_price = df['high'].max() if not df.empty else 0
    avg_volume = df['volume'].mean() if not df.empty else 0
    
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    results = {
        "weekly_data": df.reset_index().to_dict('records'),
        "summary": {
            "start_price": start_price,
            "end_price": end_price,
            "price_change": price_change,
            "percent_change": percent_change,
            "min_price": min_price,
            "max_price": max_price,
            "avg_volume": avg_volume,
            "latest_close": end_price,
            "avg_weekly_return": df['weekly_return'].mean() if not df.empty else 0,
            "volatility": df['weekly_return'].std() if not df.empty else 0
        }
    }
    
    return results

def parse_company_financials(symbol):
    """Parse company financials data"""
    data = load_json_data(symbol, "get_company_financials")
    if not data or not isinstance(data, list) or len(data) == 0:
        print(f"No valid company financials data for {symbol}")
        return None
    
    company_data = data[0]
    
    results = {
        "company_name": company_data.get("companyName", "N/A"),
        "sector": company_data.get("sector", "N/A"),
        "industry": company_data.get("industry", "N/A"),
        "market_cap": company_data.get("mktCap", 0),
        "beta": company_data.get("beta", 0),
        "price": company_data.get("price", 0),
        "changes": company_data.get("changes", 0),
        "changes_percent": (company_data.get("changes", 0) / company_data.get("price", 1)) * 100 if company_data.get("price", 0) else 0,
        "exchange": company_data.get("exchange", "N/A"),
        "currency": company_data.get("currency", "USD"),
        "description": company_data.get("description", "N/A"),
        "ceo": company_data.get("ceo", "N/A"),
        "employees": company_data.get("fullTimeEmployees", "N/A"),
        "country": company_data.get("country", "N/A"),
        "ipo_date": company_data.get("ipoDate", "N/A"),
        "dcf": company_data.get("dcf", 0),
        "last_div": company_data.get("lastDiv", 0)
    }
    
    return results

def parse_financial_ratios(symbol):
    """Parse financial ratios data for a company"""
    data = load_json_data(symbol, "get_financial_ratios")
    if not data or not isinstance(data, list) or len(data) == 0:
        print(f"No valid financial ratios data for {symbol}")
        return None
    
    results = []
    for report in data:
        date = report.get("date", "N/A")
        period = report.get("period", "N/A")
        
        result = {
            "date": date,
            "period": period,
            "calendar_year": report.get("calendarYear", "N/A"),
            
            # Liquidity Ratios
            "current_ratio": report.get("currentRatio", 0),
            "quick_ratio": report.get("quickRatio", 0),
            "cash_ratio": report.get("cashRatio", 0),
            
            # Profitability Ratios
            "gross_profit_margin": report.get("grossProfitMargin", 0),
            "operating_profit_margin": report.get("operatingProfitMargin", 0),
            "net_profit_margin": report.get("netProfitMargin", 0),
            "return_on_assets": report.get("returnOnAssets", 0),
            "return_on_equity": report.get("returnOnEquity", 0),
            "return_on_capital_employed": report.get("returnOnCapitalEmployed", 0),
            
            # Debt Ratios
            "debt_ratio": report.get("debtRatio", 0),
            "debt_equity_ratio": report.get("debtEquityRatio", 0),
            "long_term_debt_to_capitalization": report.get("longTermDebtToCapitalization", 0),
            "interest_coverage": report.get("interestCoverage", 0),
            
            # Efficiency Ratios
            "asset_turnover": report.get("assetTurnover", 0),
            "inventory_turnover": report.get("inventoryTurnover", 0),
            "receivables_turnover": report.get("receivablesTurnover", 0),
            
            # Valuation Ratios
            "price_to_earnings": report.get("priceEarningsRatio", 0),
            "price_to_book": report.get("priceToBookRatio", 0),
            "price_to_sales": report.get("priceToSalesRatio", 0),
            "dividend_yield": report.get("dividendYield", 0),
            "enterprise_value_multiple": report.get("enterpriseValueMultiple", 0),
            "price_fair_value": report.get("priceFairValue", 0)
        }
        
        results.append(result)
    
    return results

def calculate_industry_averages(companies_data):
    """Calculate industry average metrics"""
    
    profit_margins = {}
    returns = {}
    liquidity = {}
    debt = {}
    valuation = {}
    stock_perf = {}
    
    count = 0
    
    for symbol, data in companies_data.items():
        count += 1
        
        if data.get('income_statement') and len(data['income_statement']) > 0:
            latest_income = data['income_statement'][0]
            profit_margins.setdefault('gross_margin', 0)
            profit_margins.setdefault('operating_margin', 0)
            profit_margins.setdefault('net_margin', 0)
            
            profit_margins['gross_margin'] += latest_income.get('gross_margin', 0)
            profit_margins['operating_margin'] += latest_income.get('operating_margin', 0)
            profit_margins['net_margin'] += latest_income.get('net_margin', 0)
        
        if data.get('financial_ratios') and len(data['financial_ratios']) > 0:
            latest_ratios = data['financial_ratios'][0]
            returns.setdefault('return_on_assets', 0)
            returns.setdefault('return_on_equity', 0)
            
            returns['return_on_assets'] += latest_ratios.get('return_on_assets', 0)
            returns['return_on_equity'] += latest_ratios.get('return_on_equity', 0)
        
        if data.get('balance_sheet') and len(data['balance_sheet']) > 0:
            latest_balance = data['balance_sheet'][0]
            liquidity.setdefault('current_ratio', 0)
            liquidity.setdefault('debt_to_equity', 0)
            
            liquidity['current_ratio'] += latest_balance.get('current_ratio', 0)
            liquidity['debt_to_equity'] += latest_balance.get('debt_to_equity', 0)
        
        if data.get('financial_ratios') and len(data['financial_ratios']) > 0:
            latest_ratios = data['financial_ratios'][0]
            debt.setdefault('debt_ratio', 0)
            debt.setdefault('debt_equity_ratio', 0)
            debt.setdefault('interest_coverage', 0)
            
            debt['debt_ratio'] += latest_ratios.get('debt_ratio', 0)
            debt['debt_equity_ratio'] += latest_ratios.get('debt_equity_ratio', 0)
            debt['interest_coverage'] += latest_ratios.get('interest_coverage', 0)
        
        if data.get('financial_ratios') and len(data['financial_ratios']) > 0:
            latest_ratios = data['financial_ratios'][0]
            valuation.setdefault('price_to_earnings', 0)
            valuation.setdefault('price_to_book', 0)
            valuation.setdefault('price_to_sales', 0)
            valuation.setdefault('enterprise_value_multiple', 0)
            
            valuation['price_to_earnings'] += latest_ratios.get('price_to_earnings', 0)
            valuation['price_to_book'] += latest_ratios.get('price_to_book', 0)
            valuation['price_to_sales'] += latest_ratios.get('price_to_sales', 0)
            valuation['enterprise_value_multiple'] += latest_ratios.get('enterprise_value_multiple', 0)
        
        if data.get('stock_price') and data['stock_price'].get('summary'):
            stock_perf.setdefault('percent_change', 0)
            stock_perf.setdefault('volatility', 0)
            
            stock_perf['percent_change'] += data['stock_price']['summary'].get('percent_change', 0)
            stock_perf['volatility'] += data['stock_price']['summary'].get('volatility', 0)
    
    if count > 0:
        for key in profit_margins:
            profit_margins[key] /= count
        for key in returns:
            returns[key] /= count
        for key in liquidity:
            liquidity[key] /= count
        for key in debt:
            debt[key] /= count
        for key in valuation:
            valuation[key] /= count
        for key in stock_perf:
            stock_perf[key] /= count
    
    return {
        'profit_margins': profit_margins,
        'returns': returns,
        'liquidity': liquidity,
        'debt': debt,
        'valuation': valuation,
        'stock_perf': stock_perf
    }

def generate_company_summary(company_data, industry_averages):
    """Generate a summary analysis for a company"""
    summary = {}
    
    if company_data.get('company_info'):
        summary['name'] = company_data['company_info'].get('company_name', 'N/A')
        summary['sector'] = company_data['company_info'].get('sector', 'N/A')
        summary['industry'] = company_data['company_info'].get('industry', 'N/A')
        summary['market_cap'] = company_data['company_info'].get('market_cap', 0)
        summary['beta'] = company_data['company_info'].get('beta', 0)
    else:
        summary['name'] = 'N/A'
        summary['sector'] = 'N/A'
        summary['industry'] = 'N/A'
    
    summary['financials'] = {}
    if company_data.get('income_statement') and len(company_data['income_statement']) > 0:
        latest_income = company_data['income_statement'][0]
        summary['financials']['revenue'] = latest_income.get('total_revenue', 0)
        summary['financials']['net_income'] = latest_income.get('net_income', 0)
        summary['financials']['gross_margin'] = latest_income.get('gross_margin', 0)
        summary['financials']['operating_margin'] = latest_income.get('operating_margin', 0)
        summary['financials']['net_margin'] = latest_income.get('net_margin', 0)
        
        if industry_averages.get('profit_margins'):
            summary['financials']['gross_margin_vs_industry'] = latest_income.get('gross_margin', 0) - industry_averages['profit_margins'].get('gross_margin', 0)
            summary['financials']['operating_margin_vs_industry'] = latest_income.get('operating_margin', 0) - industry_averages['profit_margins'].get('operating_margin', 0)
            summary['financials']['net_margin_vs_industry'] = latest_income.get('net_margin', 0) - industry_averages['profit_margins'].get('net_margin', 0)
    
    summary['health'] = {}
    if company_data.get('balance_sheet') and len(company_data['balance_sheet']) > 0:
        latest_balance = company_data['balance_sheet'][0]
        summary['health']['current_ratio'] = latest_balance.get('current_ratio', 0)
        summary['health']['debt_to_equity'] = latest_balance.get('debt_to_equity', 0)
        
        if industry_averages.get('liquidity'):
            summary['health']['current_ratio_vs_industry'] = latest_balance.get('current_ratio', 0) - industry_averages['liquidity'].get('current_ratio', 0)
            summary['health']['debt_to_equity_vs_industry'] = latest_balance.get('debt_to_equity', 0) - industry_averages['liquidity'].get('debt_to_equity', 0)
    
    summary['stock'] = {}
    if company_data.get('stock_price') and company_data['stock_price'].get('summary'):
        stock_summary = company_data['stock_price']['summary']
        summary['stock']['latest_price'] = stock_summary.get('end_price', 0)
        summary['stock']['price_change'] = stock_summary.get('percent_change', 0)
        summary['stock']['volatility'] = stock_summary.get('volatility', 0)
        
        if industry_averages.get('stock_perf'):
            summary['stock']['price_change_vs_industry'] = stock_summary.get('percent_change', 0) - industry_averages['stock_perf'].get('percent_change', 0)
            summary['stock']['volatility_vs_industry'] = stock_summary.get('volatility', 0) - industry_averages['stock_perf'].get('volatility', 0)

    summary['valuation'] = {}
    if company_data.get('financial_ratios') and len(company_data['financial_ratios']) > 0:
        latest_ratios = company_data['financial_ratios'][0]
        summary['valuation']['pe_ratio'] = latest_ratios.get('price_to_earnings', 0)
        summary['valuation']['pb_ratio'] = latest_ratios.get('price_to_book', 0)
        summary['valuation']['ps_ratio'] = latest_ratios.get('price_to_sales', 0)
        summary['valuation']['ev_ebitda'] = latest_ratios.get('enterprise_value_multiple', 0)
        
        if industry_averages.get('valuation'):
            summary['valuation']['pe_ratio_vs_industry'] = latest_ratios.get('price_to_earnings', 0) - industry_averages['valuation'].get('price_to_earnings', 0)
            summary['valuation']['pb_ratio_vs_industry'] = latest_ratios.get('price_to_book', 0) - industry_averages['valuation'].get('price_to_book', 0)
            summary['valuation']['ps_ratio_vs_industry'] = latest_ratios.get('price_to_sales', 0) - industry_averages['valuation'].get('price_to_sales', 0)
            summary['valuation']['ev_ebitda_vs_industry'] = latest_ratios.get('enterprise_value_multiple', 0) - industry_averages['valuation'].get('enterprise_value_multiple', 0)
    
    summary['returns'] = {}
    if company_data.get('financial_ratios') and len(company_data['financial_ratios']) > 0:
        latest_ratios = company_data['financial_ratios'][0]
        summary['returns']['roa'] = latest_ratios.get('return_on_assets', 0)
        summary['returns']['roe'] = latest_ratios.get('return_on_equity', 0)
        
        if industry_averages.get('returns'):
            summary['returns']['roa_vs_industry'] = latest_ratios.get('return_on_assets', 0) - industry_averages['returns'].get('return_on_assets', 0)
            summary['returns']['roe_vs_industry'] = latest_ratios.get('return_on_equity', 0) - industry_averages['returns'].get('return_on_equity', 0)
    
    summary['strengths'] = []
    summary['weaknesses'] = []
    summary['opportunities'] = []
    summary['threats'] = []
    
    if summary.get('financials'):
        if summary['financials'].get('net_margin_vs_industry', 0) > 2:
            summary['strengths'].append("Superior profit margins compared to industry")
        elif summary['financials'].get('net_margin_vs_industry', 0) < -2:
            summary['weaknesses'].append("Below average profit margins")
    
    if summary.get('health'):
        if summary['health'].get('current_ratio', 0) > 2:
            summary['strengths'].append("Strong liquidity position")
        elif summary['health'].get('current_ratio', 0) < 1:
            summary['weaknesses'].append("Potential liquidity concerns")
            
        if summary['health'].get('debt_to_equity', 0) < 0.5:
            summary['strengths'].append("Low leverage/debt")
        elif summary['health'].get('debt_to_equity', 0) > 1.5:
            summary['weaknesses'].append("High debt levels")
    
    if summary.get('stock'):
        if summary['stock'].get('price_change', 0) > 15:
            summary['strengths'].append("Strong stock performance")
        elif summary['stock'].get('price_change', 0) < -15:
            summary['weaknesses'].append("Poor stock performance")
    
    if summary.get('valuation'):
        pe_ratio = summary['valuation'].get('pe_ratio', 0)
        if pe_ratio > 0:
            if pe_ratio < summary['valuation'].get('pe_ratio_vs_industry', pe_ratio):
                summary['opportunities'].append("Potentially undervalued compared to peers")
            elif pe_ratio > summary['valuation'].get('pe_ratio_vs_industry', pe_ratio) * 1.5:
                summary['threats'].append("Potentially overvalued compared to peers")
    
    return summary