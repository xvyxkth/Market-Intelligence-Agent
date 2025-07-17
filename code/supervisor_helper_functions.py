import os
import streamlit as st
import json
import re
from typing import List, Dict, Any, TypedDict
import base64
from io import BytesIO
from PIL import Image
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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

# List of allowed companies
company_to_ticker = {
    "Apple Inc.": "AAPL", 
    "Tesla, Inc.": "TSLA",
    "Amazon.com, Inc.": "AMZN",
    "Microsoft Corporation": "MSFT", 
    "NVIDIA Corporation": "NVDA",
    "Alphabet Inc. (Google)": "GOOGL",
    "Meta Platforms, Inc.": "META",
    "Netflix, Inc.": "NFLX",
    "JPMorgan Chase & Co.": "JPM", 
    "Visa Inc.": "V",
    "Bank of America Corporation": "BAC",
    "Advanced Micro Devices, Inc.": "AMD",
    "PayPal Holdings, Inc.": "PYPL",
    "The Walt Disney Company": "DIS",
    "AT&T Inc.": "T",
    "Pfizer Inc.": "PFE",
    "Costco Wholesale Corporation": "COST",
    "Intel Corporation": "INTC",
    "The Coca-Cola Company": "KO",
    "Target Corporation": "TGT",
    "NIKE, Inc.": "NKE",
    "SPDR S&P 500 ETF Trust": "SPY",
    "The Boeing Company": "BA",
    "Alibaba Group Holding Limited": "BABA",
    "Exxon Mobil Corporation": "XOM",
    "Walmart Inc.": "WMT",
    "General Electric Company": "GE",
    "Cisco Systems, Inc.": "CSCO",
    "Verizon Communications Inc.": "VZ",
    "Johnson & Johnson": "JNJ",
    "Chevron Corporation": "CVX",
    "Palantir Technologies Inc.": "PLTR",
    "Block, Inc. (Square)": "SQ",
    "Shopify Inc.": "SHOP",
    "Starbucks Corporation": "SBUX",
    "SoFi Technologies, Inc.": "SOFI",
    "Robinhood Markets, Inc.": "HOOD",
    "Roblox Corporation": "RBLX",
    "Snap Inc.": "SNAP",
    "Advanced Micro Devices, Inc.": "AMD",
    "Uber Technologies, Inc.": "UBER",
    "FedEx Corporation": "FDX",
    "AbbVie Inc.": "ABBV",
    "Etsy, Inc.": "ETSY",
    "Moderna, Inc.": "MRNA",
    "Lockheed Martin Corporation": "LMT",
    "General Motors Company": "GM",
    "Ford Motor Company": "F",
    "Rivian Automotive, Inc.": "RIVN",
    "Lucid Group, Inc.": "LCID",
    "Carnival Corporation & plc": "CCL",
    "Delta Air Lines, Inc.": "DAL",
    "United Airlines Holdings, Inc.": "UAL",
    "American Airlines Group Inc.": "AAL",
    "Taiwan Semiconductor Manufacturing Company Limited": "TSM",
    "Sony Group Corporation": "SONY",
    "Energy Transfer LP": "ET",
    "Nokia Corporation": "NOK",
    "Marathon Oil Corporation": "MRO",
    "Coinbase Global, Inc.": "COIN",
    "Sirius XM Holdings Inc.": "SIRI",
    "Riot Platforms, Inc.": "RIOT",
    "Cardiol Therapeutics Inc.": "CPRX",
    "Vanguard FTSE Emerging Markets ETF": "VWO",
    "SPDR Portfolio S&P 500 Growth ETF": "SPYG",
    "Roku, Inc.": "ROKU",
    "Paramount Global": "VIAC",
    "Activision Blizzard, Inc.": "ATVI",
    "Baidu, Inc.": "BIDU",
    "DocuSign, Inc.": "DOCU",
    "Zoom Video Communications, Inc.": "ZM",
    "Pinterest, Inc.": "PINS",
    "Tilray Brands, Inc.": "TLRY",
    "Walgreens Boots Alliance, Inc.": "WBA",
    "MGM Resorts International": "MGM",
    "NIO Inc.": "NIO",
    "Citigroup Inc.": "C",
    "The Goldman Sachs Group, Inc.": "GS",
    "Wells Fargo & Company": "WFC",
    "Adobe Inc.": "ADBE",
    "PepsiCo, Inc.": "PEP",
    "UnitedHealth Group Incorporated": "UNH",
    "Carrier Global Corporation": "CARR",
    "fuboTV Inc.": "FUBO",
    "HCA Healthcare, Inc.": "HCA",
    "Bilibili Inc.": "BILI",
    "Rocket Companies, Inc.": "RKT",
    "Twitter Inc.": "TWTR"
}

allowed_industries = {
    "Technology": "The Technology industry comprises companies that develop or distribute technological products and services, including software, hardware, semiconductors, and IT services. This sector drives innovation in areas like artificial intelligence, cloud computing, and consumer electronics. It is characterized by rapid growth and frequent disruption.",
    
    "Financial Services": "This industry includes firms involved in banking, investment, insurance, and wealth management. It plays a crucial role in capital allocation, risk management, and financial intermediation. Institutions here help individuals and businesses manage money and access credit markets.",
    
    "Utilities": "The Utilities sector encompasses companies that provide essential services such as electricity, gas, and water. These firms often operate as regulated monopolies due to infrastructure costs and public interest. It is a defensive sector, less sensitive to economic cycles.",
    
    "Materials": "The Materials industry includes companies involved in the discovery, development, and processing of raw materials. This includes mining, chemicals, construction materials, and packaging. The sector is cyclical and closely tied to global industrial demand and commodity prices.",
    
    "Consumer Defensive": "This industry covers companies that produce goods and services essential to everyday life, such as food, beverages, and household products. These businesses tend to be stable and resilient, performing well even during economic downturns. Often referred to as 'staples'.",
    
    "Consumer Cyclical": "The Consumer Cyclical industry includes businesses that sell non-essential goods and services such as cars, apparel, and leisure activities. Demand in this sector fluctuates with economic conditions. It typically performs well in periods of economic expansion.",
    
    "Communication Services": "This sector includes companies that facilitate communication and provide content and entertainment, such as telecom providers, media companies, and internet platforms. The industry is evolving rapidly due to streaming, digital advertising, and mobile technology.",
    
    "Healthcare": "The Healthcare industry consists of providers of medical services, pharmaceuticals, biotechnology, and medical equipment. It serves a fundamental societal need and is driven by aging populations, innovation, and regulation. The sector includes both defensive and growth-oriented businesses.",
    
    "Industrials": "Industrials encompass a wide range of businesses involved in manufacturing, construction, aerospace, defense, and logistics. These companies provide the backbone for economic infrastructure. The sector is cyclical and sensitive to business investment and global trade.",
    
    "Real Estate": "The Real Estate industry includes companies that develop, own, or manage properties such as commercial buildings, residential complexes, and REITs. It is influenced by interest rates, urbanization trends, and economic conditions. The sector offers both growth and income opportunities.",
    
    "Energy": "The Energy sector involves companies that explore, produce, and refine oil, gas, and renewable energy sources. It plays a critical role in powering economies and industries. This sector is highly cyclical and affected by geopolitical events and commodity prices."
}

def parse_user_query(user_query, state):
    llm = get_llm()
    llm_ind = get_llm()
    # Get companies, industry and tp from query
    system_prompt = (
        "You are an expert business intelligence workflow orchestrator. "
        "Given a user query, extract and return the following as JSON: "
        "1. companies (list of company tickers or names), "
        "2. industry (string), "
        "3. time_period (string). For time_period, Extract the number of days whch is either directly given or use the following guide : A quarter is 120 days. A month is 30 days. A week is 7 days. A year is 365 days."
    )
    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {user_query}")
    ]
    result = llm.invoke(prompt)
    import re # This has to be here only, try catch can't reference it otherwise
    try:
        parsed = json.loads(result.content)
    except Exception:
        # try to extract JSON from LLM output if parsing fails
        match = re.search(r"\{.*\}", result.content, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return valid JSON for query parsing.")
        parsed = json.loads(match.group(0))
    # map companies to symbols
    companies = parsed["companies"]
    
    # system prompt for company to ticker
    ticker_system_prompt = (
        "You are an expert financial data analyst. "
        "Based on the provided dictionary of company names to ticker symbols, "
        "return the ticker symbol for each company in the list. "
        "IMPORTANT: Users often provide abbreviated or common names for companies rather than full legal names. "
        "For example, 'Apple' should match 'Apple Inc.', 'Google' should match 'Alphabet Inc. (Google)', "
        "'Uber' should match 'Uber Technologies, Inc.', etc. "
        "Make these connections intelligently, looking for partial matches when appropriate. "
        "If a company truly is not found in the dictionary, indicate it as 'NOT_FOUND'. "
        "Return ONLY a JSON array of ticker symbols in the same order as the input companies."
    )
    
    ticker_prompt = [
        SystemMessage(content=ticker_system_prompt),
        HumanMessage(content=f"Company Names: {companies}\nCompany to Ticker Dictionary: {company_to_ticker}")
    ]
    
    ticker_result = llm.invoke(ticker_prompt)
    try:
        ticker_symbols = json.loads(ticker_result.content)
    except Exception:
        match = re.search(r"\[.*\]", ticker_result.content, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return valid JSON for ticker resolution.")
        ticker_symbols = json.loads(match.group(0))
    
    # handle companies not in list
    valid_tickers = []
    not_found_companies = []
    
    for i, ticker in enumerate(ticker_symbols):
        if ticker == "NOT_FOUND":
            not_found_companies.append(companies[i])
        else:
            valid_tickers.append(ticker)
    
    # if you can't find companies
    if not_found_companies:
        print(f"Error: Not all companies were found in the company_to_ticker dictionary.")
    
    # map industry to allowed set
    industry = parsed["industry"]

    industry_system_prompt = """You are an intelligent classification assistant designed to identify the most suitable industry from a predefined list of industries, based on a user-provided industry input. You are provided with a dictionary where:

    - Keys represent allowed industries.
    - Values are descriptive summaries (3–4 lines) of what each industry represents.

    You will receive an input variable called "industry" from the user, which may be imprecise, colloquial, or use non-standard terminology. Your task is to compare the user's input against the industry descriptions and select the closest matching industry key based on contextual meaning and relevance.

    Instructions:
    1. Compare the input "industry" semantically with the values (descriptions) in the dictionary.
    2. Select the key whose description best matches the user input in terms of function, domain, and activities.
    3. Return only the most suitable key (industry name) from the dictionary.
    4. If no good match is found, return "Unknown".

    Be especially careful to consider synonyms, sub-industry terms, and conceptual overlap when evaluating relevance."""

    industry_prompt = [
        SystemMessage(content=industry_system_prompt),
        HumanMessage(content=f"Input industry: {industry}\nIndustries and their Descriptions: {allowed_industries}")
    ]

    llm_res = llm_ind.invoke(industry_prompt)
    valid_industry = llm_res.content
    # Update state with parsed info
    state["companies"] = valid_tickers  # replace company names with ticker symbols
    state["industry"] = valid_industry
    state["time_period"] = parsed["time_period"]
    state["messages"].append({"role": "user", "content": user_query})
    state["task_status"] = "parsed_user_query"
    state["next_agent"] = "finmetrics_agent"
    
    return state

def render_markdown_with_images(markdown_text):
    """
    Renders markdown text with embedded base64 images properly in Streamlit
    without showing the base64 data in the text
    """

    # for clean printing, removes all triple backticks
    markdown_text = re.sub(r'```[a-zA-Z0-9]*\s*\n?', '', markdown_text)
    markdown_text = re.sub(r'\n?```\s*', '\n', markdown_text)

    # match condn --> pattern should match the entire image markdown syntax
    image_pattern = r'!\[(.*?)\]\((data:image\/(\w+);base64,([^)]+))\)'
    
    # Find all matches
    matches = list(re.finditer(image_pattern, markdown_text))
    
    # If no images then just render the markdown
    if not matches:
        st.markdown(markdown_text)
        return
    
    last_end = 0
    
    for match in matches:
        start, end = match.span()
        
        # Render the text chunk before the current image
        if start > last_end:
            st.markdown(markdown_text[last_end:start])
        
        # image information
        alt_text = match.group(1)
        base64_str = match.group(4)
        
        try:
            # display the image
            img_data = base64.b64decode(base64_str)
            st.image(
                Image.open(BytesIO(img_data)),
                caption=alt_text,
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Failed to render image '{alt_text}': {e}")
        
        # Update last_end
        last_end = end
    
    # for text after last image
    if last_end < len(markdown_text):
        st.markdown(markdown_text[last_end:])

def markdown_to_pdf(markdown_text):
    """
    Convert markdown text with embedded base64 images to PDF
    Returns the PDF as bytes
    """

    markdown_text = re.sub(r'```[a-zA-Z0-9]*\s*\n?', '', markdown_text)
    markdown_text = re.sub(r'\n?```\s*', '\n', markdown_text)
    
    # buffer to receive pdf data
    buffer = BytesIO()
    
    # margins
    doc = SimpleDocTemplate(buffer, 
                           pagesize=letter, 
                           rightMargin=0.75*inch,
                           leftMargin=0.75*inch,
                           topMargin=0.75*inch,
                           bottomMargin=0.75*inch)
    
    elements = []
    
    styles = getSampleStyleSheet()
    
    # custom paragraph styling for clean repr
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=16,
        textColor=colors.darkblue
    )
    
    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.darkblue,
        borderWidth=0,
        borderColor=colors.grey,
        borderPadding=0
    )
    
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.darkcyan,
        borderWidth=0,
        borderPadding=0
    )
    
    heading3_style = ParagraphStyle(
        'Heading3',
        parent=styles['Heading3'],
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.darkslateblue
    )
    
    normal_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,  
        spaceBefore=4,
        spaceAfter=4
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=9,
        leading=12,
        leftIndent=36,
        borderWidth=0.5,
        borderColor=colors.grey,
        borderPadding=6,
        backColor=colors.whitesmoke
    )
    
    quote_style = ParagraphStyle(
        'Quote',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        leftIndent=36,
        rightIndent=36,
        spaceBefore=6,
        spaceAfter=6,
        borderWidth=0,
        borderColor=colors.grey,
        borderPadding=0,
        textColor=colors.darkgray
    )
    
    caption_style = ParagraphStyle(
        'Caption', 
        parent=normal_style,
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        italic=True,
        spaceBefore=2,
        spaceAfter=10,
        textColor=colors.darkgray
    )
    
    image_pattern = r'!\[(.*?)\]\((data:image\/(\w+);base64,([^)]+))\)'
    
    # match code blocks
    code_block_pattern = r'```(.*?)\n(.*?)```'
    
    # match blockquotes
    blockquote_pattern = r'(^|\n)> (.*?)(?=\n[^>]|\n\n|$)'
    
    # match tables
    table_pattern = r'((\|[^\n]*\|)(\n\|[\-:| ]*\|)(\n\|[^\n]*\|)+)'
    
    # match horizontal rules
    hr_pattern = r'(^|\n)[\*\-_]{3,}(?=\n|$)'
    
    # match lists
    list_pattern = r'(^|\n)([*\-+] .*?(?=\n[^*\-+]|\n\n|$))'
    
    heading_pattern = r'(^|\n)(#+)\s+(.*?)(?=\n|$)'
    
    # process title
    title_match = re.search(r'^# (.*?)(?=\n|$)', markdown_text)
    if title_match:
        title_text = title_match.group(1)
        elements.append(Paragraph(title_text, title_style))
        markdown_text = markdown_text[title_match.end():]
    
    # process markdown
    last_pos = 0
    for heading_match in re.finditer(heading_pattern, markdown_text):
        heading_start = heading_match.start()
        
        # process text before heading
        if heading_start > last_pos:
            text_chunk = markdown_text[last_pos:heading_start]
            
            # process images, code blocks, quotes, tables, and lines
            process_chunk(text_chunk, elements, normal_style, code_style, quote_style, caption_style, 
                         image_pattern, code_block_pattern, blockquote_pattern, 
                         table_pattern, hr_pattern, list_pattern)
        
        # process heading
        heading_level = len(heading_match.group(2))  # Number of # characters
        heading_text = heading_match.group(3)
        
        if heading_level == 1:
            if last_pos > 0:  
                elements.append(PageBreak())
            elements.append(Paragraph(heading_text, heading1_style))
        elif heading_level == 2:
            elements.append(Paragraph(heading_text, heading2_style))
        elif heading_level == 3:
            elements.append(Paragraph(heading_text, heading3_style))
        else:
            # for smaller headings, add diff heading style
            custom_heading_style = ParagraphStyle(
                f'Heading{heading_level}',
                parent=normal_style,
                fontSize=11 - (heading_level - 3),  
                leading=14,
                spaceBefore=6,
                spaceAfter=2,
                textColor=colors.darkgray,
                bulletFontName='Helvetica-Bold'
            )
            elements.append(Paragraph(f"<b>{heading_text}</b>", custom_heading_style))
        
        last_pos = heading_match.end()
    
    # process text after last heading
    if last_pos < len(markdown_text):
        remaining_text = markdown_text[last_pos:]
        process_chunk(remaining_text, elements, normal_style, code_style, quote_style, caption_style,
                     image_pattern, code_block_pattern, blockquote_pattern, 
                     table_pattern, hr_pattern, list_pattern)
    
    doc.build(elements)
    
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def process_chunk(text_chunk, elements, normal_style, code_style, quote_style, caption_style,
                 image_pattern, code_block_pattern, blockquote_pattern, 
                 table_pattern, hr_pattern, list_pattern):
    """
    Process a chunk of markdown text, handling images, code blocks, quotes, etc.
    Add appropriate elements to the elements list.
    """
    
    special_elements = []
    
    # find images
    for match in re.finditer(image_pattern, text_chunk):
        special_elements.append(('image', match.span(), match))
    
    # find code blocks
    for match in re.finditer(code_block_pattern, text_chunk, re.DOTALL):
        special_elements.append(('code', match.span(), match))
    
    # find blockquotes
    for match in re.finditer(blockquote_pattern, text_chunk, re.DOTALL):
        special_elements.append(('quote', match.span(), match))
    
    # find tables
    for match in re.finditer(table_pattern, text_chunk, re.DOTALL):
        special_elements.append(('table', match.span(), match))
    
    # find lines
    for match in re.finditer(hr_pattern, text_chunk):
        special_elements.append(('hr', match.span(), match))
    
    # sort special elements by their position in text
    special_elements.sort(key=lambda x: x[1][0])
    
    # process special elements
    last_pos = 0
    for elem_type, (start, end), match in special_elements:
        if start > last_pos:
            process_text(text_chunk[last_pos:start], elements, normal_style)
        if elem_type == 'image':
            process_image(match, elements, caption_style)
        elif elem_type == 'code':
            lang = match.group(1).strip()
            code = match.group(2)
            elements.append(Paragraph(f"<code>{code}</code>", code_style))
            elements.append(Spacer(1, 0.1 * inch))
        elif elem_type == 'quote':
            quote_text = match.group(2)
            # handlinf multi-line quotes
            if '\n' in quote_text:
                for line in quote_text.split('\n'):
                    if line.startswith('> '):
                        line = line[2:]
                    elements.append(Paragraph(line, quote_style))
            else:
                elements.append(Paragraph(quote_text, quote_style))
            elements.append(Spacer(1, 0.1 * inch))
        elif elem_type == 'table':
            process_table(match.group(0), elements)
        elif elem_type == 'hr':
            # add line
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(HorizontalRule())
            elements.append(Spacer(1, 0.1 * inch))
        
        last_pos = end
    
    # process remaining text
    if last_pos < len(text_chunk):
        process_text(text_chunk[last_pos:], elements, normal_style)

def process_text(text, elements, normal_style):
    """Process regular text, handling inline markdown formatting."""
    
    if not text.strip():
        return
    
    # for inline formatting
    html = markdown.markdown(text)
    
    html = re.sub(r'<code>(.*?)</code>', r'<font face="Courier" size="9">\1</font>', html)
    
    # basic list handling
    if re.search(r'<[uo]l>', html):
        list_items = re.findall(r'<li>(.*?)</li>', html, re.DOTALL)
        for item in list_items:
            elements.append(Paragraph(f"• {item}", normal_style))
        return
    
    # split paragraphs
    paragraphs = re.split(r'</?p>', html)
    for para in paragraphs:
        if para.strip():
            elements.append(Paragraph(para.strip(), normal_style))
    
    if len(paragraphs) > 1:
        elements.append(Spacer(1, 0.1 * inch))

def process_image(match, elements, caption_style):
    """Process a markdown image and add it to elements."""
    import base64
    from io import BytesIO
    from PIL import Image
    from reportlab.platypus import Image as ReportLabImage, Paragraph, Spacer
    from reportlab.lib.units import inch
    
    try:
        alt_text = match.group(1)
        img_format = match.group(3)
        base64_str = match.group(4)
        
        # decode image
        img_data = base64.b64decode(base64_str)
        img_temp = BytesIO(img_data)
        img = Image.open(img_temp)
        
        width, height = img.size
        
        # scale image to fit page width
        max_width = 6 * inch 
        if width > max_width:
            scale_factor = max_width / width
            width = max_width
            height = height * scale_factor
        
        img_obj = ReportLabImage(img_temp, width=width, height=height)
        elements.append(img_obj)
        
        # add caption
        if alt_text:
            elements.append(Paragraph(f"{alt_text}", caption_style))
        
        elements.append(Spacer(1, 0.2 * inch))
        
    except Exception as e:
        elements.append(Paragraph(f"[Image Error: {str(e)}]", caption_style))

def process_table(table_markdown, elements):
    """Process a markdown table and add it to elements."""
    import re
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    
    lines = table_markdown.strip().split('\n')
    rows = []
    
    for i, line in enumerate(lines):
        # skip separator row
        if i == 1 and re.match(r'\|[\-:| ]+\|', line):
            continue
        
        # process data rows
        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first/last cells
        rows.append(cells)
    
    if rows:
        table = Table(rows)
        
        # table styling
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ])
        table.setStyle(style)
        
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

# Line
class HorizontalRule:
    """A simple horizontal rule flowable."""
    def __init__(self, width='100%', thickness=1, color=colors.grey, spacer=0.1):
        self.width = width
        self.thickness = thickness
        self.color = color
        self.spacer = spacer
        
    def wrap(self, availWidth, availHeight):
        if isinstance(self.width, str) and self.width.endswith('%'):
            self.width = availWidth * float(self.width[:-1]) / 100
        else:
            self.width = min(self.width, availWidth)
        return self.width, self.thickness + 2*self.spacer*inch
        
    def draw(self):
        from reportlab.lib.units import inch
        
        canv = self.canv
        canv.saveState()
        canv.setLineWidth(self.thickness)
        canv.setStrokeColor(self.color)
        y = self.spacer*inch + self.thickness/2
        canv.line(0, y, self.width, y)
        canv.restoreState()