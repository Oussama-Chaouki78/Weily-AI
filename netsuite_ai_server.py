"""
Weily AI Backend Server - Smart Version
Handles multi-currency, subsidiaries, and intelligent queries
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from openai import OpenAI
import hmac
import hashlib
import time
import random
import base64
from urllib.parse import quote
import requests
import json
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weily AI Backend",
    description="Smart AI-powered NetSuite assistant",
    version="2.0.0"
)

# CORS
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize OpenAI
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logger.info("✅ OpenAI client initialized")
except Exception as e:
    logger.error(f"❌ OpenAI initialization failed: {str(e)}")
    openai_client = None

# NetSuite Config
NETSUITE_CONFIG = {
    'account_id': os.getenv('NETSUITE_ACCOUNT_ID'),
    'consumer_key': os.getenv('NETSUITE_CONSUMER_KEY'),
    'consumer_secret': os.getenv('NETSUITE_CONSUMER_SECRET'),
    'token_id': os.getenv('NETSUITE_TOKEN_ID'),
    'token_secret': os.getenv('NETSUITE_TOKEN_SECRET'),
}

def validate_config():
    missing = [k for k, v in NETSUITE_CONFIG.items() if not v]
    if missing:
        logger.warning(f"⚠️ Missing config: {', '.join(missing)}")
        return False
    return True

NETSUITE_CONFIG['base_url'] = f"https://{NETSUITE_CONFIG['account_id']}.suitetalk.api.netsuite.com"
CONFIG_VALID = validate_config()

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    user_context: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    success: bool
    query_type: str
    data: Optional[Dict] = None
    response_text: str
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

# OAuth and SuiteQL - Same as before
def generate_oauth_header(url: str, method: str = 'GET') -> str:
    """Generate OAuth 1.0 header"""
    try:
        timestamp = str(int(time.time()))
        nonce = ''.join(random.choices('0123456789abcdef', k=32))
        
        oauth_params = {
            'oauth_consumer_key': NETSUITE_CONFIG['consumer_key'],
            'oauth_token': NETSUITE_CONFIG['token_id'],
            'oauth_signature_method': 'HMAC-SHA256',
            'oauth_timestamp': timestamp,
            'oauth_nonce': nonce,
            'oauth_version': '1.0',
            'realm': NETSUITE_CONFIG['account_id'].upper()
        }
        
        signature_params = {k: v for k, v in oauth_params.items() if k != 'realm'}
        params_string = '&'.join([f'{k}={quote(str(v), safe="")}' 
                                 for k, v in sorted(signature_params.items())])
        base_string = f'{method}&{quote(url, safe="")}&{quote(params_string, safe="")}'
        signing_key = (f'{quote(NETSUITE_CONFIG["consumer_secret"], safe="")}&'
                      f'{quote(NETSUITE_CONFIG["token_secret"], safe="")}')
        
        signature = base64.b64encode(
            hmac.new(signing_key.encode('utf-8'), 
                    base_string.encode('utf-8'), 
                    hashlib.sha256).digest()
        ).decode('utf-8')
        
        oauth_params['oauth_signature'] = signature
        auth_header = 'OAuth ' + ','.join([f'{k}="{v}"' 
                                           for k, v in sorted(oauth_params.items())])
        return auth_header
    except Exception as e:
        logger.error(f"❌ OAuth generation failed: {str(e)}")
        raise

def execute_suiteql(query: str, limit: int = 1000) -> Optional[Dict]:
    """Execute SuiteQL query"""
    if not CONFIG_VALID:
        logger.error("❌ NetSuite config invalid")
        return None
    
    url = f"{NETSUITE_CONFIG['base_url']}/services/rest/query/v1/suiteql"
    headers = {
        'Content-Type': 'application/json',
        'prefer': 'transient',
        'Authorization': generate_oauth_header(url, 'POST')
    }
    
    if 'LIMIT' not in query.upper():
        query = f"{query} LIMIT {limit}"
    
    try:
        logger.info(f"🔍 Executing SuiteQL: {query[:150]}...")
        response = requests.post(url, headers=headers, json={'q': query}, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ SuiteQL success: {len(result.get('items', []))} rows")
            return result
        else:
            logger.error(f"❌ SuiteQL Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"❌ SuiteQL Exception: {str(e)}")
        return None

# Smart AI Query Analysis
def analyze_query_smart(user_query: str) -> Dict:
    """Smart analysis with context awareness"""
    if not openai_client:
        return {"query_type": "general", "params": {}, "needs_info": False}
    
    system_prompt = """You are Weily AI for NetSuite. Analyze queries intelligently and return JSON.

CRITICAL UNDERSTANDING:
- "sales" or "sales order" = SalesOrd (type in NetSuite)
- "invoice" = CustInvc
- "today" = CURRENT_DATE only
- "this month" = current month only
- "yesterday" = CURRENT_DATE - 1
- User has MULTIPLE subsidiaries and currencies

Query types:
- sales_orders_today: Sales orders created today (use trandate = CURRENT_DATE)
- sales_orders_period: Sales orders for a time period
- invoices_today: Invoices created today
- invoices_period: Invoices for a period
- top_customers: Best customers by sales
- subsidiaries_list: List all subsidiaries
- currency_summary: Sales by currency
- general: Need more info

SMART EXTRACTION:
{
    "query_type": "sales_orders_today",
    "params": {
        "date_filter": "today|this_month|this_week|custom",
        "start_date": "2024-11-28",  // if specific date
        "end_date": "2024-11-28",
        "record_type": "SalesOrd|CustInvc|Estimate",
        "needs_subsidiary": true,  // if multiple subsidiaries exist
        "needs_currency": false,
        "limit": 100
    },
    "needs_clarification": false,
    "clarification": null
}

IMPORTANT DATE LOGIC:
- "today" → trandate = CURRENT_DATE
- "this month" → trandate >= first day of current month
- "this week" → trandate >= start of this week
- "yesterday" → trandate = CURRENT_DATE - 1
- NO date specified for historical = ERROR, ask for timeframe

Return JSON only."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Use smarter model for analysis
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        logger.info(f"🤖 Smart Analysis: {result}")
        return result
    except Exception as e:
        logger.error(f"❌ AI Analysis Error: {str(e)}")
        return {"query_type": "general", "params": {}}

# Get subsidiaries and currencies first
def get_subsidiaries() -> List[Dict]:
    """Get list of subsidiaries"""
    sql = """
        SELECT 
            id,
            name,
            currency
        FROM subsidiary
        WHERE isinactive = 'F'
        ORDER BY name
    """
    result = execute_suiteql(sql)
    return result.get('items', []) if result else []

def get_currencies() -> List[Dict]:
    """Get list of active currencies"""
    sql = """
        SELECT DISTINCT
            currency as currency_code,
            COUNT(*) as usage_count
        FROM subsidiary
        WHERE isinactive = 'F'
        GROUP BY currency
        ORDER BY usage_count DESC
    """
    result = execute_suiteql(sql)
    return result.get('items', []) if result else []

# Smart SQL Generator
def get_smart_sql(query_type: str, params: Dict) -> Optional[str]:
    """Generate intelligent SQL based on context"""
    
    record_type = params.get('record_type', 'SalesOrd')
    date_filter = params.get('date_filter', 'today')
    limit = params.get('limit', 100)
    
    # Build date condition
    if date_filter == 'today':
        date_condition = "t.trandate = CURRENT_DATE"
    elif date_filter == 'yesterday':
        date_condition = "t.trandate = CURRENT_DATE - 1"
    elif date_filter == 'this_week':
        date_condition = "t.trandate >= TRUNC(CURRENT_DATE, 'IW')"
    elif date_filter == 'this_month':
        date_condition = "t.trandate >= TRUNC(CURRENT_DATE, 'MM')"
    elif date_filter == 'custom' and params.get('start_date'):
        start = params.get('start_date')
        end = params.get('end_date', start)
        date_condition = f"t.trandate BETWEEN TO_DATE('{start}', 'YYYY-MM-DD') AND TO_DATE('{end}', 'YYYY-MM-DD')"
    else:
        date_condition = "t.trandate = CURRENT_DATE"
    
    # Main queries
    if query_type in ['sales_orders_today', 'sales_orders_period']:
        return f"""
            SELECT 
                t.id,
                t.tranid as document_number,
                t.trandate as date,
                t.type as transaction_type,
                c.companyname as customer_name,
                s.name as subsidiary,
                t.currency,
                t.foreigntotal as total_amount,
                t.status,
                t.memo
            FROM transaction t
            LEFT JOIN customer c ON t.entity = c.id
            LEFT JOIN subsidiary s ON t.subsidiary = s.id
            WHERE t.type = 'SalesOrd'
            AND {date_condition}
            AND t.status NOT IN ('Cancelled', 'Closed')
            ORDER BY t.trandate DESC, t.id DESC
        """
    
    elif query_type in ['invoices_today', 'invoices_period']:
        return f"""
            SELECT 
                t.id,
                t.tranid as document_number,
                t.trandate as date,
                c.companyname as customer_name,
                s.name as subsidiary,
                t.currency,
                t.foreigntotal as total_amount,
                t.status,
                t.duedate
            FROM transaction t
            LEFT JOIN customer c ON t.entity = c.id
            LEFT JOIN subsidiary s ON t.subsidiary = s.id
            WHERE t.type = 'CustInvc'
            AND {date_condition}
            AND t.status NOT IN ('Voided', 'Cancelled')
            ORDER BY t.trandate DESC
        """
    
    elif query_type == 'subsidiaries_list':
        return """
            SELECT 
                id,
                name,
                legalname,
                currency,
                country
            FROM subsidiary
            WHERE isinactive = 'F'
            ORDER BY name
        """
    
    elif query_type == 'currency_summary':
        return f"""
            SELECT 
                t.currency,
                s.name as subsidiary,
                COUNT(t.id) as transaction_count,
                SUM(t.foreigntotal) as total_amount
            FROM transaction t
            LEFT JOIN subsidiary s ON t.subsidiary = s.id
            WHERE t.type IN ('SalesOrd', 'CustInvc')
            AND {date_condition}
            AND t.status NOT IN ('Voided', 'Cancelled', 'Closed')
            GROUP BY t.currency, s.name
            ORDER BY total_amount DESC
        """
    
    elif query_type == 'top_customers':
        period_days = params.get('days', 30)
        return f"""
            SELECT 
                c.companyname as customer_name,
                COUNT(DISTINCT t.id) as order_count,
                SUM(t.foreigntotal) as total_sales,
                MAX(t.trandate) as last_order_date,
                t.currency
            FROM transaction t
            INNER JOIN customer c ON t.entity = c.id
            WHERE t.type IN ('SalesOrd', 'CustInvc')
            AND t.trandate >= CURRENT_DATE - {period_days}
            AND t.status NOT IN ('Voided', 'Cancelled', 'Closed')
            GROUP BY c.companyname, t.currency
            ORDER BY total_sales DESC
            LIMIT {limit}
        """
    
    return None

# Smart Response Generator
def generate_smart_response(user_query: str, data: Optional[Dict], query_type: str, params: Dict) -> str:
    """Generate intelligent, context-aware response"""
    if not openai_client:
        return "AI service unavailable."
    
    # Prepare context
    if data and 'items' in data:
        items = data['items']
        context = f"Found {len(items)} results:\n{json.dumps(items[:10], indent=2)}"
    else:
        context = "No data found for this query."
    
    system_prompt = """You are Weily, a smart NetSuite AI assistant.

CRITICAL RULES:
1. If user asks about "sales" without timeframe, you MUST see trandate in results
2. ONLY report numbers that match the user's timeframe
3. If multiple subsidiaries/currencies exist, mention them clearly
4. Format amounts with currency symbols (USD $, EUR €, etc)
5. Be conversational but accurate
6. If results seem wrong, explain what you found

Response format:
- Start with direct answer
- Break down by subsidiary if multiple exist
- Break down by currency if multiple exist
- Be specific about timeframes
- Suggest follow-up questions

Examples:
❌ BAD: "You made $8M in sales" (when user asked for TODAY but showing ALL TIME)
✅ GOOD: "Today you have 2 sales orders totaling $15,340 USD. Would you like to see all subsidiaries or a specific one?"

✅ GOOD: "This week across your 3 subsidiaries: Weily US $45K USD, Weily EU €32K EUR, Weily UK £28K GBP. Total of 12 orders."

Always be honest about what timeframe the data covers."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_query}\nParams: {json.dumps(params)}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ Response Error: {str(e)}")
        return "I found the data but had trouble explaining it. Please check the raw results."

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Smart Weily AI Backend",
        "version": "2.0.0",
        "features": ["multi-currency", "multi-subsidiary", "smart-queries"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Smart Weily AI",
        "config_valid": CONFIG_VALID,
        "openai_available": openai_client is not None
    }

@app.get("/subsidiaries")
async def list_subsidiaries():
    """List all subsidiaries"""
    subs = get_subsidiaries()
    return {"subsidiaries": subs, "count": len(subs)}

@app.get("/currencies")
async def list_currencies():
    """List all currencies"""
    curr = get_currencies()
    return {"currencies": curr, "count": len(curr)}

@app.post("/query")
async def process_query_smart(request: QueryRequest, req: Request):
    """Smart query processing with multi-subsidiary/currency awareness"""
    
    start_time = time.time()
    client_ip = req.client.host
    logger.info(f"📥 Smart Query from {client_ip}: {request.query}")
    
    try:
        if not CONFIG_VALID or not openai_client:
            raise HTTPException(status_code=500, detail="Service unavailable")
        
        # Smart analysis
        analysis = analyze_query_smart(request.query)
        query_type = analysis.get('query_type', 'general')
        params = analysis.get('params', {})
        needs_clarification = analysis.get('needs_clarification', False)
        
        # Check if we need to ask about subsidiary/currency
        if params.get('needs_subsidiary') and not params.get('subsidiary_id'):
            subs = get_subsidiaries()
            if len(subs) > 1:
                sub_list = ", ".join([s['name'] for s in subs[:5]])
                return QueryResponse(
                    success=True,
                    query_type=query_type,
                    response_text=f"I found {len(subs)} subsidiaries: {sub_list}. Which one would you like, or should I show all?",
                    needs_clarification=True,
                    clarification_question="subsidiary"
                )
        
        # Execute query
        data = None
        if query_type != 'general':
            sql = get_smart_sql(query_type, params)
            if sql:
                data = execute_suiteql(sql)
        
        # Generate smart response
        response_text = generate_smart_response(request.query, data, query_type, params)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Processed in {processing_time:.2f}s")
        
        return QueryResponse(
            success=True,
            query_type=query_type,
            data=data,
            response_text=response_text,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")