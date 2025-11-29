"""
Weily AI Backend - SuiteQL with Corrected OAuth 1.0
Fixed authentication for NetSuite SuiteQL REST API
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
from urllib.parse import quote, urlparse, parse_qs
import requests
import json
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weily AI Backend", version="3.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('ALLOWED_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize OpenAI
try:
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logger.info("✅ OpenAI initialized")
except Exception as e:
    logger.error(f"❌ OpenAI error: {e}")
    openai_client = None

# NetSuite Config - matching your mobile app credentials
NETSUITE_CONFIG = {
    'account_id': os.getenv('NETSUITE_ACCOUNT_ID', '8912186').strip(),
    'consumer_key': os.getenv('NETSUITE_CONSUMER_KEY', '').strip(),
    'consumer_secret': os.getenv('NETSUITE_CONSUMER_SECRET', '').strip(),
    'token_id': os.getenv('NETSUITE_TOKEN_ID', '').strip(),
    'token_secret': os.getenv('NETSUITE_TOKEN_SECRET', '').strip(),
}

def validate_config():
    """Validate configuration"""
    required = ['consumer_key', 'consumer_secret', 'token_id', 'token_secret']
    missing = [k for k in required if not NETSUITE_CONFIG[k]]
    if missing:
        logger.error(f"❌ Missing config: {', '.join(missing)}")
        return False
    
    logger.info(f"✅ Account ID: {NETSUITE_CONFIG['account_id']}")
    logger.info(f"✅ Consumer Key: {NETSUITE_CONFIG['consumer_key'][:16]}...")
    logger.info(f"✅ Token ID: {NETSUITE_CONFIG['token_id'][:16]}...")
    return True

CONFIG_VALID = validate_config()

if CONFIG_VALID:
    NETSUITE_CONFIG['base_url'] = f"https://{NETSUITE_CONFIG['account_id']}.suitetalk.api.netsuite.com"
    logger.info(f"✅ Base URL: {NETSUITE_CONFIG['base_url']}")

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    user_context: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    success: bool
    query_type: str
    data: Optional[Dict] = None
    response_text: str
    error: Optional[str] = None
    processing_time: Optional[float] = None

def url_encode(s):
    """URL encode for OAuth (RFC 3986)"""
    return quote(str(s), safe='')

def generate_nonce():
    """Generate 32-char alphanumeric nonce"""
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(chars) for _ in range(32))

def generate_timestamp():
    """Unix timestamp"""
    return str(int(time.time()))

def create_oauth_signature_base_string(method, url, oauth_params):
    """
    Create OAuth signature base string
    CRITICAL: For SuiteQL, URL must NOT include query parameters in base string
    """
    # Parse URL - remove query parameters for base string
    parsed = urlparse(url)
    # Base URL without query string
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    # For SuiteQL POST requests, we only use OAuth params, not URL query params
    # This is different from RESTlet which includes them
    
    # Sort OAuth parameters
    sorted_params = sorted(oauth_params.items())
    
    # Build parameter string
    param_string = '&'.join([
        f'{url_encode(k)}={url_encode(v)}'
        for k, v in sorted_params
    ])
    
    # Build signature base: METHOD&URL&PARAMS
    base_string = (
        f'{method.upper()}&'
        f'{url_encode(base_url)}&'
        f'{url_encode(param_string)}'
    )
    
    return base_string

def generate_signature(base_string, signing_key):
    """Generate HMAC-SHA256 signature"""
    signature = hmac.new(
        signing_key.encode('utf-8'),
        base_string.encode('utf-8'),
        hashlib.sha256
    ).digest()
    return base64.b64encode(signature).decode('utf-8')

def create_oauth_header(method, url):
    """
    Create OAuth 1.0 header for NetSuite SuiteQL
    """
    try:
        nonce = generate_nonce()
        timestamp = generate_timestamp()
        
        # Realm: account ID as-is (NetSuite accepts both formats)
        realm = NETSUITE_CONFIG['account_id']
        
        # OAuth parameters
        oauth_params = {
            'oauth_consumer_key': NETSUITE_CONFIG['consumer_key'],
            'oauth_token': NETSUITE_CONFIG['token_id'],
            'oauth_nonce': nonce,
            'oauth_timestamp': timestamp,
            'oauth_signature_method': 'HMAC-SHA256',
            'oauth_version': '1.0'
        }
        
        # Create signing key
        signing_key = (
            f"{url_encode(NETSUITE_CONFIG['consumer_secret'])}&"
            f"{url_encode(NETSUITE_CONFIG['token_secret'])}"
        )
        
        # Create signature base string
        signature_base = create_oauth_signature_base_string(method, url, oauth_params)
        
        logger.debug(f"Signature base: {signature_base[:150]}...")
        
        # Generate signature
        signature = generate_signature(signature_base, signing_key)
        
        # Build Authorization header
        auth_header = (
            f'OAuth realm="{realm}", '
            f'oauth_consumer_key="{oauth_params["oauth_consumer_key"]}", '
            f'oauth_token="{oauth_params["oauth_token"]}", '
            f'oauth_nonce="{oauth_params["oauth_nonce"]}", '
            f'oauth_timestamp="{oauth_params["oauth_timestamp"]}", '
            f'oauth_signature_method="{oauth_params["oauth_signature_method"]}", '
            f'oauth_version="{oauth_params["oauth_version"]}", '
            f'oauth_signature="{url_encode(signature)}"'
        )
        
        return auth_header
        
    except Exception as e:
        logger.error(f"❌ OAuth generation failed: {e}", exc_info=True)
        raise

def execute_suiteql(sql_query: str, limit: int = 1000) -> Optional[Dict]:
    """
    Execute SuiteQL query via REST API
    """
    if not CONFIG_VALID:
        logger.error("❌ Config invalid")
        return None
    
    # SuiteQL endpoint with limit parameter
    url = f"{NETSUITE_CONFIG['base_url']}/services/rest/query/v1/suiteql?limit={limit}"
    method = 'POST'
    
    try:
        # Generate OAuth header
        auth_header = create_oauth_header(method, url)
        
        headers = {
            'Content-Type': 'application/json',
            'prefer': 'transient',
            'Authorization': auth_header
        }
        
        # Clean query (remove any LIMIT clauses)
        clean_query = sql_query.replace('LIMIT 1000', '').replace('LIMIT 100', '').strip()
        
        payload = {'q': clean_query}
        
        logger.info(f"🔍 Executing SuiteQL: {clean_query[:100]}...")
        logger.debug(f"URL: {url}")
        logger.debug(f"Auth header: {auth_header[:80]}...")
        
        # Execute request
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        logger.info(f"Response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            count = len(result.get('items', []))
            logger.info(f"✅ Success: {count} rows")
            return result
            
        elif response.status_code == 401:
            logger.error(f"❌ Authentication Failed")
            try:
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                logger.error(f"Response text: {response.text}")
            
            logger.error("\n🔧 TROUBLESHOOTING:")
            logger.error("1. Verify Integration Record is ENABLED in NetSuite")
            logger.error("2. Check Access Token is ACTIVE (not revoked)")
            logger.error("3. Ensure token role has 'Web Services' permission")
            logger.error("4. Check 'SuiteAnalytics Workbook' or 'SuiteQL' permissions")
            logger.error("5. Verify credentials match exactly (including spaces)")
            
            return None
            
        else:
            logger.error(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("❌ Request timeout")
        return None
    except Exception as e:
        logger.error(f"❌ Exception: {e}", exc_info=True)
        return None

# AI Query Analysis
def analyze_query(user_query: str) -> Dict:
    """Analyze user query"""
    if not openai_client:
        return {"query_type": "general", "params": {}}
    
    system_prompt = """Analyze NetSuite queries and return JSON.

Query types:
- sales_orders_today: Sales orders created today
- sales_orders_period: Sales for date range
- invoices_today: Invoices today
- invoices_period: Invoices for period
- top_customers: Best customers
- subsidiaries_list: List subsidiaries
- general: Other

Date handling:
- "today" → CURRENT_DATE
- "this month" → current month
- "this week" → current week

Return JSON:
{
    "query_type": "sales_orders_today",
    "params": {
        "date_filter": "today",
        "limit": 100
    }
}"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"❌ AI error: {e}")
        return {"query_type": "general", "params": {}}

def get_sql_query(query_type: str, params: Dict) -> Optional[str]:
    """Generate SuiteQL based on query type"""
    
    date_filter = params.get('date_filter', 'today')
    
    # Date conditions
    if date_filter == 'today':
        date_condition = "t.trandate = CURRENT_DATE"
    elif date_filter == 'this_week':
        date_condition = "t.trandate >= TRUNC(CURRENT_DATE, 'IW')"
    elif date_filter == 'this_month':
        date_condition = "t.trandate >= TRUNC(CURRENT_DATE, 'MM')"
    else:
        date_condition = "t.trandate = CURRENT_DATE"
    
    if query_type in ['sales_orders_today', 'sales_orders_period']:
        return f"""
            SELECT 
                t.id,
                t.tranid as document_number,
                t.trandate as date,
                c.companyname as customer,
                s.name as subsidiary,
                t.currency,
                t.foreigntotal as amount,
                t.status
            FROM transaction t
            LEFT JOIN customer c ON t.entity = c.id
            LEFT JOIN subsidiary s ON t.subsidiary = s.id
            WHERE t.type = 'SalesOrd'
            AND {date_condition}
            ORDER BY t.trandate DESC
        """
    
    elif query_type in ['invoices_today', 'invoices_period']:
        return f"""
            SELECT 
                t.id,
                t.tranid as document_number,
                t.trandate as date,
                c.companyname as customer,
                t.currency,
                t.foreigntotal as amount,
                t.status
            FROM transaction t
            LEFT JOIN customer c ON t.entity = c.id
            WHERE t.type = 'CustInvc'
            AND {date_condition}
            ORDER BY t.trandate DESC
        """
    
    elif query_type == 'subsidiaries_list':
        return """
            SELECT id, name, currency
            FROM subsidiary
            WHERE isinactive = 'F'
            ORDER BY name
        """
    
    elif query_type == 'top_customers':
        return f"""
            SELECT 
                c.companyname as customer,
                COUNT(t.id) as orders,
                SUM(t.foreigntotal) as total_sales,
                t.currency
            FROM transaction t
            INNER JOIN customer c ON t.entity = c.id
            WHERE t.type = 'SalesOrd'
            AND t.trandate >= CURRENT_DATE - 30
            GROUP BY c.companyname, t.currency
            ORDER BY total_sales DESC
        """
    
    return None

def generate_response(user_query: str, data: Optional[Dict], query_type: str) -> str:
    """Generate AI response"""
    if not openai_client:
        return "Results retrieved."
    
    if data and 'items' in data:
        items = data['items']
        context = f"Found {len(items)} results:\n{json.dumps(items[:5], indent=2)}"
    else:
        context = "No data found."
    
    system_prompt = """You are Weily, a NetSuite AI assistant.
    
Be conversational and helpful. Format amounts with currency symbols.
Be specific about counts and dates."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_query}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ Response error: {e}")
        return "I found the data but couldn't generate a response."

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Weily AI Backend",
        "version": "3.1.0",
        "method": "SuiteQL REST API + OAuth 1.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "config_valid": CONFIG_VALID,
        "openai_ready": openai_client is not None,
        "base_url": NETSUITE_CONFIG.get('base_url', 'not configured')
    }

@app.get("/test-auth")
async def test_auth():
    """Test SuiteQL authentication"""
    if not CONFIG_VALID:
        raise HTTPException(status_code=500, detail="Config invalid")
    
    # Simple test query
    sql = "SELECT id, companyname FROM customer WHERE ROWNUM <= 1"
    result = execute_suiteql(sql)
    
    if result:
        return {
            "success": True,
            "message": "SuiteQL authentication successful!",
            "sample_data": result.get('items', [])
        }
    else:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Check server logs."
        )

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process user query via SuiteQL"""
    start_time = time.time()
    
    try:
        if not CONFIG_VALID:
            raise HTTPException(status_code=500, detail="Service not configured")
        
        # Analyze query
        analysis = analyze_query(request.query)
        query_type = analysis.get('query_type', 'general')
        params = analysis.get('params', {})
        
        logger.info(f"📊 Analysis: {query_type}")
        
        if query_type == 'general':
            return QueryResponse(
                success=True,
                query_type="general",
                response_text="I'm not sure about that. Try asking about sales orders, invoices, or customers.",
                processing_time=time.time() - start_time
            )
        
        # Execute SuiteQL
        sql = get_sql_query(query_type, params)
        data = None
        if sql:
            data = execute_suiteql(sql)
        
        # Generate response
        response_text = generate_response(request.query, data, query_type)
        
        return QueryResponse(
            success=True,
            query_type=query_type,
            data=data,
            response_text=response_text,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)