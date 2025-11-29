"""
Weily AI Backend Server - Fixed OAuth Version
Critical fixes for NetSuite authentication
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
from datetime import datetime
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weily AI Backend", version="2.1.0")

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

# NetSuite Config
NETSUITE_CONFIG = {
    'account_id': os.getenv('NETSUITE_ACCOUNT_ID', '').strip(),
    'consumer_key': os.getenv('NETSUITE_CONSUMER_KEY', '').strip(),
    'consumer_secret': os.getenv('NETSUITE_CONSUMER_SECRET', '').strip(),
    'token_id': os.getenv('NETSUITE_TOKEN_ID', '').strip(),
    'token_secret': os.getenv('NETSUITE_TOKEN_SECRET', '').strip(),
}

def validate_config():
    """Validate NetSuite configuration"""
    missing = [k for k, v in NETSUITE_CONFIG.items() if not v]
    if missing:
        logger.error(f"❌ Missing config: {', '.join(missing)}")
        return False
    
    # Validate account ID format
    account_id = NETSUITE_CONFIG['account_id']
    if not account_id:
        logger.error("❌ Account ID is empty")
        return False
    
    # Log config status (safely)
    logger.info(f"✅ Account ID: {account_id}")
    logger.info(f"✅ Consumer Key: {NETSUITE_CONFIG['consumer_key'][:8]}...")
    logger.info(f"✅ Token ID: {NETSUITE_CONFIG['token_id'][:8]}...")
    
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
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

def generate_oauth_header(url: str, method: str = 'POST') -> str:
    """
    Generate OAuth 1.0 header for NetSuite
    
    CRITICAL FIXES:
    1. Realm must be UPPERCASE with underscores (not hyphens)
    2. Signature parameters must be sorted correctly
    3. URL encoding must be precise
    """
    try:
        timestamp = str(int(time.time()))
        nonce = ''.join(random.choices('0123456789abcdef', k=32))
        
        # CRITICAL: Realm formatting for NetSuite
        # Example: "TSTDRV1234567_SB1" not "tstdrv1234567-sb1"
        realm = NETSUITE_CONFIG['account_id'].upper().replace('-', '_')
        
        # OAuth parameters (realm is separate)
        oauth_params = {
            'oauth_consumer_key': NETSUITE_CONFIG['consumer_key'],
            'oauth_token': NETSUITE_CONFIG['token_id'],
            'oauth_signature_method': 'HMAC-SHA256',
            'oauth_timestamp': timestamp,
            'oauth_nonce': nonce,
            'oauth_version': '1.0',
        }
        
        # Build signature base string
        # 1. Sort parameters alphabetically
        sorted_params = sorted(oauth_params.items())
        
        # 2. Build parameter string with proper encoding
        params_string = '&'.join([
            f'{quote(str(k), safe="")}={quote(str(v), safe="")}'
            for k, v in sorted_params
        ])
        
        # 3. Build base string: METHOD&URL&PARAMS
        base_string = (
            f'{method}&'
            f'{quote(url, safe="")}&'
            f'{quote(params_string, safe="")}'
        )
        
        # 4. Build signing key
        signing_key = (
            f'{quote(NETSUITE_CONFIG["consumer_secret"], safe="")}&'
            f'{quote(NETSUITE_CONFIG["token_secret"], safe="")}'
        )
        
        # 5. Generate signature
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode('utf-8'),
                base_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        # 6. Add signature to params
        oauth_params['oauth_signature'] = signature
        
        # 7. Build Authorization header with realm FIRST
        auth_header = f'OAuth realm="{realm}",' + ','.join([
            f'{k}="{v}"' for k, v in sorted(oauth_params.items())
        ])
        
        logger.debug(f"OAuth Header: {auth_header[:80]}...")
        return auth_header
        
    except Exception as e:
        logger.error(f"❌ OAuth generation failed: {str(e)}", exc_info=True)
        raise

def execute_suiteql(query: str, limit: int = 1000) -> Optional[Dict]:
    """Execute SuiteQL query with proper error handling"""
    if not CONFIG_VALID:
        logger.error("❌ NetSuite config invalid")
        return None
    
    # Build URL with limit parameter
    url = f"{NETSUITE_CONFIG['base_url']}/services/rest/query/v1/suiteql?limit={limit}"
    
    try:
        # Generate OAuth header
        auth_header = generate_oauth_header(url, 'POST')
        
        headers = {
            'Content-Type': 'application/json',
            'prefer': 'transient',
            'Authorization': auth_header
        }
        
        # Remove LIMIT from query if present
        clean_query = query.replace('LIMIT 1000', '').replace('LIMIT 100', '').strip()
        
        payload = {'q': clean_query}
        
        logger.info(f"🔍 Executing: {clean_query[:100]}...")
        logger.debug(f"URL: {url}")
        logger.debug(f"Payload: {json.dumps(payload)}")
        
        # Execute request
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            count = len(result.get('items', []))
            logger.info(f"✅ Success: {count} rows returned")
            return result
            
        elif response.status_code == 401:
            logger.error(f"❌ Authentication Failed (401)")
            error_detail = response.json()
            logger.error(f"Error: {json.dumps(error_detail, indent=2)}")
            
            # Provide helpful debugging info
            logger.error("\n🔧 TROUBLESHOOTING:")
            logger.error("1. Check Integration Record is ENABLED")
            logger.error("2. Verify Access Token is not REVOKED")
            logger.error("3. Ensure Role has 'Web Services' permission")
            logger.error("4. Check Token Role in NetSuite (Setup > Access Tokens)")
            logger.error("5. Verify Account ID format (use underscores, not hyphens)")
            
            return None
            
        else:
            logger.error(f"❌ SuiteQL Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("❌ Request timed out (30s)")
        return None
    except Exception as e:
        logger.error(f"❌ SuiteQL Exception: {str(e)}", exc_info=True)
        return None

# Smart AI Query Analysis
def analyze_query_smart(user_query: str) -> Dict:
    """Analyze user query and determine intent"""
    if not openai_client:
        return {"query_type": "general", "params": {}}
    
    system_prompt = """Analyze NetSuite queries and return JSON.

Query types:
- sales_orders_today: Sales orders created today
- sales_orders_period: Sales for a time period
- invoices_today: Invoices today
- invoices_period: Invoices for period
- top_customers: Best customers
- subsidiaries_list: List subsidiaries
- general: Other queries

Date handling:
- "today" → trandate = CURRENT_DATE
- "this month" → trandate >= first of month
- "this week" → trandate >= Monday
- No date = ask for timeframe

Return JSON:
{
    "query_type": "sales_orders_today",
    "params": {
        "date_filter": "today",
        "record_type": "SalesOrd",
        "limit": 100
    },
    "needs_clarification": false
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
        logger.error(f"❌ AI analysis error: {e}")
        return {"query_type": "general", "params": {}}

def get_smart_sql(query_type: str, params: Dict) -> Optional[str]:
    """Generate SQL based on query type"""
    
    date_filter = params.get('date_filter', 'today')
    
    # Build date condition
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
                c.companyname as customer_name,
                s.name as subsidiary,
                t.currency,
                t.foreigntotal as total_amount,
                t.status
            FROM transaction t
            LEFT JOIN customer c ON t.entity = c.id
            LEFT JOIN subsidiary s ON t.subsidiary = s.id
            WHERE t.type = 'SalesOrd'
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
    
    return None

def generate_smart_response(user_query: str, data: Optional[Dict], query_type: str) -> str:
    """Generate natural language response"""
    if not openai_client:
        return "Results retrieved but AI response unavailable."
    
    if data and 'items' in data:
        items = data['items']
        context = f"Found {len(items)} results:\n{json.dumps(items[:5], indent=2)}"
    else:
        context = "No data found."
    
    system_prompt = """You are Weily, a NetSuite AI assistant. 
    
Be conversational and helpful. Format amounts clearly with currency.
If multiple subsidiaries exist, break down by subsidiary."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_query}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=250
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
        "version": "2.1.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "config_valid": CONFIG_VALID,
        "openai_ready": openai_client is not None,
        "netsuite_url": NETSUITE_CONFIG.get('base_url', 'not configured')
    }

@app.get("/test-auth")
async def test_auth():
    """Test NetSuite authentication"""
    if not CONFIG_VALID:
        raise HTTPException(status_code=500, detail="Config invalid")
    
    # Simple test query
    sql = "SELECT id, companyname FROM customer WHERE ROWNUM <= 1"
    result = execute_suiteql(sql)
    
    if result:
        return {
            "success": True,
            "message": "Authentication successful!",
            "sample_data": result.get('items', [])
        }
    else:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Check logs for details."
        )

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process user query"""
    start_time = time.time()
    
    try:
        if not CONFIG_VALID:
            raise HTTPException(status_code=500, detail="Service not configured")
        
        # Analyze query
        analysis = analyze_query_smart(request.query)
        query_type = analysis.get('query_type', 'general')
        params = analysis.get('params', {})
        
        # Execute query
        data = None
        if query_type != 'general':
            sql = get_smart_sql(query_type, params)
            if sql:
                data = execute_suiteql(sql)
        
        # Generate response
        response_text = generate_smart_response(request.query, data, query_type)
        
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