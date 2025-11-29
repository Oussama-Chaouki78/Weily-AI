"""
Weily AI Backend - Using NetSuite Suitelet Bridge
Calls NetSuite Suitelet instead of direct OAuth
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from openai import OpenAI
import requests
import json
import logging
from typing import Optional, Dict, Any, List
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weily AI Backend", version="4.0.0")

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

# NetSuite Suitelet URL
SUITELET_URL = os.getenv('NETSUITE_SUITELET_URL', '').strip()

if SUITELET_URL:
    logger.info(f"✅ Suitelet URL: {SUITELET_URL}")
else:
    logger.warning("⚠️ NETSUITE_SUITELET_URL not configured")

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

def execute_suiteql_via_suitelet(sql_query: str, limit: int = 1000) -> Optional[Dict]:
    """
    Execute SuiteQL by calling NetSuite Suitelet
    """
    if not SUITELET_URL:
        logger.error("❌ Suitelet URL not configured")
        return None
    
    try:
        payload = {
            'sql': sql_query,
            'limit': limit
        }
        
        logger.info(f"🔍 Calling Suitelet: {sql_query[:100]}...")
        logger.debug(f"Payload: {json.dumps(payload)}")
        
        # Call Suitelet
        response = requests.post(
            SUITELET_URL,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            json=payload,
            timeout=30
        )
        
        logger.info(f"Response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                count = len(result.get('items', []))
                logger.info(f"✅ Success: {count} rows")
                return result
            else:
                logger.error(f"❌ Suitelet Error: {result.get('error')}")
                return None
        else:
            logger.error(f"❌ HTTP Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error("❌ Request timeout")
        return None
    except Exception as e:
        logger.error(f"❌ Exception: {e}", exc_info=True)
        return None

def analyze_query(user_query: str) -> Dict:
    """Analyze user query with AI"""
    if not openai_client:
        return {"query_type": "general", "params": {}}
    
    system_prompt = """Analyze NetSuite queries and return JSON.

Query types:
- sales_orders_today: Sales orders created today
- sales_orders_period: Sales for date range
- invoices_today: Invoices today
- invoices_period: Invoices for period
- top_customers: Best customers by sales
- subsidiaries_list: List all subsidiaries
- general: Other queries

Date handling:
- "today" → CURRENT_DATE
- "this month" → TRUNC(CURRENT_DATE, 'MM')
- "this week" → TRUNC(CURRENT_DATE, 'IW')
- "yesterday" → CURRENT_DATE - 1

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
        logger.error(f"❌ AI analysis error: {e}")
        return {"query_type": "general", "params": {}}

def get_sql_query(query_type: str, params: Dict) -> Optional[str]:
    """Generate SuiteQL based on query type"""
    
    date_filter = params.get('date_filter', 'today')
    
    # Build date condition
    if date_filter == 'today':
        date_condition = "t.trandate = CURRENT_DATE"
    elif date_filter == 'yesterday':
        date_condition = "t.trandate = CURRENT_DATE - 1"
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
                s.name as subsidiary,
                t.currency,
                t.foreigntotal as amount,
                t.status
            FROM transaction t
            LEFT JOIN customer c ON t.entity = c.id
            LEFT JOIN subsidiary s ON t.subsidiary = s.id
            WHERE t.type = 'CustInvc'
            AND {date_condition}
            ORDER BY t.trandate DESC
        """
    
    elif query_type == 'subsidiaries_list':
        return """
            SELECT 
                id, 
                name, 
                currency,
                country
            FROM subsidiary
            WHERE isinactive = 'F'
            ORDER BY name
        """
    
    elif query_type == 'top_customers':
        days = params.get('days', 30)
        return f"""
            SELECT 
                c.companyname as customer,
                COUNT(t.id) as order_count,
                SUM(t.foreigntotal) as total_sales,
                t.currency
            FROM transaction t
            INNER JOIN customer c ON t.entity = c.id
            WHERE t.type IN ('SalesOrd', 'CustInvc')
            AND t.trandate >= CURRENT_DATE - {days}
            GROUP BY c.companyname, t.currency
            ORDER BY total_sales DESC
        """
    
    return None

def generate_response(user_query: str, data: Optional[Dict], query_type: str) -> str:
    """Generate natural language response"""
    if not openai_client:
        return "Query executed successfully."
    
    if data and 'items' in data:
        items = data['items']
        context = f"Found {len(items)} results:\n{json.dumps(items[:5], indent=2)}"
    else:
        context = "No data found."
    
    system_prompt = """You are Weily, a friendly NetSuite AI assistant.

Be conversational and helpful. Format amounts with currency symbols.
Be specific about dates, counts, and subsidiaries.
Keep responses concise but informative."""
    
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
        logger.error(f"❌ Response generation error: {e}")
        return "I found the data but had trouble explaining it."

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Weily AI Backend",
        "version": "4.0.0",
        "method": "NetSuite Suitelet Bridge",
        "status": "running",
        "suitelet_configured": bool(SUITELET_URL)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "openai_ready": openai_client is not None,
        "suitelet_url": SUITELET_URL if SUITELET_URL else "not configured"
    }

@app.get("/test-suitelet")
async def test_suitelet():
    """Test Suitelet connection"""
    if not SUITELET_URL:
        raise HTTPException(status_code=500, detail="Suitelet URL not configured")
    
    # Simple test query
    sql = "SELECT id, companyname FROM customer WHERE ROWNUM <= 1"
    result = execute_suiteql_via_suitelet(sql)
    
    if result and result.get('success'):
        return {
            "success": True,
            "message": "Suitelet connection successful!",
            "sample_data": result.get('items', [])
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Suitelet call failed. Check logs."
        )

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process user query"""
    start_time = time.time()
    
    try:
        if not SUITELET_URL:
            raise HTTPException(status_code=500, detail="Suitelet not configured")
        
        # Analyze query
        analysis = analyze_query(request.query)
        query_type = analysis.get('query_type', 'general')
        params = analysis.get('params', {})
        
        logger.info(f"📊 Query type: {query_type}")
        
        if query_type == 'general':
            return QueryResponse(
                success=True,
                query_type="general",
                response_text="I'm not sure about that. Try asking about sales orders, invoices, customers, or subsidiaries.",
                processing_time=time.time() - start_time
            )
        
        # Generate SQL
        sql = get_sql_query(query_type, params)
        if not sql:
            return QueryResponse(
                success=False,
                query_type=query_type,
                response_text="I couldn't generate a query for that request.",
                processing_time=time.time() - start_time
            )
        
        # Execute via Suitelet
        data = execute_suiteql_via_suitelet(sql)
        
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