"""
Weily AI Backend Server
Deployed on Render/Railway - Always Running
Handles AI processing and NetSuite data access
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

app = FastAPI(title="Weily AI Backend")

# CORS - Allow NetSuite domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your NetSuite domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# NetSuite Config
NETSUITE_CONFIG = {
    'account_id': os.getenv('NETSUITE_ACCOUNT_ID'),
    'consumer_key': os.getenv('NETSUITE_CONSUMER_KEY'),
    'consumer_secret': os.getenv('NETSUITE_CONSUMER_SECRET'),
    'token_id': os.getenv('NETSUITE_TOKEN_ID'),
    'token_secret': os.getenv('NETSUITE_TOKEN_SECRET'),
    'base_url': f"https://{os.getenv('NETSUITE_ACCOUNT_ID')}.suitetalk.api.netsuite.com"
}

class QueryRequest(BaseModel):
    query: str
    user_context: dict = {}

class QueryResponse(BaseModel):
    success: bool
    query_type: str
    data: dict = None
    response_text: str
    error: str = None

def generate_oauth_header(url, method='GET'):
    """Generate OAuth 1.0 header"""
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
    params_string = '&'.join([f'{k}={quote(str(v), safe="")}' for k, v in sorted(signature_params.items())])
    base_string = f'{method}&{quote(url, safe="")}&{quote(params_string, safe="")}'
    signing_key = f'{quote(NETSUITE_CONFIG["consumer_secret"], safe="")}&{quote(NETSUITE_CONFIG["token_secret"], safe="")}'
    
    signature = base64.b64encode(
        hmac.new(signing_key.encode('utf-8'), base_string.encode('utf-8'), hashlib.sha256).digest()
    ).decode('utf-8')
    
    oauth_params['oauth_signature'] = signature
    auth_header = 'OAuth ' + ','.join([f'{k}="{v}"' for k, v in sorted(oauth_params.items())])
    
    return auth_header

def execute_suiteql(query):
    """Execute SuiteQL query"""
    url = f"{NETSUITE_CONFIG['base_url']}/services/rest/query/v1/suiteql"
    headers = {
        'Content-Type': 'application/json',
        'prefer': 'transient',
        'Authorization': generate_oauth_header(url, 'POST')
    }
    
    try:
        response = requests.post(url, headers=headers, json={'q': query}, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"SuiteQL Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"SuiteQL Exception: {str(e)}")
        return None

def analyze_query(user_query):
    """Analyze query using OpenAI"""
    system_prompt = """You are Weily AI for NetSuite. Analyze queries and return JSON.

Query types:
- sales_summary: Sales data
- sales_by_customer: Top customers
- invoice_details: Invoice info
- inventory_status: Stock levels
- accounts_receivable: Money owed to us
- accounts_payable: Bills to pay
- customers_list: Customer list
- general: Other queries

Return JSON:
{
    "query_type": "type_here",
    "params": {"key": "value"}
}
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"AI Analysis Error: {str(e)}")
        return {"query_type": "general", "params": {}}

def generate_response(user_query, data):
    """Generate natural language response"""
    context = f"Data: {json.dumps(data, indent=2)}" if data else "No data."
    
    system_prompt = """You are Weily AI. Provide clear, professional business insights.
Format numbers with currency symbols. Be concise."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {user_query}\n\n{context}"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.get("/")
async def root():
    return {"message": "Weily AI Backend Running", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Weily AI"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query and return NetSuite data + AI response"""
    
    try:
        # Analyze query
        analysis = analyze_query(request.query)
        query_type = analysis.get('query_type', 'general')
        params = analysis.get('params', {})
        
        data = None
        
        # Execute NetSuite queries based on type
        if query_type == 'sales_summary':
            months = params.get('months', 1)
            sql = f"""
            SELECT 
                SUM(foreigntotal) as total_sales,
                COUNT(*) as transaction_count,
                AVG(foreigntotal) as average_sale
            FROM transaction
            WHERE type = 'CustInvc'
            AND trandate >= ADD_MONTHS(CURRENT_DATE, -{months})
            AND status NOT IN ('Voided', 'Cancelled')
            """
            data = execute_suiteql(sql)
        
        elif query_type == 'sales_by_customer':
            months = params.get('months', 1)
            limit = params.get('limit', 10)
            sql = f"""
            SELECT 
                c.companyname as customer_name,
                SUM(t.foreigntotal) as total_sales,
                COUNT(t.id) as order_count
            FROM transaction t
            INNER JOIN customer c ON t.entity = c.id
            WHERE t.type = 'CustInvc'
            AND t.trandate >= ADD_MONTHS(CURRENT_DATE, -{months})
            GROUP BY c.companyname
            ORDER BY total_sales DESC
            LIMIT {limit}
            """
            data = execute_suiteql(sql)
        
        elif query_type == 'invoice_details':
            invoice_number = params.get('invoice_number', '')
            sql = f"""
            SELECT 
                t.tranid as invoice_number,
                t.trandate as date,
                c.companyname as customer,
                t.foreigntotal as amount,
                t.status
            FROM transaction t
            INNER JOIN customer c ON t.entity = c.id
            WHERE t.tranid = '{invoice_number}'
            """
            data = execute_suiteql(sql)
        
        elif query_type == 'inventory_status':
            sql = """
            SELECT 
                itemid,
                displayname,
                quantityavailable,
                cost
            FROM item
            WHERE isinactive = 'F'
            LIMIT 20
            """
            data = execute_suiteql(sql)
        
        elif query_type == 'accounts_receivable':
            sql = """
            SELECT 
                SUM(foreigntotal) as total_ar,
                COUNT(*) as open_invoices
            FROM transaction
            WHERE type = 'CustInvc'
            AND status = 'Open'
            """
            data = execute_suiteql(sql)
        
        # Generate AI response
        response_text = generate_response(request.query, data)
        
        return QueryResponse(
            success=True,
            query_type=query_type,
            data=data,
            response_text=response_text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)