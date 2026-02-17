import sqlite3
import datetime
import json
import hashlib
import os
import numpy as np
import logging
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# --- Configuration ---
# API Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"  # Updated Base URL

# Caching Config
MAX_CACHE_SIZE = 50      # Keep small for demo (LRU eviction test)
CACHE_TTL_SECONDS = 86400 # 24 Hours
SIMILARITY_THRESHOLD = 0.80
COST_PER_1M_TOKENS = 0.60
TOKENS_PER_REQ = 800

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AICacheSystem")

app = FastAPI()

# --- CORS (Crucial for Assignment Submission) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Setup ---
DB_NAME = "smart_cache.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Cache Table: Stores query, response, embedding, and access metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                response_text TEXT,
                embedding TEXT,
                created_at REAL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 1
            )
        ''')
        # Analytics Table: Logs every hit/miss for the GET /analytics endpoint
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT, -- 'HIT', 'MISS'
                latency_ms REAL,
                timestamp REAL
            )
        ''')
        conn.commit()

init_db()

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    application: str = "customer support chatbot"

class ChatResponse(BaseModel):
    answer: str
    cached: bool
    latency: float
    cacheKey: str

# --- Helper Functions ---

def get_utc_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).timestamp()

def compute_cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b) if (norm_a and norm_b) else 0.0

async def get_embedding(text: str):
    """Fetch embedding from OpenAI (or Generate Mock if API fails)."""
    if not OPENAI_API_KEY:
        # Deterministic Mock Embedding for testing without key
        np.random.seed(len(text)) 
        return np.random.rand(1536).tolist()

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Embedding failed: {e}. Using random vector.")
        return np.random.rand(1536).tolist()

async def get_llm_response(text: str):
    """Fetch answer from OpenAI (or Mock)."""
    if not OPENAI_API_KEY:
        return f"Mock AI Response for: {text}"

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Call failed: {e}")
        return "System is currently offline (Mock Fallback)."

# --- Caching Core Logic ---

def clean_cache():
    """Implements TTL (Time-To-Live) and LRU (Least Recently Used) eviction."""
    now = get_utc_timestamp()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. TTL Policy: Delete expired items
    expiry_time = now - CACHE_TTL_SECONDS
    cursor.execute("DELETE FROM cache_entries WHERE created_at < ?", (expiry_time,))
    
    # 2. LRU Policy: If cache is too big, remove oldest 'last_accessed'
    cursor.execute("SELECT count(*) FROM cache_entries")
    count = cursor.fetchone()[0]
    
    if count >= MAX_CACHE_SIZE:
        # Remove the one item that hasn't been used for the longest time
        cursor.execute("""
            DELETE FROM cache_entries 
            WHERE query_hash = (
                SELECT query_hash FROM cache_entries ORDER BY last_accessed ASC LIMIT 1
            )
        """)
    
    conn.commit()
    conn.close()

def log_event(event_type: str, latency: float):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            "INSERT INTO request_logs (event_type, latency_ms, timestamp) VALUES (?, ?, ?)",
            (event_type, latency, get_utc_timestamp())
        )

# --- API Endpoints ---

@app.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.datetime.now()
    clean_cache() # Run cleanup routine
    
    query_hash = hashlib.md5(request.query.encode()).hexdigest()
    
    # --- STRATEGY 1: EXACT MATCH ---
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM cache_entries WHERE query_hash = ?", (query_hash,))
    exact_match = cursor.fetchone()
    
    if exact_match:
        # HIT (Exact)
        cursor.execute("UPDATE cache_entries SET last_accessed = ?, access_count = access_count + 1 WHERE query_hash = ?", 
                       (get_utc_timestamp(), query_hash))
        conn.commit()
        latency = (datetime.datetime.now() - start_time).total_seconds() * 1000
        log_event("HIT", latency)
        return {
            "answer": exact_match['response_text'],
            "cached": True,
            "latency": latency,
            "cacheKey": f"EXACT-{query_hash[:8]}"
        }

# --- STRATEGY 2: SEMANTIC MATCH ---
    query_embedding = await get_embedding(request.query)
    
    cursor.execute("SELECT query_hash, response_text, embedding FROM cache_entries")
    rows = cursor.fetchall()
    
    best_score = -1
    best_entry = None
    
    print(f"\n--- Checking Semantic Match for: '{request.query}' ---") # <--- DEBUG
    
    for row in rows:
        cached_emb = json.loads(row['embedding'])
        score = compute_cosine_similarity(query_embedding, cached_emb)
        
        # Print the score to your terminal
        print(f"   vs cached item: {score:.4f}") # <--- DEBUG
        
        if score > best_score:
            best_score = score
            best_entry = row

    print(f"Best Score: {best_score:.4f} (Threshold: {SIMILARITY_THRESHOLD})") # <--- DEBUG

    if best_score > SIMILARITY_THRESHOLD:
        # ... rest of the code ...
        # HIT (Semantic)
        cursor.execute("UPDATE cache_entries SET last_accessed = ?, access_count = access_count + 1 WHERE query_hash = ?", 
                       (get_utc_timestamp(), best_entry['query_hash']))
        conn.commit()
        latency = (datetime.datetime.now() - start_time).total_seconds() * 1000
        log_event("HIT", latency)
        return {
            "answer": best_entry['response_text'],
            "cached": True,
            "latency": latency,
            "cacheKey": f"SEMANTIC-{best_score:.2f}"
        }

    # --- STRATEGY 3: CACHE MISS (CALL LLM) ---
    llm_response = await get_llm_response(request.query)
    
    # Store new entry
    cursor.execute(
        "INSERT OR REPLACE INTO cache_entries (query_hash, query_text, response_text, embedding, created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?)",
        (query_hash, request.query, llm_response, json.dumps(query_embedding), get_utc_timestamp(), get_utc_timestamp())
    )
    conn.commit()
    conn.close()
    
    latency = (datetime.datetime.now() - start_time).total_seconds() * 1000
    log_event("MISS", latency)
    
    return {
        "answer": llm_response,
        "cached": False,
        "latency": latency,
        "cacheKey": "MISS-NEW_ENTRY"
    }

@app.get("/analytics")
async def analytics():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Stats from DB
    cursor.execute("SELECT count(*) FROM request_logs")
    total_requests = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM request_logs WHERE event_type='HIT'")
    cache_hits = cursor.fetchone()[0]
    
    cursor.execute("SELECT count(*) FROM cache_entries")
    cache_size = cursor.fetchone()[0]
    
    cache_misses = total_requests - cache_hits
    hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0
    
    # Calculate Savings
    # Formula: Hits * Tokens_Saved * Cost
    # Cost = (Tokens / 1,000,000) * 0.60
    tokens_saved = cache_hits * TOKENS_PER_REQ
    cost_savings = (tokens_saved / 1000000) * COST_PER_1M_TOKENS
    
    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total_requests,
        "cacheHits": cache_hits,
        "cacheMisses": cache_misses,
        "cacheSize": cache_size,
        "costSavings": round(cost_savings, 4),
        "savingsPercent": round(hit_rate * 100, 1),
        "strategies": ["exact match (MD5)", "semantic similarity (Cosine)", "LRU eviction", "TTL expiration"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)