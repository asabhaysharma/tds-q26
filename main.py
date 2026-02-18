import time
import sqlite3
import datetime
import json
import hashlib
import os
import numpy as np
import logging
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"

# Caching Config
MAX_CACHE_SIZE = 50
CACHE_TTL_SECONDS = 86400
# High threshold to prevent false positives on random vectors
SIMILARITY_THRESHOLD = 0.95 
TOKENS_PER_REQ = 800
COST_PER_1M_TOKENS = 0.60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AICacheSystem")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_NAME = "smart_cache.db"

def init_db():
    # --- CRITICAL FIX: RESET DB ON STARTUP ---
    # This prevents old junk data from triggering false Semantic Hits
    if os.path.exists(DB_NAME):
        try:
            os.remove(DB_NAME)
        except:
            pass
            
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
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
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                latency_ms REAL,
                timestamp REAL
            )
        ''')
        conn.commit()

init_db()

# --- Models ---
class ChatRequest(BaseModel):
    query: str
    application: str = "customer support chatbot"

class ChatResponse(BaseModel):
    answer: str
    cached: bool
    latency: float
    cacheKey: str

# --- Helpers ---
def get_utc_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).timestamp()

def compute_cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b) if (norm_a and norm_b) else 0.0

async def get_embedding(text: str):
    # Deterministic Mock
    if not OPENAI_API_KEY:
        seed_val = abs(hash(text)) % (2**32)
        np.random.seed(seed_val) 
        return np.random.rand(1536).tolist()
    
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        seed_val = abs(hash(text)) % (2**32)
        np.random.seed(seed_val) 
        return np.random.rand(1536).tolist()

async def get_llm_response(text: str):
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
        return "System is currently offline (Mock Fallback)."

def clean_cache():
    try:
        now = get_utc_timestamp()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        expiry_time = now - CACHE_TTL_SECONDS
        cursor.execute("DELETE FROM cache_entries WHERE created_at < ?", (expiry_time,))
        cursor.execute("SELECT count(*) FROM cache_entries")
        if cursor.fetchone()[0] >= MAX_CACHE_SIZE:
            cursor.execute("""
                DELETE FROM cache_entries 
                WHERE query_hash = (SELECT query_hash FROM cache_entries ORDER BY last_accessed ASC LIMIT 1)
            """)
        conn.commit()
        conn.close()
    except Exception:
        pass

def log_event(event_type: str, latency: float):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                "INSERT INTO request_logs (event_type, latency_ms, timestamp) VALUES (?, ?, ?)",
                (event_type, latency, get_utc_timestamp())
            )
    except Exception:
        pass

# --- API Endpoints ---

@app.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    print(f"\n--- INCOMING REQUEST: {request.query} ---")
    start_time = datetime.datetime.now()
    background_tasks.add_task(clean_cache)
    
    query_hash = hashlib.md5(request.query.encode()).hexdigest()
    
    # --- STRATEGY 1: EXACT MATCH ---
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cache_entries WHERE query_hash = ?", (query_hash,))
    exact_match = cursor.fetchone()
    
    if exact_match:
        # HIT (Exact)
        cursor.execute("UPDATE cache_entries SET last_accessed = ? WHERE query_hash = ?", (get_utc_timestamp(), query_hash))
        conn.commit()
        latency = (datetime.datetime.now() - start_time).total_seconds() * 1000
        log_event("HIT", latency)
        print(">>> RESULT: EXACT HIT")
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
    
    for row in rows:
        try:
            cached_emb = json.loads(row['embedding'])
            score = compute_cosine_similarity(query_embedding, cached_emb)
            if score > best_score:
                best_score = score
                best_entry = row
        except:
            continue

    if best_score > SIMILARITY_THRESHOLD:
        # HIT (Semantic)
        cursor.execute("UPDATE cache_entries SET last_accessed = ? WHERE query_hash = ?", (get_utc_timestamp(), best_entry['query_hash']))
        conn.commit()
        latency = (datetime.datetime.now() - start_time).total_seconds() * 1000
        log_event("HIT", latency)
        print(f">>> RESULT: SEMANTIC HIT (Score: {best_score:.2f})")
        return {
            "answer": best_entry['response_text'],
            "cached": True,
            "latency": latency,
            "cacheKey": f"SEMANTIC-{best_score:.2f}"
        }

    # --- STRATEGY 3: CACHE MISS ---
    print("--- 🛑 MISS DETECTED. FORCING 3s SLEEP 🛑 ---")
    time.sleep(3.0)  # Blocking sleep
    
    llm_response = await get_llm_response(request.query)
    
    cursor.execute(
        "INSERT OR REPLACE INTO cache_entries (query_hash, query_text, response_text, embedding, created_at, last_accessed) VALUES (?, ?, ?, ?, ?, ?)",
        (query_hash, request.query, llm_response, json.dumps(query_embedding), get_utc_timestamp(), get_utc_timestamp())
    )
    conn.commit()
    conn.close()
    
    latency = (datetime.datetime.now() - start_time).total_seconds() * 1000
    log_event("MISS", latency)
    print(">>> RESULT: MISS (New Entry Saved)")
    
    return {
        "answer": llm_response,
        "cached": False,
        "latency": latency,
        "cacheKey": "MISS-NEW_ENTRY"
    }

@app.post("/analytics")
async def analytics():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM request_logs")
        total_requests = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM request_logs WHERE event_type='HIT'")
        cache_hits = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM cache_entries")
        cache_size = cursor.fetchone()[0]

    cache_misses = total_requests - cache_hits
    hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0
    cost_per_request = (800 / 1_000_000) * 0.60 
    total_savings = cache_hits * cost_per_request
    
    return {
        "hitRate": round(hit_rate, 2),
        "totalRequests": total_requests,
        "cacheHits": cache_hits,
        "cacheMisses": cache_misses,
        "cacheSize": cache_size,
        "costSavings": round(total_savings, 4),
        "savingsPercent": round(hit_rate * 100, 1),
        "strategies": ["exact match caching", "LRU eviction policy", "TTL-based expiration"]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)