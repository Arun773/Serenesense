from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://placeholder.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "placeholder_key")

# Only create client if valid credentials are provided
try:
    if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "https://placeholder.supabase.co" and SUPABASE_KEY != "placeholder_key":
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    else:
        supabase = None
except Exception as e:
    print(f"Warning: Could not initialize Supabase client: {e}")
    supabase = None

def store_result(user_id, data):
    if supabase is None:
        print("Warning: Supabase not configured. Data not stored.")
        return {"data": None, "error": "Database not configured"}
    return supabase.table('results').insert({"user_id": user_id, **data}).execute()

def fetch_results(user_id):
    if supabase is None:
        print("Warning: Supabase not configured. Returning empty results.")
        return {"data": [], "error": "Database not configured"}
    return supabase.table('results').select('*').eq('user_id', user_id).execute()