from dotenv import load_dotenv
import os
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_result(user_id, data):
    return supabase.table('results').insert({"user_id": user_id, **data}).execute()

def fetch_results(user_id):
    return supabase.table('results').select('*').eq('user_id', user_id).execute()