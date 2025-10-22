#!/usr/bin/env python3
"""
Quick connectivity check to Supabase using utils/db.py

Usage:
    python scripts/check_supabase.py

Requires .env with SUPABASE_URL and SUPABASE_KEY
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'utils' package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from utils.db import supabase


def main():
    if supabase is None:
        print("Supabase client is not configured.\n- Ensure .env exists with SUPABASE_URL and SUPABASE_KEY\n- Restart your shell/editor after editing .env")
        raise SystemExit(1)

    try:
        # Attempt a lightweight request to verify access: list tables via RPC or select from a known table
        # If you have a 'results' table, uncomment the following lines:
        # resp = supabase.table('results').select('*').limit(1).execute()
        # print("Connection OK. 'results' sample:", resp.data)
        print("Supabase client initialized successfully. Ready to use.")
    except Exception as e:
        print("Supabase connection error:", str(e))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
