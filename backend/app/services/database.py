"""Schedule access is optional; model serving does not require Supabase credentials."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from supabase import create_client

load_dotenv(Path(__file__).resolve().parents[3] / '.env')


@lru_cache(maxsize=1)
def get_supabase():
    url, key = os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY')
    if not url or not key:
        raise HTTPException(503, 'Schedule database is not configured')
    return create_client(url, key)
