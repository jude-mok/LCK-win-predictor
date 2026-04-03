from supabase import create_client, Client
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path("/Users/seungyunmok/Developer/LOL_ML/.env")
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

