from fastapi import APIRouter, HTTPException
from app.services.database import supabase
from datetime import date

router = APIRouter(prefix="/schedule", tags=["schedule"])

@router.get("/")
def get_current_week():
    today = date.today().isoformat()
    
    response = supabase.table("schedule")\
        .select("*")\
        .gte("date", today)\
        .order("date")\
        .limit(1)\
        .execute()
    
    if not response.data:
        raise HTTPException(status_code=404, detail="No upcoming matches")
    
    current_week = response.data[0]["week"]
    
    matches = supabase.table("schedule")\
        .select("*")\
        .eq("week", current_week)\
        .order("date")\
        .execute()
    
    return matches.data

@router.get("/entire")
def get_entire_schedule():
    response = supabase.table("schedule")\
        .select("*")\
        .order("date")\
        .execute()
    return response.data

@router.get("/entire/round/{round_number}")
def get_schedule_by_round(round_number: int):
    response = supabase.table("schedule")\
        .select("*")\
        .eq("round", round_number)\
        .order("date")\
        .execute()
    return response.data

@router.get("/entire/week/{week_number}")
def get_schedule_by_week(week_number: int):
    response = supabase.table("schedule")\
        .select("*")\
        .eq("week", week_number)\
        .order("date")\
        .execute()
    return response.data

@router.get("/{match_id}")
def get_match_by_id(match_id: int):
    response = supabase.table("schedule")\
        .select("*")\
        .eq("id", match_id)\
        .single()\
        .execute()
    return response.data