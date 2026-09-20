"""Resolve live DB Main lineups and validate per-request substitutions."""
from fastapi import HTTPException
from ..schemas.player import Roster, SimulationRequest
from .database import get_supabase
from .player_runtime import ROLES


def catalog(service):
    try:
        rows = get_supabase().table('players').select(
            'team_code,team_name,position,player_name,player_id,league,roster_status,roster_as_of'
        ).execute().data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, '선수 DB를 불러오지 못했습니다.') from exc
    teams = {}
    for row in rows:
        if row['position'] not in ROLES or row['roster_status'] not in ('Main', 'sub'):
            raise HTTPException(503, '선수 DB의 position 또는 roster_status 설정을 확인해 주세요.')
        team = teams.setdefault(row['team_code'], {'team_code': row['team_code'], 'team_name': row['team_name'], 'players': [], 'roster': {}})
        history = service.state['players'].get(row['player_id'])
        team['players'].append({**row, 'available': history is not None,
                                'history_league': history.get('history_league', 'LCK') if history else None})
        if row['roster_status'] == 'Main':
            if row['position'] in team['roster']:
                raise HTTPException(503, f"{row['team_code']} {row['position']}: Main은 한 명이어야 합니다.")
            team['roster'][row['position']] = row['player_id']
    return teams


def resolve(body, teams):
    selected, names = [], []
    for team_key, overrides in [(body.team1, body.team1_roster), (body.team2, body.team2_roster)]:
        matches = [t for code, t in teams.items() if team_key in (code, t['team_name'])]
        if len(matches) != 1:
            raise HTTPException(422, f'알 수 없는 팀: {team_key}')
        team = matches[0]
        if set(team['roster']) != set(ROLES):
            raise HTTPException(422, f"{team['team_code']}: 포지션별 Main 5명을 설정해 주세요.")
        roster = {**team['roster'], **(overrides or {})}
        for role, player_id in roster.items():
            player = next((p for p in team['players'] if p['player_id'] == player_id and p['position'] == role), None)
            if not player:
                raise HTTPException(422, '해당 팀·포지션에 등록된 선수만 선택할 수 있습니다.')
            if not player['available']:
                raise HTTPException(422, f"{player['player_name']}: 경기 피처를 먼저 업데이트해야 합니다.")
        selected.append(Roster(**roster))
        names.append(team['team_name'])
    if names[0] == names[1]:
        raise HTTPException(422, '서로 다른 팀을 선택해 주세요.')
    return SimulationRequest(team1=names[0], team2=names[1], best_of=body.best_of,
                             team1_roster=selected[0], team2_roster=selected[1])
