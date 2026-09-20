export const API = (process.env.NEXT_PUBLIC_API_URL || "/api").replace(/\/$/, "");
export const roles = ["top", "jng", "mid", "bot", "sup"] as const;
export type Role = typeof roles[number];
export type Roster = Record<Role, string>;
export interface Prediction {
  team1: string; team2: string; best_of: number;
  team1_win_rate: number; team2_win_rate: number;
  uses_cl_history?: boolean;
  model_version: string; last_game_at: string;
  rosters: {role: string; player_id: string; name: string}[][];
}
export interface Snapshot extends Prediction {
  prediction_id: string; predicted_at: string; scheduled_start: string;
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `요청을 완료하지 못했습니다 (${response.status}).`);
  }
  return response.json();
}
