"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Prediction, roles, Roster } from "../../lib/api";

type Catalog = {teams: Record<string, {roster: Roster}>; players: Record<string, {name: string; role: string; team: string}>; last_game_at: string};
export default function Simulate() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [teams, setTeams] = useState<string[]>([]);
  const [rosters, setRosters] = useState<Roster[]>([]);
  const [bestOf, setBestOf] = useState(3);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api<Catalog>("/predict/players").then(data => {
      if (cancelled) return;
      const names = Object.keys(data.teams).slice(0, 2);
      setCatalog(data); setTeams(names); setRosters(names.map(n => data.teams[n].roster));
    }).catch(e => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, []);
  async function simulate() {
    setBusy(true); setError(""); setResult(null);
    try {
      setResult(await api<Prediction>("/predict/simulate", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({team1: teams[0], team2: teams[1], best_of: bestOf, team1_roster: rosters[0], team2_roster: rosters[1]})}));
    } catch (e) { setError(e instanceof Error ? e.message : "계산하지 못했습니다."); }
    finally { setBusy(false); }
  }
  return <main className="min-h-screen bg-[#faf8ff] px-6 py-16 text-slate-900"><div className="max-w-4xl mx-auto space-y-8">
    <Link href="/" className="text-blue-700">← 경기 목록</Link>
    <h1 className="text-4xl font-bold">선수 명단 시뮬레이션</h1>
    <p className="text-slate-600">기본 명단은 마지막 경기에서 관측한 선수입니다. 다음 경기의 확정 명단이 아니며, 계산 결과는 공식 경기 기록으로 저장되지 않습니다.</p>
    {error && <p role="alert" className="text-red-700">{error}</p>}
    {!catalog && !error && <p>선수 명단을 불러오는 중…</p>}
    {catalog && <>
      <p className="text-sm text-slate-600">경기 데이터 기준: {new Date(catalog.last_game_at).toLocaleString("ko-KR")}</p>
      <fieldset disabled={busy} className="space-y-6 disabled:opacity-60">
        <div className="grid md:grid-cols-2 gap-6">{teams.map((team, i) => <section key={i} className="bg-white rounded-2xl p-6 space-y-4 shadow-sm">
          <label className="block font-bold">팀 {i + 1}<select className="block border rounded-lg p-3 w-full mt-2" value={team} onChange={e => {
            setTeams(teams.map((t, j) => j === i ? e.target.value : t));
            setRosters(rosters.map((r, j) => j === i ? catalog.teams[e.target.value].roster : r)); setResult(null);
          }}>{Object.keys(catalog.teams).map(name => <option key={name}>{name}</option>)}</select></label>
          {roles.map(role => <label key={role} className="block text-sm"><span className="uppercase">{role}</span>
            <select className="block border rounded-lg p-2 w-full mt-1" value={rosters[i][role]} onChange={e => {
              setRosters(rosters.map((r, j) => j === i ? {...r, [role]: e.target.value} : r)); setResult(null);
            }}>{Object.entries(catalog.players).filter(([, p]) => p.role === role).map(([id, p]) => <option key={id} value={id}>{p.name} · {p.team}</option>)}</select>
          </label>)}
        </section>)}</div>
        <label className="inline-flex gap-3 items-center">경기 형식<select className="bg-white border rounded-lg p-2" value={bestOf} onChange={e => {setBestOf(Number(e.target.value));setResult(null);}}><option value={3}>BO3</option><option value={5}>BO5</option></select></label>
        <button className="block rounded-full bg-blue-700 text-white px-8 py-3 font-bold disabled:opacity-50" disabled={teams[0] === teams[1]} onClick={simulate}>{busy ? "계산 중…" : "승률 계산"}</button>
      </fieldset>
    </>}
    {result && <section aria-live="polite" className="bg-blue-700 text-white p-8 rounded-2xl space-y-3">
      <h2 className="font-bold">시뮬레이션 결과 · BO{result.best_of}</h2>
      <p className="text-2xl">{result.team1} {(result.team1_win_rate * 100).toFixed(1)}% · {result.team2} {(result.team2_win_rate * 100).toFixed(1)}%</p>
      <p className="text-sm opacity-80">선수 A · 모델 {result.model_version.slice(0, 12)} · 저장되지 않은 결과</p>
    </section>}
  </div></main>;
}
