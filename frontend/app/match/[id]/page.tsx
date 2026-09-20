"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import styles from "./match.module.css";
import { api, Prediction, roles, Role } from "../../../lib/api";

type Player = {player_id: string; player_name: string; position: Role; roster_status: "Main" | "sub"; available: boolean};
type Team = {team_name: string; roster: Record<Role, string>; players: Player[]};
type Catalog = {teams: Record<string, Team>};
type Match = {team_1: string; team_2: string; date: string; time: string; winner: string | null};
type Selection = {team1_roster?: Partial<Record<Role, string>>; team2_roster?: Partial<Record<Role, string>>};

export default function MatchDetail() {
  const { id } = useParams();
  const [context, setContext] = useState<{match: Match; catalog: Catalog} | null>(null);
  const [selection, setSelection] = useState<Selection>({});
  const [bestOf, setBestOf] = useState(3);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      setContext(null); setSelection({}); setResult(null); setError(""); setBusy(true);
      try {
        const [match, catalog] = await Promise.all([
          api<Match>(`/schedule/${id}`, {signal: controller.signal}),
          api<Catalog>("/predict/rosters", {signal: controller.signal}),
        ]);
        if (!controller.signal.aborted) setContext({match, catalog});
      } catch (e) { if (!controller.signal.aborted) {setError(e instanceof Error ? e.message : "불러오지 못했습니다.");setBusy(false);} }
    }
    void load();
    return () => controller.abort();
  }, [id]);
  useEffect(() => {
    if (!context) return;
    const controller = new AbortController();
    async function predict() {
      setBusy(true); setResult(null); setError("");
      try {
        const prediction = await api<Prediction>("/predict/predict", {method: "POST", signal: controller.signal,
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({team1: context!.match.team_1, team2: context!.match.team_2, best_of: bestOf, ...selection})});
        if (!controller.signal.aborted) setResult(prediction);
      } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "계산하지 못했습니다."); }
      finally { if (!controller.signal.aborted) setBusy(false); }
    }
    void predict();
    return () => controller.abort();
  }, [context, selection, bestOf]);

  const changed = Object.values(selection).reduce((count, roster) => count + Object.keys(roster || {}).length, 0);
  const names: Record<string,string> = {HLE:"한화생명",BRO:"한진 브리온",GEN:"Gen.G",T1:"T1",DK:"Dplus KIA",KT:"KT 롤스터",BFX:"BNK 피어엑스",NS:"농심 레드포스",KRX:"키움 DRX",DNS:"DN 수퍼스"};
  const roleNames: Record<Role,string> = {top:"탑",jng:"정글",mid:"미드",bot:"원딜",sup:"서포터"};
  return <div className={styles.page}>
    <nav className={styles.nav}><div><Link href="/" className={styles.brand}><span className={styles.brandMark}>L</span>LCK Predict</Link><Link href="/" className={styles.navLink}>경기 일정 <span aria-hidden>↗</span></Link></div></nav>
    <main className={styles.main}>
      <Link href="/" className={styles.back}>〈 <span>경기 목록</span></Link>
      <header className={styles.intro}><p>LCK MATCH PREVIEW</p><h1>이 라인업이라면,<br className={styles.mobileBreak}/> 누가 이길까?</h1><span>선수를 바꿔보며 우리 팀의 승률을 확인해 보세요.</span></header>
      <section className={styles.prediction} aria-label="경기 승률" aria-busy={busy}>
        <div className={styles.cardTop}><span className={styles.badge}>AI 승률 예측</span><div className={styles.segment} aria-label="경기 형식">{[3,5].map(n => <button key={n} aria-pressed={bestOf===n} onClick={() => {setResult(null);setBestOf(n);}}>BO{n}</button>)}</div></div>
        {context ? <>
          <p className={styles.date}>{context.match.date.replaceAll("-", ".")} <span>·</span> {context.match.time.slice(0,5)} KST</p>
          <div className={styles.matchup}>
            {[context.match.team_1,context.match.team_2].map((code,i) => <div key={code} className={styles.contender}>
              <div className={styles.logo}><Image src={`/logos/${code}.png`} alt="" width={64} height={64}/></div>
              <h2>{names[code] || code}</h2>
              <p className={i===0 ? styles.blueOdds : styles.darkOdds}>{result ? ((i===0 ? result.team1_win_rate : result.team2_win_rate)*100).toFixed(1) : "—"}<span>%</span></p>
            </div>)}
            <span className={styles.vs}>VS</span>
          </div>
          <div className={styles.probabilityBar}>{result && <><span style={{width:`${result.team1_win_rate*100}%`}}/><span style={{width:`${result.team2_win_rate*100}%`}}/></>}</div>
          <div className={styles.predictionCaption} aria-live="polite">{busy ? "선택한 라인업으로 계산하고 있어요…" : error ? "예측을 불러오지 못했어요" : changed ? "변경한 라인업으로 계산했어요" : "현재 주전 라인업 기준이에요"}<span>BO{bestOf} · {bestOf===3 ? "2" : "3"}선승제</span></div>
        </> : <div className={styles.loading}>경기를 불러오고 있어요…</div>}
      </section>
      {error && <p role="alert" className={styles.error}>{error}</p>}
      {context && <>
        <div className={styles.sectionHeading}><div><h2>출전 라인업 <span>10명</span></h2><p>선수를 누르면 같은 포지션의 후보로 교체할 수 있어요.</p></div><button disabled={!changed} onClick={() => {setResult(null);setSelection({});}}>↺ 기본 명단</button></div>
        <div className={styles.rosters}>{[context.match.team_1,context.match.team_2].map((code,i) => {
          const team = context.catalog.teams[code];
          const key = i===0 ? "team1_roster" : "team2_roster";
          return <section key={code} className={styles.rosterCard}>
            <div className={styles.teamHeading}><Image src={`/logos/${code}.png`} alt="" width={32} height={32}/><h3>{names[code] || code}</h3><span>{code}</span></div>
            {team ? roles.map(role => {
              const selectedId = selection[key]?.[role] ?? team.roster[role] ?? "";
              const player = team.players.find(p => p.player_id === selectedId);
              const swapped = selectedId !== team.roster[role];
              return <label key={role} className={`${styles.playerRow} ${swapped ? styles.swapped : ""}`}>
                <span className={styles.roleIcon}>{role === "sup" ? "✦" : role === "jng" ? "♧" : role === "mid" ? "◇" : role === "bot" ? "◎" : "⌁"}</span>
                <span className={styles.playerIdentity}><strong>{player?.player_name || "선수 미설정"}</strong><small>{roleNames[role]} <span>· {role.toUpperCase()}</span></small></span>
                <span className={swapped ? styles.subBadge : styles.mainBadge}>{swapped ? "교체" : "Main"}</span><span className={styles.chevron}>⌄</span>
                <select aria-label={`${code} ${roleNames[role]} 선수`} value={selectedId} onChange={e => {
                  const value=e.target.value;setResult(null);setSelection(prev => {
                    const roster={...prev[key]};
                    if(value===team.roster[role]) delete roster[role]; else roster[role]=value;
                    return {...prev,[key]:roster};
                  });
                }}>
                  {!team.roster[role] && <option value="" disabled>Main 미설정</option>}
                  {team.players.filter(p => p.position===role).sort((a,b) => Number(b.roster_status==="Main")-Number(a.roster_status==="Main")).map(p => <option key={p.player_id} value={p.player_id} disabled={!p.available}>{p.player_name} · {p.roster_status}{!p.available ? " · 경기 데이터 없음" : ""}</option>)}
                </select>
              </label>;
            }) : <p className={styles.loading}>등록된 명단이 없어요.</p>}
          </section>;
        })}</div>
        <aside className={styles.note}><span aria-hidden>ⓘ</span><div><strong>자유롭게 바꿔봐도 괜찮아요</strong><p>선수 교체는 이번 예측에만 적용돼요. 팀의 기본 명단은 그대로 유지됩니다.</p></div></aside>
      </>}
      <footer className={styles.footer}>
        <p>현재 명단과 보유 경기 데이터로 계산한 예상 승률이에요. 과거 경기 당시의 예측과는 달라요.</p>
        {result && <p>데이터 기준 {new Date(result.last_game_at).toLocaleDateString("ko-KR")} · Player A</p>}
        {result?.uses_cl_history && <p>일부 선수는 CL 경기 통계를 사용해요. CL 선수의 예측 정확도는 별도 검증 전입니다.</p>}
      </footer>
    </main>
  </div>;
}
