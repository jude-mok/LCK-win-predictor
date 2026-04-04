"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

const TEAMS_MAP: Record<string, string> = {
  T1: "T1",
  GEN: "Gen.G",
  HLE: "한화생명",
  DK: "Dplus KIA",
  KT: "KT 롤스터",
  BFX: "BNK 피어엑스",
  NS: "농심 레드포스",
  KRX: "키움 DRX",
  DNS: "DN 수퍼스",
  BRO: "한진 브리온",
};

const PREDICT_NAME_MAP: Record<string, string> = {
  T1: "T1",
  GEN: "Gen.G",
  HLE: "Hanwha Life Esports",
  DK: "Dplus Kia",
  KT: "KT Rolster",
  BFX: "BNK FEARX",
  NS: "Nongshim RedForce",
  KRX: "DRX",
  DNS: "DN SOOPers",
  BRO: "HANJIN BRION",
};

const METRIC_LABELS: Record<string, { label: string; subtitle: string; icon: string }> = {
  roll_winrate:     { label: "최근 승률",        subtitle: "최근 성적",      icon: "trending_up" },
  roll_golddiff15:  { label: "15분 골드 차이",   subtitle: "초반 기세",      icon: "payments" },
  roll_firstdragon: { label: "첫 드래곤 획득률", subtitle: "오브젝트 장악",  icon: "pets" },
  roll_firstherald: { label: "첫 전령 획득률",   subtitle: "매크로 압박",    icon: "swords" },
  roll_firsttower:  { label: "첫 포탑 파괴율",   subtitle: "라인 지배력",    icon: "fort" },
  patch_winrate:    { label: "패치 승률",        subtitle: "메타 적응력",    icon: "terminal" },
};

interface Match {
  id: number;
  team_1: string;
  team_2: string;
  date: string;
  time: string;
  round: number;
  split: string;
  year: number;
  winner: string | null;
  week: number;
}

interface Prediction {
  blue_team: string;
  blue_win_rate: number;
  red_team: string;
  red_win_rate: number;
  predicted_winner: string;
  blue_stats: Record<string, number>;
  red_stats: Record<string, number>;
}

export default function MatchDetail() {
  const { id } = useParams();
  const [match, setMatch] = useState<Match | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;

    fetch(`http://localhost:8000/schedule/${id}`)
      .then((res) => res.json())
      .then(async (matchData: Match) => {
        setMatch(matchData);

        const predRes = await fetch("http://localhost:8000/predict/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            blue_team: PREDICT_NAME_MAP[matchData.team_1],
            red_team: PREDICT_NAME_MAP[matchData.team_2],
          }),
        });
        const predData = await predRes.json();
        setPrediction(predData);
        setLoading(false);
      });
  }, [id]);

  const formatDate = (date: string) => {
    const d = new Date(date);
    return d.toLocaleDateString("ko-KR", { month: "long", day: "numeric", weekday: "short" });
  };

  const formatTime = (time: string) => time.slice(0, 5);

  const formatStatValue = (key: string, value: number) => {
    if (key === "roll_golddiff15") return value > 0 ? `+${Math.round(value)}g` : `${Math.round(value)}g`;
    return `${Math.round(value * 100)}%`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#faf8ff] flex items-center justify-center">
        <p className="text-[#424656]">불러오는 중...</p>
      </div>
    );
  }

  if (!match || !prediction) return null;

  const blueWin = Math.round(prediction.blue_win_rate * 100);
  const redWin = Math.round(prediction.red_win_rate * 100);

  return (
    <div className="bg-[#faf8ff] min-h-screen text-[#191b24]">
      {/* Nav */}
      <nav className="bg-white/80 backdrop-blur-xl shadow-sm sticky top-0 z-50">
        <div className="flex justify-between items-center w-full px-8 py-4 max-w-7xl mx-auto">
          <a href="/" className="text-xl font-black text-blue-600 tracking-tight">
            LCK Predict
          </a>
          <div className="flex items-center gap-4">
            <button className="px-6 py-2 rounded-full font-bold text-sm bg-[#0064ff] text-white transition-transform active:scale-95">
              예측하기
            </button>
            <button className="px-4 py-2 rounded-full font-bold text-sm text-slate-500 hover:text-blue-500">
              로그인
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-8 pt-12 pb-20">

        {/* Back Button */}
        <a
          href="/"
          className="inline-flex items-center gap-2 mb-8 px-4 py-2 rounded-full bg-[#e6e7f4] text-[#424656] font-bold text-sm hover:bg-[#d8d9e6] transition-all"
        >
          <span className="material-symbols-outlined text-base leading-none">arrow_back</span>
          경기 목록으로
        </a>

        {/* Match Header */}
        <section className="relative overflow-hidden rounded-2xl bg-[#f2f3ff] p-12 mb-8">
          <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-12">
            {/* Team 1 */}
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="w-32 h-32 rounded-full bg-white p-4 shadow-sm flex items-center justify-center">
                <img
                  src={`/logos/${match.team_1}.png`}
                  alt={match.team_1}
                  className="w-full h-full object-contain"
                />
              </div>
              <div>
                <h2 className="text-3xl font-extrabold tracking-tight">
                  {TEAMS_MAP[match.team_1] || match.team_1}
                </h2>
              </div>
            </div>

            {/* VS */}
            <div className="flex flex-col items-center">
              <span className="text-5xl font-black text-[#c2c6d8] italic">VS</span>
              <div className="mt-4 px-4 py-1.5 rounded-full bg-[#e1e2ee] text-xs font-bold">
                {formatDate(match.date)} · {formatTime(match.time)} KST
              </div>
              {match.winner && match.winner !== "NULL" && (
                <div className="mt-3 px-4 py-1.5 rounded-full bg-[#004ecb] text-white text-xs font-bold">
                  우승팀: {TEAMS_MAP[match.winner] || match.winner}
                </div>
              )}
            </div>

            {/* Team 2 */}
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="w-32 h-32 rounded-full bg-white p-4 shadow-sm flex items-center justify-center">
                <img
                  src={`/logos/${match.team_2}.png`}
                  alt={match.team_2}
                  className="w-full h-full object-contain"
                />
              </div>
              <div>
                <h2 className="text-3xl font-extrabold tracking-tight">
                  {TEAMS_MAP[match.team_2] || match.team_2}
                </h2>
              </div>
            </div>
          </div>

          {/* Background blur */}
          <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/5 rounded-full -mr-20 -mt-20 blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-orange-500/5 rounded-full -ml-20 -mb-20 blur-3xl"></div>
        </section>

        {/* AI Prediction Banner */}
        <section className="mb-12">
          <div className="bg-[#0064ff] p-8 rounded-2xl text-white flex flex-col md:flex-row items-center justify-between gap-8 relative overflow-hidden">
            <div className="z-10">
              <div className="flex items-center gap-2 mb-2">
                <span className="material-symbols-outlined">psychology</span>
                <span className="text-xs font-bold uppercase tracking-wider opacity-80">AI 분석</span>
              </div>
              <h3 className="text-2xl font-extrabold mb-1">승리 확률</h3>
              <p className="text-white/70 text-sm">최근 경기 데이터 기반</p>
            </div>
            <div className="flex-1 w-full max-w-2xl z-10">
              <div className="flex justify-between mb-3 font-bold text-lg">
                <span>{match.team_1} {blueWin}%</span>
                <span>{redWin}% {match.team_2}</span>
              </div>
              <div className="h-6 w-full bg-white/20 rounded-full flex overflow-hidden">
                <div className="h-full bg-white/80 transition-all" style={{ width: `${blueWin}%` }}></div>
                <div className="h-full bg-white/30 transition-all" style={{ width: `${redWin}%` }}></div>
              </div>
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-[#004ecb] to-[#0064ff] opacity-50"></div>
          </div>
        </section>

        {/* Statistical Breakdown */}
        <section>
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-2xl font-extrabold">통계 분석</h3>
            <div className="flex gap-4">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#004ecb]"></span>
                <span className="text-xs font-bold">{match.team_1}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#a03200]"></span>
                <span className="text-xs font-bold">{match.team_2}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(METRIC_LABELS).map(([key, meta]) => {
              const blueVal = prediction.blue_stats[key] ?? 0;
              const redVal = prediction.red_stats[key] ?? 0;
              const total = Math.abs(blueVal) + Math.abs(redVal);
              const blueWidth = total === 0 ? 50 : Math.round((Math.abs(blueVal) / total) * 100);
              const redWidth = 100 - blueWidth;

              return (
                <div key={key} className="bg-[#f2f3ff] p-6 rounded-2xl">
                  <div className="flex justify-between items-end mb-4">
                    <div>
                      <p className="text-xs text-[#424656] font-bold uppercase tracking-wider">
                        {meta.subtitle}
                      </p>
                      <h4 className="font-bold text-lg">{meta.label}</h4>
                    </div>
                    <span className="material-symbols-outlined text-[#004ecb]">{meta.icon}</span>
                  </div>
                  <div className="flex justify-between text-sm font-bold mb-2">
                    <span className="text-[#004ecb]">{formatStatValue(key, blueVal)}</span>
                    <span className="text-[#a03200]">{formatStatValue(key, redVal)}</span>
                  </div>
                  <div className="h-3 w-full bg-[#e1e2ee] rounded-full flex overflow-hidden">
                    <div
                      className="h-full bg-[#004ecb] transition-all"
                      style={{ width: `${blueWidth}%` }}
                    ></div>
                    <div
                      className="h-full bg-[#a03200]/40 transition-all"
                      style={{ width: `${redWidth}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* CTA */}
        <section className="mt-20 flex flex-col items-center">
          <div className="text-center mb-8">
            <h3 className="text-4xl font-extrabold mb-4">지금 바로 예측에 참여하세요!</h3>
            <p className="text-[#424656] max-w-xl mx-auto">
              다른 팬들과 함께 이 경기를 예측해보세요.
            </p>
          </div>
          <div className="flex gap-4">
            <button className="px-12 py-5 rounded-full bg-[#004ecb] text-white font-extrabold text-lg shadow-xl shadow-blue-500/20 hover:scale-105 active:scale-95 transition-all">
              지금 예측하기
            </button>
            <a
              href="/"
              className="px-8 py-5 rounded-full bg-[#e6e7f4] text-[#191b24] font-bold text-lg hover:bg-[#d8d9e6] transition-all"
            >
              경기 목록으로
            </a>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-slate-50 py-12 mt-20 border-t border-slate-100">
        <div className="max-w-7xl mx-auto px-8 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-lg font-bold text-slate-900">LCK Predict</div>
          <div className="flex gap-8">
            <a className="text-sm text-slate-500 hover:text-blue-600" href="#">개인정보처리방침</a>
            <a className="text-sm text-slate-500 hover:text-blue-600" href="#">이용약관</a>
            <a className="text-sm text-slate-500 hover:text-blue-600" href="#">고객지원</a>
          </div>
          <p className="text-sm text-slate-500">© 2026 LCK Predict.</p>
        </div>
      </footer>
    </div>
  );
}
