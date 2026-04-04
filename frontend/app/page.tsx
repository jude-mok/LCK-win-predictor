"use client";

import { useEffect, useState } from "react";

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

export default function Home() {
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"upcoming" | "finished">("upcoming");
  const [predictions, setPredictions] = useState<Record<number, { blue: number; red: number }>>({});
  const [currentWeek, setCurrentWeek] = useState<number>(1);
  const [activeWeek, setActiveWeek] = useState<number>(1);
  const [weekMatches, setWeekMatches] = useState<Match[]>([]);
  const [weekLoading, setWeekLoading] = useState(false);

  // 현재 week 번호 가져오기
  useEffect(() => {
    fetch("https://lck-win-predictor-production.up.railway.app/schedule/")
      .then((res) => res.json())
      .then((data: Match[]) => {
        if (data.length > 0) {
          setCurrentWeek(data[0].week);
          setActiveWeek(data[0].week);
        }
        setLoading(false);
      });
  }, []);

  // week별 경기 데이터 + AI 예측
  useEffect(() => {
    if (currentWeek === 0) return;
    setWeekLoading(true);

    fetch(`https://lck-win-predictor-production.up.railway.app/schedule/entire/week/${currentWeek}`)
      .then((res) => res.json())
      .then(async (data: Match[]) => {
        setWeekMatches(data);
        setWeekLoading(false);

        const upcoming = data.filter((m) => m.winner === "NULL" || !m.winner);
        const results: Record<number, { blue: number; red: number }> = {};

        await Promise.all(
          upcoming.map(async (match) => {
            try {
              const res = await fetch("https://lck-win-predictor-production.up.railway.app/predict/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  blue_team: PREDICT_NAME_MAP[match.team_1],
                  red_team: PREDICT_NAME_MAP[match.team_2],
                }),
              });
              const pred = await res.json();
              results[match.id] = {
                blue: Math.round(pred.blue_win_rate * 100),
                red: Math.round(pred.red_win_rate * 100),
              };
            } catch (e) {
              console.error(e);
            }
          })
        );

        setPredictions((prev) => ({ ...prev, ...results }));
      });
  }, [currentWeek]);

  // finished 탭 AI 예측 lazy loading
  useEffect(() => {
    if (filter !== "finished") return;

    const finished = weekMatches.filter((m) => m.winner && m.winner !== "NULL");
    const results: Record<number, { blue: number; red: number }> = { ...predictions };

    const toFetch = finished.filter((m) => !predictions[m.id]);
    if (toFetch.length === 0) return;

    Promise.all(
      toFetch.map(async (match) => {
        try {
          const res = await fetch("https://lck-win-predictor-production.up.railway.app/predict/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              blue_team: PREDICT_NAME_MAP[match.team_1],
              red_team: PREDICT_NAME_MAP[match.team_2],
            }),
          });
          const pred = await res.json();
          results[match.id] = {
            blue: Math.round(pred.blue_win_rate * 100),
            red: Math.round(pred.red_win_rate * 100),
          };
        } catch (e) {
          console.error(e);
        }
      })
    ).then(() => setPredictions({ ...results }));
  }, [filter, weekMatches.length]);

  const today = new Date().toISOString().split("T")[0];

  const filtered = weekMatches.filter((m) => {
    if (filter === "upcoming") return m.winner === "NULL" || !m.winner;
    return m.winner && m.winner !== "NULL";
  });

  const formatTime = (time: string) => time.slice(0, 5);
  const formatDate = (date: string) => {
    const isToday = date === today;
    if (isToday) return "오늘";
    const d = new Date(date);
    return `${d.getMonth() + 1}/${d.getDate()}`;
  };

  return (
    <div className="bg-[#faf8ff] min-h-screen text-[#191b24]">
      {/* Nav */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-xl shadow-sm">
        <div className="flex justify-between items-center h-20 px-8 max-w-[1440px] mx-auto">
          <a className="text-2xl font-black tracking-tighter text-blue-600" href="#">
            LCK Predict
          </a>
          <div className="flex items-center gap-4">
            <button className="p-2 rounded-full hover:bg-slate-100 transition-all">
              <span className="material-symbols-outlined text-slate-600">notifications</span>
            </button>
            <button className="p-2 rounded-full hover:bg-slate-100 transition-all">
              <span className="material-symbols-outlined text-slate-600">account_balance_wallet</span>
            </button>
          </div>
        </div>
      </nav>

      <main className="pt-32 pb-20 px-8 max-w-[1440px] mx-auto min-h-screen">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
          {/* Left */}
          <div className="lg:col-span-8 space-y-10">
            <header className="space-y-8">
              <h1 className="text-5xl font-extrabold tracking-tight">LCK 경기 예측</h1>

              {/* 예정 / 완료 탭 */}
              <div className="flex gap-2 p-1.5 bg-[#f2f3ff] w-fit rounded-full">
                <button
                  onClick={() => setFilter("upcoming")}
                  className={`px-8 py-3 rounded-full font-bold text-sm transition-all ${
                    filter === "upcoming"
                      ? "bg-white shadow-sm text-[#004ecb]"
                      : "text-[#424656] hover:bg-white/50"
                  }`}
                >
                  예정
                </button>
                <button
                  onClick={() => setFilter("finished")}
                  className={`px-8 py-3 rounded-full font-bold text-sm transition-all ${
                    filter === "finished"
                      ? "bg-white shadow-sm text-[#004ecb]"
                      : "text-[#424656] hover:bg-white/50"
                  }`}
                >
                  완료
                </button>
              </div>

              {/* Week 필터 */}
              <div className="flex items-center gap-3 overflow-x-auto pb-2">
                {Array.from({ length: 9 }, (_, i) => i + 1).map((week) => (
                  <button
                    key={week}
                    onClick={() => setCurrentWeek(week)}
                    disabled={
                      filter === "upcoming" ? week < activeWeek : week > activeWeek
                    }
                    className={`px-5 py-2 rounded-full font-bold text-sm whitespace-nowrap transition-all ${
                      (filter === "upcoming" ? week < activeWeek : week > activeWeek)
                        ? "bg-white text-[#c0c2d6] border border-[#e6e7f4] cursor-not-allowed opacity-50"
                        : currentWeek === week
                        ? "bg-[#004ecb] text-white shadow-md"
                        : "bg-white text-[#424656] border border-[#e6e7f4] hover:border-[#004ecb]"
                    }`}
                  >
                    Week {week}
                  </button>
                ))}
              </div>
            </header>

            {/* Match List */}
            <div className="space-y-6">
              {loading || weekLoading ? (
                <p className="text-[#424656]">불러오는 중...</p>
              ) : filtered.length === 0 ? (
                <p className="text-[#424656]">경기가 없습니다.</p>
              ) : (
                filtered.map((match) => (
                  <div
                    key={match.id}
                    className="bg-white p-8 rounded-2xl shadow-sm border border-white/50 hover:shadow-xl transition-all duration-500 group"
                    onClick={() => window.location.href = `/match/${match.id}`}
                  >
                    <div className="flex flex-col md:flex-row justify-between items-center gap-8">
                      {/* Teams */}
                      <div className="flex-1 flex items-center justify-between gap-4 w-full">
                        <div className="flex flex-col items-center gap-3 flex-1">
                          <div className="w-20 h-20 flex items-center justify-center group-hover:scale-105 transition-transform">
                            <img
                              src={`/logos/${match.team_1}.png`}
                              alt={match.team_1}
                              className="w-16 h-16 object-contain"
                            />
                          </div>
                          <span className="font-bold text-lg">
                            {TEAMS_MAP[match.team_1] || match.team_1}
                          </span>
                        </div>

                        <div className="flex flex-col items-center gap-1">
                          <span className="text-xs font-bold text-[#004ecb] bg-blue-50 px-3 py-1 rounded-full uppercase tracking-widest">
                            BO3
                          </span>
                          <span className="text-3xl font-black text-[#d8d9e6]">VS</span>
                          <span className="text-xs font-medium text-[#424656]">
                            {formatDate(match.date)} {formatTime(match.time)}
                          </span>
                        </div>

                        <div className="flex flex-col items-center gap-3 flex-1">
                          <div className="w-20 h-20 flex items-center justify-center group-hover:scale-105 transition-transform">
                            <img
                              src={`/logos/${match.team_2}.png`}
                              alt={match.team_2}
                              className="w-16 h-16 object-contain"
                            />
                          </div>
                          <span className="font-bold text-lg">
                            {TEAMS_MAP[match.team_2] || match.team_2}
                          </span>
                        </div>
                      </div>

                      <div className="w-px h-16 bg-[#ecedfa] hidden md:block"></div>

                      {/* 예측 / 우승팀 */}
                      <div className="flex flex-col items-center md:items-end gap-4 min-w-[200px]">
                        {match.winner && match.winner !== "NULL" ? (
                          <div className="text-right">
                            <p className="text-xs font-bold text-[#424656] mb-1 uppercase tracking-tighter">
                              우승팀
                            </p>
                            <p className="text-2xl font-black text-[#004ecb]">
                              {TEAMS_MAP[match.winner] || match.winner}
                            </p>
                          </div>
                        ) : (
                          <>
                            <div className="text-right">
                              <p className="text-xs font-bold text-[#424656] mb-1 uppercase tracking-tighter">
                                참여자
                              </p>
                              <p className="text-2xl font-black">
                                — <span className="text-sm font-medium text-slate-400">명</span>
                              </p>
                            </div>
                            <button className="w-full py-4 px-8 bg-gradient-to-br from-[#004ecb] to-[#0064ff] text-white font-bold rounded-full shadow-lg shadow-blue-500/20 active:scale-95 transition-all hover:brightness-110">
                              예측하기
                            </button>
                          </>
                        )}
                      </div>
                    </div>

                    {/* AI Prediction Bar */}
                    <div className="mt-8 pt-8 border-t border-[#ecedfa] space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs font-bold uppercase tracking-tight">
                          <span className="text-[#424656] flex items-center gap-1.5">
                            <span className="material-symbols-outlined text-[14px]">smart_toy</span>
                            AI 모델 예측
                          </span>
                          <div className="flex gap-4">
                            <span className="text-[#004ecb]">
                              {match.team_1} {predictions[match.id]?.blue ?? "—"}%
                            </span>
                            <span className="text-[#a03200]">
                              {match.team_2} {predictions[match.id]?.red ?? "—"}%
                            </span>
                          </div>
                        </div>
                        <div className="h-1.5 w-full bg-[#e6e7f4] rounded-full overflow-hidden flex">
                          <div
                            className="h-full bg-[#004ecb]/60 transition-all duration-700"
                            style={{ width: `${predictions[match.id]?.blue ?? 50}%` }}
                          ></div>
                          <div
                            className="h-full bg-[#a03200]/10 transition-all duration-700"
                            style={{ width: `${predictions[match.id]?.red ?? 50}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Sidebar */}
          <aside className="lg:col-span-4 space-y-8">
            {/* LP Card */}
            <div className="bg-gradient-to-br from-[#004ecb] to-[#0064ff] p-8 rounded-2xl text-white shadow-2xl shadow-blue-600/30">
              <div className="flex justify-between items-start mb-10">
                <div>
                  <p className="text-sm font-medium opacity-80 mb-1">내 예측 포인트</p>
                  <h3 className="text-4xl font-black tracking-tight">
                    0 <span className="text-lg opacity-70">LP</span>
                  </h3>
                </div>
                <span className="material-symbols-outlined text-4xl opacity-50">
                  account_balance_wallet
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <button className="py-3 bg-white/20 backdrop-blur-md rounded-full font-bold text-sm hover:bg-white/30 transition-all">
                  내역
                </button>
                <button className="py-3 bg-white text-[#004ecb] rounded-full font-bold text-sm hover:bg-slate-50 transition-all">
                  상점
                </button>
              </div>
            </div>

            {/* Leaderboard */}
            <div className="bg-[#f2f3ff] p-8 rounded-2xl space-y-6">
              <div className="flex justify-between items-center">
                <h4 className="text-lg font-extrabold">예측 랭킹</h4>
                <a className="text-xs font-bold text-[#004ecb] hover:underline" href="#">
                  전체 보기
                </a>
              </div>
              <div className="space-y-4">
                {[
                  { rank: 1, name: "test_user1", rate: "89%", pts: "2.4M", color: "bg-yellow-400" },
                  { rank: 2, name: "test_user2", rate: "82%", pts: "1.9M", color: "bg-slate-300" },
                  { rank: 3, name: "test_user3", rate: "78%", pts: "1.5M", color: "bg-orange-300" },
                ].map((user) => (
                  <div
                    key={user.rank}
                    className="flex items-center gap-4 bg-white p-3 rounded-full shadow-sm"
                  >
                    <div
                      className={`w-10 h-10 rounded-full ${user.color} text-white flex items-center justify-center font-bold text-sm`}
                    >
                      {user.rank}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-bold truncate">{user.name}</p>
                      <p className="text-[10px] text-slate-400 font-bold uppercase">
                        {user.rate} 승률
                      </p>
                    </div>
                    <p className="font-black text-[#004ecb] text-sm pr-2">{user.pts}</p>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        </div>
      </main>

      {/* Footer */}
      <footer className="w-full py-12 border-t border-slate-100 bg-slate-50">
        <div className="flex flex-col md:flex-row justify-between items-center px-12 gap-6 max-w-[1440px] mx-auto">
          <div className="flex flex-col gap-2">
            <span className="font-bold text-slate-900">LCK Predict</span>
            <p className="text-sm text-slate-500">© 2026 LCK Predict.</p>
          </div>
          <div className="flex gap-8">
            <a className="text-slate-500 hover:text-blue-500 text-sm" href="#">이용약관</a>
            <a className="text-slate-500 hover:text-blue-500 text-sm" href="#">개인정보처리방침</a>
            <a className="text-slate-500 hover:text-blue-500 text-sm" href="#">고객지원</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
