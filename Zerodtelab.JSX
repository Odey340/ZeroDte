import React, { useState, useMemo } from "react";
import {
  ComposedChart, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell, Area,
} from "recharts";

/* ============================================================
   CORE QUANT MATH  (verified against textbook values)
   Black–Scholes pricing + greeks, normal CDF/PDF, Monte Carlo
   ============================================================ */
const SQRT2PI = Math.sqrt(2 * Math.PI);
const pdf = (x) => Math.exp(-0.5 * x * x) / SQRT2PI;
function erf(x) {
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return s * y;
}
const cdf = (x) => 0.5 * (1 + erf(x / Math.SQRT2));

// T in YEARS. Trading-time convention: 6.5h session, 252 days/yr.
const hoursToYears = (h) => (h / 6.5) / 252;

function bs(S, K, T, r, sigma, type) {
  if (T <= 0 || sigma <= 0) {
    const intr = type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    return { price: intr, delta: 0, gamma: 0, theta: 0, vega: 0 };
  }
  const sqT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqT);
  const d2 = d1 - sigma * sqT;
  const Nd1 = cdf(d1), Nd2 = cdf(d2);
  const price = type === "call"
    ? S * Nd1 - K * Math.exp(-r * T) * Nd2
    : K * Math.exp(-r * T) * cdf(-d2) - S * cdf(-d1);
  const gamma = pdf(d1) / (S * sigma * sqT);
  const vega = (S * pdf(d1) * sqT) / 100;          // per 1 vol-point
  const thetaYr = type === "call"
    ? -(S * pdf(d1) * sigma) / (2 * sqT) - r * K * Math.exp(-r * T) * Nd2
    : -(S * pdf(d1) * sigma) / (2 * sqT) + r * K * Math.exp(-r * T) * cdf(-d2);
  const delta = type === "call" ? Nd1 : Nd1 - 1;
  return { price, delta, gamma, theta: thetaYr / 252, vega }; // theta per trading day
}

// Box–Muller standard normal
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const intrinsic = (type, S, K) => type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);

/* Strategy -> array of legs {type, strike, qty(+long/-short)} */
function buildLegs(strategy, S0, sigIV, T, nSigma, wing) {
  const off = nSigma * sigIV * S0 * Math.sqrt(T);
  const Kc = S0 + off, Kp = S0 - off, K0 = S0;
  switch (strategy) {
    case "short_strangle":
      return [{ type: "put", strike: Kp, qty: -1 }, { type: "call", strike: Kc, qty: -1 }];
    case "short_straddle":
      return [{ type: "put", strike: K0, qty: -1 }, { type: "call", strike: K0, qty: -1 }];
    case "long_straddle":
      return [{ type: "put", strike: K0, qty: 1 }, { type: "call", strike: K0, qty: 1 }];
    case "iron_condor":
      return [
        { type: "put", strike: Kp - wing, qty: 1 }, { type: "put", strike: Kp, qty: -1 },
        { type: "call", strike: Kc, qty: -1 }, { type: "call", strike: Kc + wing, qty: 1 },
      ];
    case "credit_put_spread":
      return [{ type: "put", strike: Kp - wing, qty: 1 }, { type: "put", strike: Kp, qty: -1 }];
    default:
      return [];
  }
}

/* ============================================================
   PALETTE / TYPOGRAPHY  — "quant research terminal"
   ============================================================ */
const C = {
  bg: "#0a0d12", panel: "#10151c", panel2: "#0c1116", line: "#1c242e",
  lineSoft: "#161d26", text: "#e6edf3", mut: "#7d8b99", mut2: "#566370",
  amber: "#f0b429", amberDim: "#8a6a1f", cyan: "#38bdf8", green: "#3fe07f",
  red: "#f0556b", redDim: "#5e2630", violet: "#a78bfa",
};
const MONO = "'IBM Plex Mono', ui-monospace, monospace";
const SERIF = "'Fraunces', Georgia, serif";

/* small reusable bits ------------------------------------- */
function Stat({ label, value, sub, color }) {
  return (
    <div style={{
      background: C.panel2, border: `1px solid ${C.line}`, borderRadius: 4,
      padding: "10px 12px", display: "flex", flexDirection: "column", gap: 3, minWidth: 0,
    }}>
      <div style={{ font: `500 9.5px ${MONO}`, letterSpacing: ".11em",
        textTransform: "uppercase", color: C.mut }}>{label}</div>
      <div style={{ font: `600 19px ${MONO}`, color: color || C.text, lineHeight: 1,
        overflow: "hidden", textOverflow: "ellipsis" }}>{value}</div>
      {sub && <div style={{ font: `400 10px ${MONO}`, color: C.mut2 }}>{sub}</div>}
    </div>
  );
}

function Slider({ label, value, set, min, max, step, fmt, accent }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <span style={{ font: `500 10px ${MONO}`, letterSpacing: ".08em",
          textTransform: "uppercase", color: C.mut }}>{label}</span>
        <span style={{ font: `600 12px ${MONO}`, color: accent || C.text }}>
          {fmt ? fmt(value) : value}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => set(parseFloat(e.target.value))}
        style={{ accentColor: accent || C.amber, height: 3, cursor: "pointer" }} />
    </div>
  );
}

function Seg({ options, value, set }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
      {options.map((o) => {
        const on = o.v === value;
        return (
          <button key={o.v} onClick={() => set(o.v)} style={{
            font: `500 10.5px ${MONO}`, letterSpacing: ".04em", padding: "6px 10px",
            borderRadius: 3, cursor: "pointer", whiteSpace: "nowrap",
            border: `1px solid ${on ? C.amber : C.line}`,
            background: on ? "rgba(240,180,41,0.12)" : "transparent",
            color: on ? C.amber : C.mut, transition: "all .12s",
          }}>{o.l}</button>
        );
      })}
    </div>
  );
}

const tipStyle = {
  background: "#0b0f14", border: `1px solid ${C.line}`, borderRadius: 4,
  font: `400 11px ${MONO}`, color: C.text, padding: "6px 8px",
};

/* ============================================================
   GREEKS LAB
   ============================================================ */
function GreeksLab() {
  const [S0] = useState(5000);
  const [K, setK] = useState(5000);
  const [iv, setIv] = useState(15);
  const [hrs, setHrs] = useState(6.5);
  const r = 0.045;
  const T = hoursToYears(hrs);
  const sigma = iv / 100;

  const curve = useMemo(() => {
    const out = [];
    const lo = S0 * 0.97, hi = S0 * 1.03;
    for (let i = 0; i <= 120; i++) {
      const S = lo + (hi - lo) * (i / 120);
      const g = bs(S, K, T, r, sigma, "call");
      out.push({ S: +S.toFixed(1), gamma: g.gamma, theta: g.theta, delta: g.delta });
    }
    return out;
  }, [K, T, sigma]);

  const blowup = useMemo(() => {
    const out = [];
    for (let h = 6.5; h >= 0.05; h -= 0.05) {
      const g = bs(S0, S0, hoursToYears(h), r, sigma, "call");
      out.push({ h: +h.toFixed(2), gamma: g.gamma });
    }
    return out;
  }, [sigma]);

  const atm = bs(S0, K, T, r, sigma, "call");

  return (
    <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 16 }}>
      {/* controls */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Slider label="Strike K" value={K} set={setK} min={4850} max={5150} step={5}
          accent={C.cyan} fmt={(v) => v.toFixed(0)} />
        <Slider label="Implied vol" value={iv} set={setIv} min={5} max={60} step={0.5}
          accent={C.violet} fmt={(v) => v.toFixed(1) + "%"} />
        <Slider label="Hours to expiry" value={hrs} set={setHrs} min={0.1} max={6.5} step={0.1}
          accent={C.amber} fmt={(v) => v.toFixed(1) + "h"} />
        <div style={{ height: 1, background: C.line }} />
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <Stat label="ATM Γ" value={atm.gamma.toExponential(2)} color={C.amber} />
          <Stat label="Θ /day" value={atm.theta.toFixed(2)} color={C.red} />
          <Stat label="Δ" value={bs(S0, K, T, r, sigma, "call").delta.toFixed(3)} color={C.cyan} />
          <Stat label="Vega" value={atm.vega.toFixed(3)} color={C.violet} />
        </div>
        <p style={{ font: `400 11px ${MONO}`, color: C.mut, lineHeight: 1.6, margin: 0 }}>
          Drag <span style={{ color: C.amber }}>hours → 0</span> and watch ATM gamma spike
          while theta decay accelerates. That convexity is the entire 0DTE risk: sellers
          harvest theta only by being short exploding gamma.
        </p>
      </div>

      {/* charts */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ background: C.panel2, border: `1px solid ${C.line}`, borderRadius: 6, padding: "12px 14px 6px" }}>
          <ChartTitle>Γ &amp; Θ vs spot &nbsp;·&nbsp; call, T = {hrs.toFixed(1)}h</ChartTitle>
          <ResponsiveContainer width="100%" height={210}>
            <ComposedChart data={curve} margin={{ top: 6, right: 8, left: -6, bottom: 0 }}>
              <CartesianGrid stroke={C.lineSoft} vertical={false} />
              <XAxis dataKey="S" tick={{ fill: C.mut2, fontSize: 10, fontFamily: MONO }}
                stroke={C.line} tickFormatter={(v) => v.toFixed(0)} minTickGap={40} />
              <YAxis yAxisId="g" tick={{ fill: C.amber, fontSize: 10, fontFamily: MONO }}
                stroke={C.line} tickFormatter={(v) => v.toExponential(0)} width={48} />
              <YAxis yAxisId="t" orientation="right" tick={{ fill: C.red, fontSize: 10, fontFamily: MONO }}
                stroke={C.line} width={40} />
              <Tooltip contentStyle={tipStyle} labelStyle={{ color: C.mut }}
                formatter={(v, n) => [typeof v === "number" ? v.toExponential(3) : v, n]} />
              <ReferenceLine yAxisId="g" x={K} stroke={C.cyan} strokeDasharray="3 3"
                label={{ value: "K", fill: C.cyan, fontSize: 10, fontFamily: MONO }} />
              <Line yAxisId="g" dataKey="gamma" name="Γ" stroke={C.amber} dot={false} strokeWidth={2} />
              <Line yAxisId="t" dataKey="theta" name="Θ/day" stroke={C.red} dot={false} strokeWidth={1.6} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: C.panel2, border: `1px solid ${C.line}`, borderRadius: 6, padding: "12px 14px 6px" }}>
          <ChartTitle>ATM Γ explosion as expiry approaches &nbsp;·&nbsp; the 1/√T blowup</ChartTitle>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={blowup} margin={{ top: 6, right: 12, left: -6, bottom: 0 }}>
              <defs>
                <linearGradient id="gam" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor={C.amberDim} /><stop offset="100%" stopColor={C.amber} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={C.lineSoft} vertical={false} />
              <XAxis dataKey="h" reversed tick={{ fill: C.mut2, fontSize: 10, fontFamily: MONO }}
                stroke={C.line} tickFormatter={(v) => v + "h"} />
              <YAxis tick={{ fill: C.amber, fontSize: 10, fontFamily: MONO }} stroke={C.line}
                tickFormatter={(v) => v.toExponential(0)} width={48} />
              <Tooltip contentStyle={tipStyle} labelStyle={{ color: C.mut }}
                formatter={(v) => [v.toExponential(3), "Γ"]} labelFormatter={(l) => l + "h to expiry"} />
              <Line dataKey="gamma" stroke="url(#gam)" dot={false} strokeWidth={2.4} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   STRATEGY / EXPECTANCY LAB
   ============================================================ */
const STRATS = [
  { v: "short_strangle", l: "Short strangle" },
  { v: "iron_condor", l: "Iron condor" },
  { v: "credit_put_spread", l: "Credit put spread" },
  { v: "short_straddle", l: "Short straddle" },
  { v: "long_straddle", l: "Long straddle" },
];

function StrategyLab() {
  const S0 = 5000, r = 0.045, mult = 100;
  const [strategy, setStrategy] = useState("short_strangle");
  const [iv, setIv] = useState(18);
  const [rv, setRv] = useState(15);
  const [hrs, setHrs] = useState(6.5);
  const [nSigma, setNSigma] = useState(1.0);
  const [wing, setWing] = useState(25);
  const [cost, setCost] = useState(0.5);
  const [nSims, setNSims] = useState(25000);

  const T = hoursToYears(hrs);
  const sigIV = iv / 100, sigRV = rv / 100;
  const usesWing = strategy === "iron_condor" || strategy === "credit_put_spread";

  const model = useMemo(() => {
    const legs = buildLegs(strategy, S0, sigIV, T, nSigma, wing).map((lg) => ({
      ...lg, entry: bs(S0, lg.strike, T, r, sigIV, lg.type).price,
    }));
    const nLegs = legs.length;
    const costTot = nLegs * cost;
    const netCredit = -legs.reduce((s, lg) => s + lg.qty * lg.entry, 0);

    const pnlAt = (ST) =>
      legs.reduce((s, lg) => s + lg.qty * (intrinsic(lg.type, ST, lg.strike) - lg.entry), 0) - costTot;

    // deterministic payoff curve
    const payoff = [];
    const lo = S0 * 0.94, hi = S0 * 1.06;
    for (let i = 0; i <= 160; i++) {
      const ST = lo + (hi - lo) * (i / 160);
      payoff.push({ ST: +ST.toFixed(1), pnl: +(pnlAt(ST) * mult).toFixed(2) });
    }

    // monte carlo
    const pnls = new Float64Array(nSims);
    let wins = 0, sum = 0, sumsq = 0, maxL = Infinity, maxG = -Infinity;
    let sumWin = 0, nWin = 0, sumLoss = 0, nLoss = 0;
    for (let i = 0; i < nSims; i++) {
      const ST = S0 * Math.exp((r - 0.5 * sigRV * sigRV) * T + sigRV * Math.sqrt(T) * randn());
      const pnl = pnlAt(ST) * mult;
      pnls[i] = pnl; sum += pnl; sumsq += pnl * pnl;
      if (pnl > 0) { wins++; sumWin += pnl; nWin++; } else { sumLoss += pnl; nLoss++; }
      if (pnl < maxL) maxL = pnl;
      if (pnl > maxG) maxG = pnl;
    }
    const mean = sum / nSims;
    const variance = sumsq / nSims - mean * mean;
    const std = Math.sqrt(Math.max(variance, 0));
    const sorted = Array.from(pnls).sort((a, b) => a - b);
    const tailN = Math.max(1, Math.floor(nSims * 0.05));
    const cvar5 = sorted.slice(0, tailN).reduce((a, b) => a + b, 0) / tailN;
    const median = sorted[Math.floor(nSims / 2)];
    const avgWin = nWin ? sumWin / nWin : 0;
    const avgLoss = nLoss ? sumLoss / nLoss : 0;
    const kelly = variance > 0 ? mean / variance : 0; // continuous-approx, per $ var

    // histogram
    const bins = 45;
    const bMin = sorted[0], bMax = sorted[sorted.length - 1];
    const span = (bMax - bMin) || 1;
    const hist = Array.from({ length: bins }, (_, i) => ({
      x: bMin + span * ((i + 0.5) / bins), count: 0,
      mid: bMin + span * ((i + 0.5) / bins),
    }));
    for (let i = 0; i < nSims; i++) {
      let idx = Math.floor(((pnls[i] - bMin) / span) * bins);
      if (idx >= bins) idx = bins - 1; if (idx < 0) idx = 0;
      hist[idx].count++;
    }

    return {
      legs, netCredit, winRate: wins / nSims, mean, median, std, cvar5,
      maxL, maxG, avgWin, avgLoss, kelly, payoff, hist,
    };
  }, [strategy, sigIV, sigRV, T, nSigma, wing, cost, nSims]);

  const trap = model.winRate >= 0.6 && model.mean <= 0;
  const edge = model.mean > 0;
  const expColor = model.mean > 0 ? C.green : C.red;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* banner */}
      <div style={{
        border: `1px solid ${trap ? C.red : edge ? C.green : C.line}`,
        background: trap ? "rgba(240,85,107,0.08)" : edge ? "rgba(63,224,127,0.07)" : C.panel2,
        borderRadius: 6, padding: "11px 14px", display: "flex", gap: 12, alignItems: "center",
      }}>
        <div style={{ font: `900 22px ${MONO}`, color: trap ? C.red : edge ? C.green : C.mut, lineHeight: 1 }}>
          {trap ? "⚠" : edge ? "✓" : "·"}
        </div>
        <div style={{ font: `400 12px ${MONO}`, color: C.text, lineHeight: 1.5 }}>
          {trap ? (
            <><b style={{ color: C.red }}>The trap.</b> Win rate{" "}
              {(model.winRate * 100).toFixed(0)}% but expectancy is <b style={{ color: C.red }}>negative</b>.
              You win most days and still bleed — the losses are bigger than the wins. This is what
              "win more than I lose" hides.</>
          ) : edge ? (
            <><b style={{ color: C.green }}>Positive edge.</b> Expectancy{" "}
              ${model.mean.toFixed(2)}/contract. Note it survives only while IV &gt; realized
              (the variance risk premium) — and inspect the left tail before sizing.</>
          ) : (
            <>Set the parameters and read expectancy, not win rate. Push realized vol above implied
              to see a high win rate flip to a losing strategy.</>
          )}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "264px 1fr", gap: 16 }}>
        {/* controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div>
            <Label>Strategy</Label>
            <Seg options={STRATS} value={strategy} set={setStrategy} />
          </div>
          <Slider label="Implied vol (priced in)" value={iv} set={setIv} min={5} max={50} step={0.5}
            accent={C.amber} fmt={(v) => v.toFixed(1) + "%"} />
          <Slider label="Realized vol (the market)" value={rv} set={setRv} min={5} max={50} step={0.5}
            accent={C.cyan} fmt={(v) => v.toFixed(1) + "%"} />
          <Slider label="Hours to expiry" value={hrs} set={setHrs} min={0.5} max={6.5} step={0.5}
            accent={C.violet} fmt={(v) => v.toFixed(1) + "h"} />
          <Slider label="Strike distance" value={nSigma} set={setNSigma} min={0.25} max={2.5} step={0.05}
            accent={C.green} fmt={(v) => v.toFixed(2) + "σ"} />
          {usesWing &&
            <Slider label="Wing width" value={wing} set={setWing} min={5} max={100} step={5}
              accent={C.mut} fmt={(v) => v.toFixed(0) + " pt"} />}
          <Slider label="Cost / leg (spread)" value={cost} set={setCost} min={0} max={3} step={0.1}
            accent={C.red} fmt={(v) => "$" + v.toFixed(1)} />
          <div>
            <Label>Simulations</Label>
            <Seg value={nSims} set={setNSims} options={[
              { v: 10000, l: "10k" }, { v: 25000, l: "25k" }, { v: 50000, l: "50k" }, { v: 100000, l: "100k" },
            ]} />
          </div>
          <div style={{ font: `400 10px ${MONO}`, color: C.mut2, lineHeight: 1.6 }}>
            GBM terminal price · ${mult}/pt multiplier · premium priced at IV, path drawn at
            realized vol. The gap between them is your only structural edge.
          </div>
        </div>

        {/* right: stats + charts */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
            <Stat label="Win rate" value={(model.winRate * 100).toFixed(1) + "%"} color={C.cyan}
              sub="% days green" />
            <Stat label="Expectancy" value={"$" + model.mean.toFixed(2)} color={expColor}
              sub="per contract" />
            <Stat label="Avg win" value={"$" + model.avgWin.toFixed(0)} color={C.green} />
            <Stat label="Avg loss" value={"$" + model.avgLoss.toFixed(0)} color={C.red} />
            <Stat label="Max loss" value={"$" + model.maxL.toFixed(0)} color={C.red}
              sub="worst sim" />
            <Stat label="CVaR 5%" value={"$" + model.cvar5.toFixed(0)} color={C.red}
              sub="avg of worst 5%" />
            <Stat label="Net credit" value={"$" + (model.netCredit * mult).toFixed(0)} color={C.amber} />
            <Stat label="Kelly f*" value={(model.kelly * 1000).toFixed(2) + "‰"} color={C.violet}
              sub="≈μ/σ² — quarter it" />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            <div style={{ background: C.panel2, border: `1px solid ${C.line}`, borderRadius: 6, padding: "12px 14px 6px" }}>
              <ChartTitle>Payoff at expiry &nbsp;·&nbsp; $/contract</ChartTitle>
              <ResponsiveContainer width="100%" height={190}>
                <ComposedChart data={model.payoff} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
                  <CartesianGrid stroke={C.lineSoft} vertical={false} />
                  <XAxis dataKey="ST" tick={{ fill: C.mut2, fontSize: 10, fontFamily: MONO }}
                    stroke={C.line} tickFormatter={(v) => v.toFixed(0)} minTickGap={36} />
                  <YAxis tick={{ fill: C.mut2, fontSize: 10, fontFamily: MONO }} stroke={C.line} width={44} />
                  <Tooltip contentStyle={tipStyle} labelStyle={{ color: C.mut }}
                    formatter={(v) => ["$" + v, "P&L"]} labelFormatter={(l) => "S_T " + l} />
                  <ReferenceLine y={0} stroke={C.mut2} />
                  <ReferenceLine x={S0} stroke={C.cyan} strokeDasharray="3 3" />
                  <Line dataKey="pnl" stroke={C.amber} dot={false} strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            <div style={{ background: C.panel2, border: `1px solid ${C.line}`, borderRadius: 6, padding: "12px 14px 6px" }}>
              <ChartTitle>P&amp;L distribution &nbsp;·&nbsp; {(nSims / 1000) | 0}k sims</ChartTitle>
              <ResponsiveContainer width="100%" height={190}>
                <BarChart data={model.hist} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
                  <CartesianGrid stroke={C.lineSoft} vertical={false} />
                  <XAxis dataKey="mid" tick={{ fill: C.mut2, fontSize: 10, fontFamily: MONO }}
                    stroke={C.line} tickFormatter={(v) => Math.round(v)} minTickGap={32} />
                  <YAxis tick={{ fill: C.mut2, fontSize: 10, fontFamily: MONO }} stroke={C.line} width={32}
                    tickFormatter={(v) => v >= 1000 ? (v / 1000).toFixed(0) + "k" : v} />
                  <Tooltip contentStyle={tipStyle} labelStyle={{ color: C.mut }}
                    formatter={(v) => [v, "count"]} labelFormatter={(l) => "P&L ≈ $" + Math.round(l)} />
                  <ReferenceLine x={0} stroke={C.text} strokeDasharray="2 2" />
                  <ReferenceLine x={model.mean} stroke={C.amber} strokeWidth={1.4}
                    label={{ value: "μ", fill: C.amber, fontSize: 11, fontFamily: MONO, position: "top" }} />
                  <Bar dataKey="count">
                    {model.hist.map((b, i) => (
                      <Cell key={i} fill={b.mid >= 0 ? C.green : C.red} fillOpacity={0.82} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChartTitle({ children }) {
  return <div style={{ font: `500 10px ${MONO}`, letterSpacing: ".1em", textTransform: "uppercase",
    color: C.mut, marginBottom: 4 }}>{children}</div>;
}
function Label({ children }) {
  return <div style={{ font: `500 10px ${MONO}`, letterSpacing: ".08em", textTransform: "uppercase",
    color: C.mut, marginBottom: 7 }}>{children}</div>;
}

/* ============================================================
   SHELL
   ============================================================ */
export default function ZeroDTELab() {
  const [tab, setTab] = useState("greeks");
  return (
    <div style={{ background: C.bg, color: C.text, minHeight: "100%", padding: "22px 24px 30px",
      fontFamily: MONO }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,900&display=swap');
        @keyframes rise { from { opacity:0; transform:translateY(8px);} to {opacity:1; transform:none;} }
        input[type=range]{ -webkit-appearance:none; appearance:none; background:#1c242e; border-radius:99px; }
        input[type=range]::-webkit-slider-thumb{ -webkit-appearance:none; width:13px; height:13px;
          border-radius:50%; background:currentColor; cursor:pointer; }
        ::-webkit-scrollbar{ width:9px; height:9px; } ::-webkit-scrollbar-thumb{ background:#222b35; border-radius:9px; }
      `}</style>

      <div style={{ maxWidth: 1080, margin: "0 auto", animation: "rise .5s ease both" }}>
        {/* masthead */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end",
          borderBottom: `1px solid ${C.line}`, paddingBottom: 14, marginBottom: 18, flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ font: `500 10px ${MONO}`, letterSpacing: ".3em", color: C.amber,
              textTransform: "uppercase", marginBottom: 6 }}>0DTE Research Desk</div>
            <h1 style={{ font: `900 32px ${SERIF}`, margin: 0, letterSpacing: "-.01em", lineHeight: 1 }}>
              Greeks &amp; Expectancy Lab
            </h1>
            <p style={{ font: `400 12px ${MONO}`, color: C.mut, margin: "8px 0 0", maxWidth: 560, lineHeight: 1.5 }}>
              Black–Scholes greeks under T→0 and a Monte-Carlo expectancy engine. The thesis:
              <span style={{ color: C.text }}> win rate and expectancy are nearly orthogonal.</span> Design for the second.
            </p>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {[{ v: "greeks", l: "Greeks Lab" }, { v: "strategy", l: "Expectancy Lab" }].map((t) => {
              const on = t.v === tab;
              return (
                <button key={t.v} onClick={() => setTab(t.v)} style={{
                  font: `600 11px ${MONO}`, letterSpacing: ".06em", textTransform: "uppercase",
                  padding: "9px 14px", borderRadius: 4, cursor: "pointer",
                  border: `1px solid ${on ? C.amber : C.line}`,
                  background: on ? "rgba(240,180,41,0.12)" : "transparent",
                  color: on ? C.amber : C.mut,
                }}>{t.l}</button>
              );
            })}
          </div>
        </div>

        {tab === "greeks" ? <GreeksLab /> : <StrategyLab />}

        <div style={{ marginTop: 22, paddingTop: 14, borderTop: `1px solid ${C.line}`,
          font: `400 10.5px ${MONO}`, color: C.mut2, lineHeight: 1.6 }}>
          Toy model: GBM terminal price, flat vol, European 0DTE, frictions as a per-leg spread cost.
          Real surfaces have skew, jumps, and fatter tails — so treat positive edge here as
          <span style={{ color: C.mut }}> necessary, never sufficient.</span> Not financial advice.
        </div>
      </div>
    </div>
  );
}
