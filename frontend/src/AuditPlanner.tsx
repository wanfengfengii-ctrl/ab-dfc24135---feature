import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  AuditPlanResponse,
  AuditRangeInput,
  createAuditPlan,
} from "./api";

interface RangeRow {
  start: string;
  end: string;
  quota: string;
}

interface Props {
  session: string;
  chunkCount: number;
  plan: AuditPlanResponse | null;
  onCommitted: () => void;
}

function parseScores(text: string): number[] | null {
  const parts = text
    .split(/[\s,，;；]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const out: number[] = [];
  for (const part of parts) {
    if (!/^(0|[1-9][0-9]*)$/.test(part)) return null;
    out.push(Number(part));
  }
  return out;
}

function parseIndex(text: string): number | null {
  if (!/^(0|[1-9][0-9]*)$/.test(text.trim())) return null;
  return Number(text.trim());
}

export default function AuditPlanner({ session, chunkCount, plan, onCommitted }: Props) {
  const [target, setTarget] = useState("2");
  const [scoresText, setScoresText] = useState("");
  const [rows, setRows] = useState<RangeRow[]>([
    { start: "0", end: "", quota: "1" },
  ]);
  const [result, setResult] = useState<AuditPlanResponse | null>(plan);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Restore the persisted form (and its result) at most once per receipt.
  const initializedFor = useRef<string | null>(null);

  useEffect(() => {
    const key = plan?.created_at ?? plan?.receipt_id ?? null;
    if (plan && plan.session === session && initializedFor.current !== key) {
      initializedFor.current = key;
      setTarget(String(plan.request?.target ?? plan.target));
      setScoresText((plan.request?.risk_scores ?? []).join(", "));
      const rs = plan.request?.ranges ?? plan.ranges.map((r) => ({
        start: r.start,
        end: r.end,
        quota: r.quota,
      }));
      setRows(rs.map((r) => ({ start: String(r.start), end: String(r.end), quota: String(r.quota) })));
      setResult(plan);
      setError(null);
    }
    if (!plan) initializedFor.current = null;
  }, [plan, session]);

  const parsed = useMemo(() => {
    const problems: string[] = [];
    const t = parseIndex(target);
    if (t === null) problems.push("目标抽检块数必须是整数");
    else if (t < 2 || t > 16) problems.push("目标抽检块数必须在 2–16 之间");
    else if (t > chunkCount) problems.push(`目标抽检块数不能超过总块数 ${chunkCount}`);

    const scores = parseScores(scoresText);
    if (scores === null) problems.push("风险分必须是 0–100 的整数（逗号、空格或换行分隔）");
    else if (scores.length !== chunkCount)
      problems.push(`风险分数量 ${scores.length} 与实际块数 ${chunkCount} 不一致`);
    else if (scores.some((s) => s < 0 || s > 100))
      problems.push("每个风险分必须在 0–100 之间");

    const ranges: AuditRangeInput[] = [];
    if (rows.length < 1 || rows.length > 4) {
      problems.push("重点区间必须为 1–4 个");
    }
    rows.forEach((row, i) => {
      const s = parseIndex(row.start);
      const e = parseIndex(row.end);
      const q = parseIndex(row.quota);
      if (s === null || e === null || q === null) {
        problems.push(`区间 ${i + 1} 的起止块号与配额必须是非负整数`);
        return;
      }
      if (e < s) problems.push(`区间 ${i + 1}：起点 ${s} 不得大于终点 ${e}`);
      if (e >= chunkCount) problems.push(`区间 ${i + 1}：终点 ${e} 超出块号范围（0–${chunkCount - 1}）`);
      if (q > e - s + 1) problems.push(`区间 ${i + 1}：配额 ${q} 超过区间长度 ${e - s + 1}`);
      ranges.push({ start: s, end: e, quota: q });
    });
    const ordered = [...ranges].sort((a, b) => a.start - b.start);
    for (let i = 1; i < ordered.length; i++) {
      if (ordered[i].start <= ordered[i - 1].end) {
        problems.push(
          `重点区间不得重叠：[${ordered[i - 1].start},${ordered[i - 1].end}] 与 [${ordered[i].start},${ordered[i].end}]`
        );
        break;
      }
    }
    return { problems, target: t, scores, ranges };
  }, [target, scoresText, rows, chunkCount]);

  // Any edit invalidates the displayed result: stale plans never linger.
  const touch = () => {
    setResult(null);
    setError(null);
  };

  const updateRow = (i: number, field: keyof RangeRow, value: string) => {
    touch();
    setRows((prev) => prev.map((r, j) => (j === i ? { ...r, [field]: value } : r)));
  };

  const submit = async () => {
    if (parsed.problems.length > 0 || parsed.target === null || parsed.scores === null) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const out = await createAuditPlan(session, {
        target: parsed.target,
        risk_scores: parsed.scores,
        ranges: parsed.ranges,
      });
      setResult(out);
      onCommitted();
    } catch (e) {
      const err = e as ApiError;
      setError(err.message || `${err.status} 请求被拒绝`);
    } finally {
      setBusy(false);
    }
  };

  const selected = new Set(result?.solvable ? result.blocks : []);
  const riskOf = result?.request?.risk_scores ?? parsed.scores ?? [];
  const rangeOf = (b: number): AuditPlanResponse["ranges"][number] | null =>
    result ? result.ranges.find((r) => r.start <= b && b <= r.end) ?? null : null;

  return (
    <div className="audit">
      <h3>抽检计划（仅已封存会话可生成）</h3>

      <label className="field">
        <span>目标抽检块数（2–16，且不超过总块数 {chunkCount}）</span>
        <input
          type="number"
          min={2}
          max={Math.min(16, chunkCount)}
          value={target}
          onChange={(e) => {
            touch();
            setTarget(e.target.value);
          }}
          disabled={busy}
        />
      </label>

      <label className="field">
        <span>逐块风险分（{chunkCount} 个，0–100，逗号/空格/换行分隔）</span>
        <textarea
          rows={3}
          placeholder={`10, 80, 0, 55 … 共 ${chunkCount} 个 0–100 的整数`}
          value={scoresText}
          onChange={(e) => {
            touch();
            setScoresText(e.target.value);
          }}
          disabled={busy}
        />
      </label>

      <div className="field">
        <span>重点区间（1–4 个互不重叠的连续闭区间，各设最低抽检数）</span>
        {rows.map((row, i) => (
          <div className="range-row" key={i}>
            <span className="range-tag">区间 {i + 1}</span>
            <input
              type="number"
              min={0}
              placeholder="起点"
              value={row.start}
              onChange={(e) => updateRow(i, "start", e.target.value)}
              disabled={busy}
            />
            <em>–</em>
            <input
              type="number"
              min={0}
              placeholder="终点"
              value={row.end}
              onChange={(e) => updateRow(i, "end", e.target.value)}
              disabled={busy}
            />
            <em>配额≥</em>
            <input
              type="number"
              min={0}
              placeholder="最低抽检数"
              value={row.quota}
              onChange={(e) => updateRow(i, "quota", e.target.value)}
              disabled={busy}
            />
            <button
              type="button"
              onClick={() => {
                touch();
                setRows((prev) => prev.filter((_, j) => j !== i));
              }}
              disabled={busy || rows.length <= 1}
              title="删除该区间"
            >
              删除
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => {
            touch();
            setRows((prev) =>
              prev.length < 4 ? [...prev, { start: "", end: "", quota: "1" }] : prev
            );
          }}
          disabled={busy || rows.length >= 4}
        >
          增加区间
        </button>
      </div>

      {parsed.problems.length > 0 && (
        <ul className="audit-errors">
          {parsed.problems.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ul>
      )}

      <div className="row">
        <button
          type="button"
          className="primary"
          onClick={() => void submit()}
          disabled={busy || parsed.problems.length > 0}
        >
          {busy ? "正在生成…" : "生成抽检计划"}
        </button>
      </div>

      {error && (
        <div className="notice bad-notice">
          请求被定位拒绝，未产生计划：{error}
        </div>
      )}

      {result && !result.solvable && (
        <div className="notice blocked">
          <strong>无解 · 阻断条件</strong>
          <div>{result.block_reason}</div>
          <div>未伪造任何抽检计划，也未保留旧计划。</div>
        </div>
      )}

      {result?.solvable && (
        <div className="plan-result">
          <div className="plan-summary">
            <div>
              抽中块号：
              <strong>[{result.blocks.join(", ")}]</strong>
            </div>
            <div>
              共 {result.blocks.length} 块 · 风险分总和 <strong>{result.risk_sum}</strong>
            </div>
          </div>

          <table className="quota-table">
            <thead>
              <tr>
                <th>重点区间</th>
                <th>最低抽检数</th>
                <th>实际抽中</th>
                <th>核算</th>
              </tr>
            </thead>
            <tbody>
              {result.ranges.map((r, i) => (
                <tr key={i}>
                  <td>[{r.start}, {r.end}]</td>
                  <td>{r.quota}</td>
                  <td>{r.selected}</td>
                  <td>
                    {r.selected >= r.quota ? (
                      <span className="good">达标（{r.selected}≥{r.quota}）</span>
                    ) : (
                      <span className="bad">不足（{r.selected}&lt;{r.quota}）</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="block-table-wrap">
            <table className="block-table">
              <thead>
                <tr>
                  <th>块号</th>
                  <th>风险分</th>
                  <th>所属重点区间</th>
                  <th>是否抽中</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: result.chunk_count }, (_, b) => {
                  const r = rangeOf(b);
                  return (
                    <tr key={b} className={selected.has(b) ? "picked" : ""}>
                      <td>{b}</td>
                      <td>{riskOf[b] ?? "—"}</td>
                      <td>{r ? `[${r.start}, ${r.end}]（配额 ${r.quota}）` : "—"}</td>
                      <td>{selected.has(b) ? "✔ 抽中" : ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
