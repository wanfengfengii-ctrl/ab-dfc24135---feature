import { useCallback, useMemo, useState } from "react";
import {
  ApiError,
  AuditBlocker,
  AuditPlan,
  AuditZoneInput,
  createAuditPlan,
} from "./api";

interface Props {
  session: string;
  chunkCount: number;
  // Persisted plan restored from the server (GET). The parent re-keys this
  // component on every server sync, so this is the initial display state;
  // entering new conditions clears it locally until the next sync.
  restored: AuditPlan | null;
}

interface ZoneDraft {
  start: string;
  end: string;
  quota: string;
}

const EMPTY_ZONE: ZoneDraft = { start: "", end: "", quota: "" };

function parseInt10(raw: string): number | null {
  if (!/^-?\d+$/.test(raw.trim())) return null;
  return Number(raw.trim());
}

export default function AuditPlanCard({ session, chunkCount, restored }: Props) {
  const [target, setTarget] = useState("");
  const [riskText, setRiskText] = useState("");
  const [zones, setZones] = useState<ZoneDraft[]>([{ ...EMPTY_ZONE }]);
  // The plan currently on display; initialized from the server and cleared as
  // soon as any condition changes - stale results are never retained.
  const [plan, setPlan] = useState<AuditPlan | null>(restored);
  const [blockers, setBlockers] = useState<AuditBlocker[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState("");

  const risks = useMemo((): { values: number[] | null; count: number } => {
    const parts = riskText.split(/[\s,，、]+/).filter((p) => p.length > 0);
    const out: number[] = [];
    for (const p of parts) {
      const v = parseInt10(p);
      if (v === null) return { values: null, count: parts.length };
      out.push(v);
    }
    return { values: out, count: out.length };
  }, [riskText]);

  const parsedZones = useMemo((): { zones: AuditZoneInput[] | null } => {
    const out: AuditZoneInput[] = [];
    for (const z of zones) {
      const s = parseInt10(z.start);
      const e = parseInt10(z.end);
      const q = parseInt10(z.quota);
      if (s === null || e === null || q === null) return { zones: null };
      out.push({ start: s, end: e, quota: q });
    }
    return { zones: out };
  }, [zones]);

  // Editing the conditions drops any stale result: an old plan must never be
  // shown alongside new, not-yet-submitted conditions.
  const invalidate = useCallback(() => {
    setPlan(null);
    setBlockers(null);
    setFormError("");
  }, []);

  const clientError = useMemo(() => {
    const t = parseInt10(target);
    if (t === null) return "目标抽检块数必须是整数";
    if (t < 2 || t > 16) return "目标抽检块数必须在 2 至 16 之间";
    if (t > chunkCount) return `目标 ${t} 超过总块数 ${chunkCount}`;
    if (risks.values === null) return "风险分必须是 0 至 100 的整数（逗号/空白分隔）";
    if (risks.count !== chunkCount)
      return `风险分数量 ${risks.count} 与实际块数 ${chunkCount} 不一致`;
    if (risks.values.some((v) => v < 0 || v > 100)) return "每个风险分必须在 0 至 100 之间";
    if (zones.length < 1 || zones.length > 4) return "重点区间必须为 1 至 4 个";
    const pz = parsedZones.zones;
    if (pz === null) return "区间起止与配额必须是整数";
    for (let i = 0; i < pz.length; i++) {
      const z = pz[i];
      if (z.start < 0 || z.end < 0 || z.end >= chunkCount || z.start > z.end)
        return `区间 ${i + 1} 非法：要求 0 ≤ 起 ≤ 止 < ${chunkCount}`;
      if (z.quota < 1) return `区间 ${i + 1} 配额至少为 1`;
      if (z.quota > z.end - z.start + 1)
        return `区间 ${i + 1} 配额 ${z.quota} 超过区间长度 ${z.end - z.start + 1}`;
    }
    const sorted = [...pz].sort((a, b) => a.start - b.start);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].start <= sorted[i - 1].end)
        return `区间重叠：[${sorted[i - 1].start}, ${sorted[i - 1].end}] 与 [${sorted[i].start}, ${sorted[i].end}]`;
    }
    return "";
  }, [target, risks, zones, parsedZones, chunkCount]);

  const submit = useCallback(async () => {
    if (!parsedZones.zones || risks.values === null) return;
    setBusy(true);
    invalidate();
    try {
      const result = await createAuditPlan(session, {
        target: Number(target),
        risks: risks.values,
        zones: parsedZones.zones,
      });
      if (result.plan) {
        setPlan(result.plan);
      } else if (result.status === 409) {
        setBlockers(result.blockers ?? []);
      } else {
        setFormError(result.error ?? `请求被拒绝（HTTP ${result.status}）`);
      }
    } catch (e) {
      setFormError(`网络错误：${(e as ApiError).message ?? e}`);
    } finally {
      setBusy(false);
    }
  }, [invalidate, parsedZones.zones, risks.values, session, target]);

  const updateZone = (i: number, patch: Partial<ZoneDraft>) => {
    invalidate();
    setZones((prev) => prev.map((z, j) => (j === i ? { ...z, ...patch } : z)));
  };

  return (
    <section className="card audit">
      <h2>抽检计划（仅已封存会话）</h2>
      <p className="sub" style={{ margin: "0 0 12px" }}>
        共 {chunkCount} 块 · 目标 2–16 块 · 任意两块不相邻 · 风险分 0–100，
        数量必须等于块数 · 1–4 个互不重叠连续区间，各设最低抽检数
      </p>

      <label className="field">
        <span>目标抽检块数</span>
        <input
          value={target}
          inputMode="numeric"
          placeholder="2 至 16"
          onChange={(e) => {
            invalidate();
            setTarget(e.target.value);
          }}
          disabled={busy}
        />
      </label>

      <label className="field">
        <span>逐块风险分（{chunkCount} 个，逗号/空白分隔）</span>
        <textarea
          className="risk-input"
          rows={2}
          value={riskText}
          placeholder={`例如 ${chunkCount >= 3 ? "12, 80, 45 …" : "12, 80"}`}
          onChange={(e) => {
            invalidate();
            setRiskText(e.target.value);
          }}
          disabled={busy}
        />
        {riskText && (
          <em className={risks.count === chunkCount ? "good" : "bad"}>
            已填写 {risks.count} / {chunkCount}
          </em>
        )}
      </label>

      <div className="field">
        <span>重点区间（闭区间块号，最低抽检数 ≥ 1）</span>
        {zones.map((z, i) => (
          <div className="zone-row" key={i}>
            <em className="zone-tag">区间 {i + 1}</em>
            <input
              value={z.start}
              inputMode="numeric"
              placeholder="起"
              onChange={(e) => updateZone(i, { start: e.target.value })}
              disabled={busy}
            />
            <span>–</span>
            <input
              value={z.end}
              inputMode="numeric"
              placeholder="止"
              onChange={(e) => updateZone(i, { end: e.target.value })}
              disabled={busy}
            />
            <span>配额 ≥</span>
            <input
              className="quota-input"
              value={z.quota}
              inputMode="numeric"
              placeholder="n"
              onChange={(e) => updateZone(i, { quota: e.target.value })}
              disabled={busy}
            />
            {zones.length > 1 && (
              <button
                type="button"
                onClick={() => {
                  invalidate();
                  setZones((prev) => prev.filter((_, j) => j !== i));
                }}
                disabled={busy}
              >
                删除
              </button>
            )}
          </div>
        ))}
        <div className="row" style={{ marginBottom: 0 }}>
          <button
            type="button"
            onClick={() => {
              invalidate();
              setZones((prev) => (prev.length < 4 ? [...prev, { ...EMPTY_ZONE }] : prev));
            }}
            disabled={busy || zones.length >= 4}
          >
            添加区间（最多 4 个）
          </button>
        </div>
      </div>

      {(clientError || formError) && (
        <div className="bad">{clientError || formError}</div>
      )}

      <div className="row">
        <button
          className="primary"
          onClick={() => void submit()}
          disabled={busy || clientError !== ""}
        >
          {busy ? "正在生成…" : "生成抽检计划"}
        </button>
      </div>

      {blockers && (
        <div className="notice bad-notice">
          <strong>无解：服务器明确阻断，未生成任何计划。</strong>
          <ul className="blockers">
            {blockers.length === 0 && <li>当前条件下不存在可行组合。</li>}
            {blockers.map((b, i) => (
              <li key={i}>
                <code>{b.type}</code>：{b.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {plan && <PlanView plan={plan} />}
    </section>
  );
}

function PlanView({ plan }: { plan: AuditPlan }) {
  return (
    <div className="plan-result">
      <div className="notice good-notice">
        计划已生成并随回执持久化：抽检 <strong>{plan.selected.length}</strong> 块，
        风险分总和 <strong>{plan.risk_total}</strong>，选中块号{" "}
        <code>[{plan.selected.join(", ")}]</code>
        <div className="plan-time">生成时间 (UTC)：{plan.created_at}</div>
      </div>

      <h3>区间配额核算</h3>
      <table className="quota-table">
        <thead>
          <tr>
            <th>区间</th>
            <th>配额</th>
            <th>区间内选中</th>
            <th>核算</th>
          </tr>
        </thead>
        <tbody>
          {plan.zones.map((z, i) => (
            <tr key={i}>
              <td>
                [{z.start}, {z.end}]
              </td>
              <td>≥ {z.quota}</td>
              <td>{z.selected_count}（{z.selected.length ? z.selected.join(", ") : "—"}）</td>
              <td className={z.quota_met ? "good" : "bad"}>
                {z.quota_met ? "达标" : "未达标"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>逐块风险分与区间归属</h3>
      <div className="audit-grid">
        {plan.per_chunk.map((c) => (
          <div
            key={c.index}
            className={`audit-cell ${c.selected ? "picked" : ""} ${
              c.zone !== null ? "in-zone" : ""
            }`}
            title={
              c.zone !== null
                ? `块 ${c.index}：风险 ${c.risk}，属于区间 ${c.zone + 1}`
                : `块 ${c.index}：风险 ${c.risk}，不属于任何区间`
            }
          >
            <span className="ac-idx">#{c.index}</span>
            <span className="ac-risk">{c.risk}</span>
            <span className="ac-zone">{c.zone !== null ? `区${c.zone + 1}` : "—"}</span>
          </div>
        ))}
      </div>
      <div className="legend">
        <span className="swatch picked" /> 选中抽检
        <span className="swatch in-zone" /> 位于重点区间
        <span className="swatch" /> 普通块（格内：块号 / 风险分 / 区间）
      </div>
    </div>
  );
}
