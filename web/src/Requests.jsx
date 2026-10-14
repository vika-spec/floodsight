import React from "react";
import { fmt } from "./api.js";

// Every outside data request of a run: which service, what, how long, and why it waited. Shown live while a
// run downloads (Job panel) and kept in results.json (Method tab), so the slow part is never a black box.
// Each row has a bar on a shared time axis: overlapping bars ran at the same time.

const MARK = { running: "…", done: "✓", cached: "↺", failed: "!" };

function meta(r) {
  const out = [];
  if (r.status === "cached") out.push("reused from an earlier run");
  if (r.requests) out.push(`${r.requests} request${r.requests === 1 ? "" : "s"}`);
  if (r.mb >= 0.05) out.push(`${r.mb.toFixed(1)} MB`);
  if (r.pu) out.push(`${fmt.n(r.pu, 1)} processing units`);
  if (r.detail && r.status !== "cached") out.push(r.detail);
  return out.join(" · ");
}

// items: [{source, what, status, start (s, any origin), seconds, ...}]; now: same clock as start (live view)
export default function RequestList({ items, now = null, title = null, note = false }) {
  if (!items || !items.length) return null;
  const rows = items.map((r) => ({ ...r, dur: r.status === "running" && now != null ? Math.max(0, now - r.start) : r.seconds || 0 }));
  const t0 = Math.min(...rows.map((r) => r.start));
  const t1 = Math.max(...rows.map((r) => r.start + r.dur), t0 + 1);
  const span = t1 - t0;
  const bySource = {};
  rows.forEach((r) => { if (r.status !== "cached") bySource[r.source] = (bySource[r.source] || 0) + r.dur; });
  const slow = rows.filter((r) => r.status !== "cached").sort((a, b) => b.dur - a.dur)[0];
  const waited = rows.reduce((s, r) => s + (r.waited_s || 0), 0);
  return (
    <section className="reqs" aria-label="Data requests">
      {title && <h3 className="reqs-title">{title}</h3>}
      {rows.length > 2 && <p className="reqs-sum">
        {rows.length} data request{rows.length === 1 ? "" : "s"} to {Object.keys(bySource).length || 1} service{Object.keys(bySource).length === 1 ? "" : "s"}
        {slow && slow.dur >= 1 ? `; slowest: ${slow.what} (${fmt.dur(slow.dur)}, ${slow.source})` : ""}
        {waited >= 1 ? `; ${fmt.dur(waited)} of it spent waiting for busy servers` : ""}.
      </p>}
      <ol className="reqs-list">
        {rows.map((r, i) => (
          <li key={r.id ?? i} className={`req ${r.status}`}>
            <span className="req-mark" aria-label={r.status}>{MARK[r.status] || "·"}</span>
            <span className="req-what"><b>{r.what}</b><small>{r.source}</small></span>
            <span className="req-time">{r.status === "cached" ? "0 s" : fmt.dur(r.dur)}</span>
            {rows.length > 1 && <span className="req-bar" aria-hidden="true">
              <i style={{ left: `${(100 * (r.start - t0)) / span}%`, width: `${Math.max((100 * r.dur) / span, 0.8)}%` }} />
            </span>}
            {meta(r) && <span className="req-meta">{meta(r)}</span>}
          </li>
        ))}
      </ol>
      {note && <p className="fine">Bars share one time axis: bars that overlap ran at the same time. Images are cut out and calibrated on the
        Copernicus servers, so the time here is mostly their processing and the network, not this computer.</p>}
    </section>
  );
}
