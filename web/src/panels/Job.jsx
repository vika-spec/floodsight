import React, { useEffect, useRef, useState } from "react";
import { api, fmt, parseBbox } from "../api.js";
import { STAGES } from "../theme.js";
import RequestList from "../Requests.jsx";

const FAIL = {
  no_data: "No usable satellite images",
  input: "Check the area and date",
  error: "The analysis stopped",
  quota: "The Copernicus account has reached its limit",
};

export default function JobPanel({ ctl, jobId, onDone, onBack }) {
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);
  const [now, setNow] = useState(Date.now());
  const skew = useRef(0); // browser clock minus server clock, so stage timers are right on another device
  const logRef = useRef(null);
  const fitted = useRef(false);

  useEffect(() => {
    let live = true;
    let timer;
    const poll = async () => {
      try {
        const j = await api.job(jobId);
        if (!live) return;
        setJob(j);
        setErr(null);
        if (j.server_time) skew.current = Date.now() / 1000 - j.server_time;
        if (!fitted.current && j.params?.bbox) {
          const b = parseBbox(j.params.bbox);
          ctl.current.setAoi(b, { color: "#1E6EE6" });
          ctl.current.fitBbox(b);
          fitted.current = true;
        }
        if (j.state === "done") { setTimeout(() => live && onDone(j.run_id), 700); return; }
        if (j.state === "failed") return;
      } catch (e) {
        if (!live) return;
        setErr(e.message);
      }
      timer = setTimeout(poll, 1500);
    };
    poll();
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => { live = false; clearTimeout(timer); clearInterval(tick); };
  }, [jobId, ctl, onDone]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log?.length]);

  if (err && !job) {
    return (
      <div className="pad">
        <h1 className="h1">Can't follow this analysis</h1>
        <p className="error">{err}</p>
        <button className="btn" onClick={onBack}>Start a new analysis</button>
      </div>
    );
  }
  if (!job) return <div className="pad muted">Connecting…</div>;

  const t = now / 1000 - skew.current; // "now" on the server clock
  const started = Object.fromEntries((job.stages || []).map((s) => [s.name, s]));
  const demo = job.params?.demo;
  const steps = (demo ? [["Loading demo valley", "Load the synthetic demo valley"], ...STAGES.slice(2)] : STAGES)
    .filter(([k]) => !(k.startsWith("Segmentation") && job.params?.use_model === false) || started[k]);
  // data requests grouped under the step in which they started
  const acts = (job.activity || []).map((a) => ({ ...a, seconds: a.end ? a.end - a.start : null }));
  const stageList = job.stages || [];
  const actsFor = (key) => {
    const i = stageList.findIndex((x) => x.name === key);
    if (i < 0) return [];
    const a0 = stageList[i].start, a1 = stageList[i + 1]?.start ?? Infinity;
    return acts.filter((a) => a.start >= a0 - 0.5 && a.start < a1);
  };
  const elapsed = job.state === "running" && job.stages?.length
    ? (t - job.stages[0].start) : job.elapsed_s;

  return (
    <div className="pad job">
      <h1 className="h1">{job.state === "failed" ? FAIL[job.error_kind] || "The analysis stopped"
        : job.state === "done" ? "Done, opening the map" : job.state === "queued" ? "Waiting for the previous analysis" : "Analysing"}</h1>
      <p className="lede">
        {demo ? "Offline demo valley" : <>Area {job.params?.bbox?.split(",").map((x) => Number(x).toFixed(2)).join(", ")}</>},
        flood on {fmt.date(job.params?.date)}.{" "}
        {job.state !== "failed" && <>Elapsed {fmt.dur(Math.max(0, elapsed))}.</>}
      </p>
      {job.state === "queued" && <p className="callout">Number {job.queue_position} in line. Analyses run one at a time so they don't compete for the Copernicus connection.</p>}

      {job.state === "failed" && (
        <div className="callout bad" role="alert">
          <p>{job.error}</p>
          {job.error_kind === "no_data" && <p>Try a date a few days later, or a slightly larger area so a full radar frame covers it.</p>}
          {job.error_kind === "quota" && (
            <button className="btn primary" onClick={() => { try { sessionStorage.setItem("fs_ownkeys", "1"); } catch { /* */ } onBack(); }}>
              Use my own Copernicus keys
            </button>
          )}
          <button className="btn" onClick={onBack}>Change the area or date</button>
        </div>
      )}

      <ol className="stages">
        {steps.map(([key, label]) => {
          const s = started[key];
          const state = !s ? "todo" : s.end ? "done" : job.state === "failed" ? "fail" : "now";
          const dur = s ? ((s.end || t) - s.start) : null;
          return (
            <li key={key} className={state} aria-current={state === "now" ? "step" : undefined}>
              <span className="mark" aria-hidden="true" />
              <span className="st-label">{label}</span>
              {dur != null && <span className="st-time">{fmt.dur(dur)}</span>}
              {actsFor(key).length > 0 && (
                <div className="st-reqs"><RequestList items={actsFor(key)} now={t} note={key.startsWith("Downloading")} /></div>
              )}
            </li>
          );
        })}
      </ol>


      <details className="log" open={job.state === "failed"}>
        <summary>Technical log</summary>
        <pre ref={logRef}>{(job.log || []).join("\n") || "Starting…"}</pre>
      </details>
      {err && <p className="hint warn">{err} Retrying…</p>}
    </div>
  );
}
