import React, { useEffect, useMemo, useState } from "react";
import { SHOWCASE } from "../showcase.js";
import { api, areaKm2, bboxText, fmt, parseBbox } from "../api.js";

const store = {
  get() { try { return JSON.parse(localStorage.getItem("fs.form") || "{}"); } catch { return {}; } },
  set(v) { try { localStorage.setItem("fs.form", JSON.stringify(v)); } catch { /* private mode */ } },
};

const CHECKS = [
  ["copernicus_account", "Copernicus account in .env", "Add CDSE_USERNAME and CDSE_PASSWORD to .env"],
  ["sentinelhub_key", "Fast radar processing key (Sentinel Hub)", "Add SH_CLIENT_ID and SH_CLIENT_SECRET to .env (openEO is used without it, slower)"],
  ["openeo_login", "openEO login saved (backup)", "Run python scripts/check_setup.py --login once"],
  ["osm_key", "OpenStreetMap history key", "Add OHSOME_API_KEY to .env (otherwise the older keyless ohsome v1 is tried)"],
  ["model_weights", "Segmentation model weights", "Put siamese_unet.pt in models/ (rules still run without it)"],
  ["gpu", "GPU for the model", "Runs on CPU without one, a little slower"],
];

export default function NewPanel({ ctl, status, runs, onJob, onOpen }) {
  const saved = useMemo(store.get, []);
  const [bbox, setBbox] = useState(saved.bbox || null);
  const [text, setText] = useState(saved.bbox ? bboxText(saved.bbox) : "");
  const [date, setDate] = useState(saved.date || "2026-08-26");
  const [dryOn, setDryOn] = useState("");
  const [q, setQ] = useState("");
  const [found, setFound] = useState(null);       // null = not searched, [] = nothing found
  const [picked, setPicked] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchErr, setSearchErr] = useState(null);
  const [name, setName] = useState("");
  const [res, setRes] = useState(saved.res || 20);
  const [backend, setBackend] = useState("auto");
  const [useModel, setUseModel] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [reused, setReused] = useState(null);
  const [code, setCode] = useState(() => { try { return sessionStorage.getItem("fs_code") || ""; } catch { return ""; } });
  const pub = status?.public?.enabled ? status.public : null;
  // optional own Copernicus (Sentinel Hub) keys: kept only in this page's memory, sent with the request, never stored
  const [ownOpen, setOwnOpen] = useState(() => { try { return sessionStorage.getItem("fs_ownkeys") === "1"; } catch { return false; } });
  const [shId, setShId] = useState("");
  const [shSecret, setShSecret] = useState("");
  const ownKeys = shId.trim() && shSecret.trim();
  useEffect(() => { if (pub && pub.runs_left_for_you === 0) setOwnOpen(true); }, [pub?.runs_left_for_you]);   // eslint-disable-line

  const limits = status?.limits || { max_area_km2: 3000, batch_above_km2: 400, min_date: "2014-10-03" };
  const area = bbox ? areaKm2(bbox) : null;
  const maxKm2 = pub ? Math.min(pub.max_km2, limits.max_area_km2) : limits.max_area_km2;
  const tooBig = area != null && area > maxKm2;

  useEffect(() => {
    ctl.current?.setAoi(bbox);
    store.set({ bbox, date, res });
  }, [bbox, date, res, ctl]);
  useEffect(() => () => ctl.current?.stopDraw(), [ctl]);
  useEffect(() => { if (saved.bbox) ctl.current?.fitBbox(saved.bbox); }, [saved, ctl]);

  const setArea = (b, fit = true) => {
    setBbox(b);
    setText(b ? bboxText(b) : "");
    setReused(null);
    setError(null);
    if (b && fit) ctl.current.fitBbox(b);
  };

  const search = async (e) => {
    e?.preventDefault();
    if (q.trim().length < 2 || searching) return;
    setSearching(true);
    setSearchErr(null);
    try {
      const out = await api.places(q.trim());
      setFound(out.results || []);
    } catch (err) {
      setFound(null);
      setSearchErr(err.message);
    } finally {
      setSearching(false);
    }
  };

  const pick = (r) => {
    setPicked(r);
    setArea(r.bbox, true);
  };

  const draw = () => {
    if (drawing) { ctl.current.stopDraw(); setDrawing(false); return; }
    setDrawing(true);
    ctl.current.startDraw((b) => { setDrawing(false); setArea(b, true); });
  };

  const typed = (v) => {
    setText(v);
    const b = parseBbox(v);
    if (b) { setBbox(b); setReused(null); }
  };

  const start = async (extra = {}) => {
    setBusy(true);
    setError(null);
    try {
      const body = extra.demo
        ? { demo: true, date: "2026-08-26" }
        : { bbox: bbox.join(","), date, name: name.trim() || null, resolution_m: res, backend, use_model: useModel,
            before_date: dryOn || null,
            ...(pub ? { access_code: code } : {}),
            ...(ownKeys ? { sh_client_id: shId.trim(), sh_client_secret: shSecret.trim() } : {}), ...extra };
      if (pub && code) { try { sessionStorage.setItem("fs_code", code); } catch { /* private mode */ } }
      const out = await api.start(body);
      if (out.reused) {
        if (extra.demo) onOpen(out.run_id);
        else setReused(out.run_id);
      } else onJob(out.job);
    } catch (e) {
      const m = String(e.message || "");
      if (m.startsWith("[own-keys]")) {           // our shared account's limit: offer the visitor's own keys
        setOwnOpen(true);
        setError(m.replace("[own-keys]", "").trim());
      } else setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const canRun = bbox && !tooBig && date && !busy;

  return (
    <div className="pad new">
      <h1 className="h1">Map a flood from space</h1>
      <p className="lede">Sentinel-1 radar sees through monsoon clouds. FloodSight compares images from before and after the
        flood, maps new water and debris, and works out which villages have lost their road to a hospital.</p>

      {(() => {
        const have = new Map(runs.map((r) => [r.run_id, r]));
        const cards = SHOWCASE.filter((c) => have.has(c.id));
        if (!cards.length) return null;
        return (
          <section className="showcase" aria-label="Finished analyses">
            <h2 className="h3">Open a finished analysis <small>instant, no code needed</small></h2>
            <div className="sc-grid">
              {cards.map((c) => {
                const r = have.get(c.id), k = r.check;
                return (
                  <a key={c.id} className={`sc-card ${c.kind}`} href={`#/run/${encodeURIComponent(c.id)}`}>
                    <span className="sc-top"><b>{c.title}</b><em>{c.tag}</em></span>
                    <small>{c.sub}</small>
                    <span className="sc-facts">
                      {r.verdict === "flood_detected" ? `${fmt.n(r.cut_off || 0)} settlements cut off` : r.verdict === "possible_flood" ? "Possible flooding, not confirmed" : "No significant change"}
                      {k && k.iou != null ? ` · IoU ${Number(k.iou).toFixed(2)} · ${Math.round(100 * (k.precision || 0))}% of flagged damage confirmed` : ""}
                    </span>
                  </a>
                );
              })}
            </div>
          </section>
        );
      })()}

      <button className="linkbtn mp-link" onClick={() => window.dispatchEvent(new Event("fs-model"))}>How the AI model was tested →</button>

      <h2 className="h3 or-new">Or analyse a new area</h2>

      <section className="step">
        <h2 className="h2"><span className="stepno">1</span>Area</h2>
        <form className="field place-search" onSubmit={search}>
          <span>Find a place: name, or a point as latitude, longitude</span>
          <div className="row">
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Betrawati, Nuwakot, Nepal  or  27.9705, 85.1866"
              aria-label="Place name or coordinates" maxLength={200} />
            <button className="btn secondary" type="submit" disabled={searching || q.trim().length < 2}>
              {searching ? "Searching…" : "Search"}
            </button>
          </div>
        </form>
        {searchErr && <p className="hint warn">{searchErr}</p>}
        {found && found.length === 0 && (
          <p className="hint warn">No place found. Add the district or state and the country, or draw the box on the map.</p>
        )}
        {found && found.length > 0 && (
          <ul className="place-results">
            {found.map((r, i) => (
              <li key={i}>
                <button className={picked === r ? "on" : ""} onClick={() => pick(r)}>
                  <span>{r.name}</span>
                  <small>{r.kind ? `${r.kind} · ` : ""}box: {r.bbox_from}</small>
                </button>
              </li>
            ))}
          </ul>
        )}
        {picked && bbox && (
          <p className="hint">A place outline is not the flood area. Move or redraw the box so it covers the river or
            valley that flooded and the roads to the nearest town.</p>
        )}
        {found && found[0]?.source !== "typed" && <p className="hint small">Place search: Nominatim, data © OpenStreetMap contributors. It only
          positions the box; the analysis uses satellite, elevation and pre-event map data.</p>}
        <div className="row">
          <button className={`btn ${drawing ? "active" : "secondary"}`} onClick={draw} aria-pressed={drawing}>
            {drawing ? "Drag a box on the map…" : bbox ? "Draw a new box" : "Draw a box on the map"}
          </button>
          {bbox && <button className="btn ghost" onClick={() => setArea(null)}>Clear</button>}
        </div>
        <label className="field">
          <span>Corners (west, south, east, north in degrees)</span>
          <input value={text} onChange={(e) => typed(e.target.value)} placeholder="85.08, 27.85, 85.40, 28.29"
            inputMode="decimal" spellCheck="false" aria-invalid={text && !parseBbox(text) ? "true" : "false"} />
        </label>
        {text && !parseBbox(text) && <p className="hint warn">Type four numbers: west, south, east, north (west &lt; east, south &lt; north).</p>}
        {area != null && (
          <p className={`hint ${tooBig ? "warn" : ""}`}>
            {fmt.n(area)} km²
            {tooBig ? ` is over the ${fmt.n(maxKm2)} km² limit. Draw a tighter box around the valley.`
              : area > limits.batch_above_km2
                ? ". Large areas are processed as a queued job on Copernicus: allow 10 to 20 minutes the first time, a few minutes on a re-run."
                : ". Usually ready in a few minutes."}
          </p>
        )}
        {status?.presets?.length > 0 && (
          <details className="presets">
            <summary>Or pick a known flood</summary>
            <ul>
              {status.presets.map((p) => (
                <li key={p.name}>
                  <button onClick={() => { setArea(parseBbox(p.bbox)); setDate(p.date); }}>
                    <span>{p.name}</span><time>{fmt.date(p.date)}</time>
                  </button>
                </li>
              ))}
            </ul>
          </details>
        )}
      </section>

      <section className="step">
        <h2 className="h2"><span className="stepno">2</span>Flood date</h2>
        <label className="field">
          <span>Day the flood happened</span>
          <input type="date" value={date} min={limits.min_date} max={limits.max_date}
            onChange={(e) => { setDate(e.target.value); setReused(null); }} />
        </label>
        <p className="hint">Radar images up to a month before and two weeks after this day are searched, on the same orbit track.</p>
        <label className="field">
          <span>Area still dry on (optional)</span>
          <input type="date" value={dryOn} min={limits.min_date} max={date || limits.max_date}
            onChange={(e) => { setDryOn(e.target.value); setReused(null); }} />
        </label>
        <p className="hint">Only for floods that built up over days or weeks (monsoon plains). The "before" images and the
          OpenStreetMap snapshot are then taken on or before this day, so water that was already there on the
          flood date is still counted as flood. Leave empty for a sudden flood.</p>
      </section>

      {pub && (
        <section className="step">
          <h2 className="h2"><span className="stepno">3</span>Access code</h2>
          <label className="field">
            <span>Judges' access code (needed only to start a new analysis)</span>
            <input type="password" value={code} onChange={(e) => setCode(e.target.value)} autoComplete="off" />
          </label>
          <p className="hint">Finished analyses and the offline demo are open without a code. New analyses download
            satellite data under the team's Copernicus quota, so this public server allows {fmt.n(pub.runs_left_for_you)} more
            new {pub.runs_left_for_you === 1 ? "analysis" : "analyses"} from your connection (of {fmt.n(pub.max_runs_per_visitor)}),
            {" "}{fmt.n(pub.max_runs_day)} per day in total ({fmt.n(pub.runs_today)} used today), and areas up to {fmt.n(pub.max_km2)} km².
            A run takes about 5 to 10 minutes here. With your own Copernicus keys (below) these limits do not apply.</p>
        </section>
      )}

      {pub && (
        <details className="advanced ownkeys" open={ownOpen} onToggle={(e) => setOwnOpen(e.currentTarget.open)}>
          <summary>Use my own Copernicus keys (optional)</summary>
          <p className="hint">With a free Copernicus Data Space account you can run as many analyses as your own quota allows.
            Create Sentinel Hub OAuth client keys in your account (the project README and video show how) and paste them here.
            We never ask for your account password.</p>
          <label className="field"><span>Sentinel Hub client ID</span>
            <input value={shId} onChange={(e) => setShId(e.target.value)} autoComplete="off" spellCheck={false} /></label>
          <label className="field"><span>Sentinel Hub client secret</span>
            <input type="password" value={shSecret} onChange={(e) => setShSecret(e.target.value)} autoComplete="off" /></label>
          <p className="fine">Your keys are sent over an encrypted connection, used only for your next analysis, kept in memory
            while it runs and then forgotten: they are never saved, logged or shown again. That run uses only your account
            (no fallback to ours). You can delete the keys in your Copernicus account afterwards.
            {ownKeys ? " Your keys will be used for the next analysis." : ""}</p>
        </details>
      )}

      {!pub && (
      <details className="advanced">
        <summary>More options</summary>
        <label className="field"><span>Name for this analysis (optional)</span>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. trishuli_case" /></label>
        <label className="field"><span>Map detail</span>
          <select value={res} onChange={(e) => setRes(Number(e.target.value))}>
            <option value={10}>10 m pixels (slowest, small areas)</option>
            <option value={20}>20 m pixels (recommended)</option>
            <option value={30}>30 m pixels (fastest, large areas)</option>
          </select></label>
        <label className="field"><span>Radar processing</span>
          <select value={backend} onChange={(e) => setBackend(e.target.value)}>
            <option value="auto">Automatic (fastest available, with backups)</option>
            <option value="sentinelhub">Copernicus Sentinel Hub only (fast)</option>
            <option value="openeo">Copernicus openEO only</option>
            <option value="local">Our own raw GRD processing, on this computer (slow: downloads full radar files)</option>
          </select></label>
        <label className="check"><input type="checkbox" checked={useModel} onChange={(e) => setUseModel(e.target.checked)} />
          Use the AI segmentation model alongside the physical rules</label>
      </details>
      )}

      {error && <p className="error" role="alert">{error}</p>}
      {reused ? (
        <div className="callout">
          <p>This area and date have already been analysed with these settings.</p>
          <div className="row">
            <button className="btn" onClick={() => onOpen(reused)}>Open the saved result</button>
            <button className="btn secondary" onClick={() => start({ force: true })}>Analyse again</button>
          </div>
        </div>
      ) : (
        <button className="btn primary wide" disabled={!canRun} onClick={() => start()}>
          {busy ? "Starting…" : "Analyse this area"}
        </button>
      )}
      {!bbox && <p className="hint">Draw or type an area first.</p>}
      <button className="link" onClick={() => start({ demo: true })} disabled={busy}>
        No internet? Run the offline demo valley (about 15 seconds)
      </button>

      {status && !status.offline && !pub && (
        <section className="ready">
          <h2 className="h3">Ready for live satellite data?</h2>
          <ul>
            {CHECKS.map(([k, label, fix]) => {
              const v = status.checks?.[k];
              const ok = v === true || (typeof v === "string" && v);
              const unknown = v === null || v === undefined;
              return (
                <li key={k} className={ok ? "ok" : unknown ? "unk" : "no"}>
                  <span className="dot" aria-hidden="true" />
                  <span>{label}{typeof v === "string" ? `: ${v}` : ""}
                    {!ok && <small>{unknown ? "Could not check from here. " : ""}{fix}</small>}</span>
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {status?.offline && <p className="error">The FloodSight server is not reachable. Start it with python serve.py.</p>}
      {status?.busy?.length > 0 && (
        <p className="callout">An analysis is running now.{" "}
          <a href={`#/job/${status.busy[0].id}`}>Follow its progress</a>.</p>
      )}

      {runs.length > 0 && (
        <section className="recent">
          <h2 className="h3">Earlier analyses</h2>
          <ul>
            {runs.slice(0, 12).map((r) => (
              <li key={r.run_id}>
                <a href={`#/run/${encodeURIComponent(r.run_id)}`}>
                  <span className={`vdot ${r.verdict === "flood_detected" ? "hit" : r.verdict === "possible_flood" ? "maybe" : "calm"}`} aria-hidden="true" />
                  <span className="r-main">
                    <b>{r.run_id}</b>
                    <small>{r.demo ? "Demo valley, " : ""}flood {fmt.date(r.flood_date)}, {fmt.n(r.area_km2)} km²</small>
                  </span>
                  <span className="r-side">
                    {r.verdict === "flood_detected" ? <>{fmt.n(r.cut_off)} cut off</> : r.verdict === "possible_flood" ? "Not confirmed" : "No change"}
                    {r.validated && <small>Checked vs EMS</small>}
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
