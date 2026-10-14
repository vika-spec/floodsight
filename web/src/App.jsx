import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, fmt } from "./api.js";
import { MapCtl } from "./mapctl.js";
import LayerCard from "./LayerCard.jsx";
import NewPanel from "./panels/New.jsx";
import JobPanel from "./panels/Job.jsx";
import RunPanel from "./panels/Run.jsx";
import Lightbox from "./Lightbox.jsx";
import ModelPage from "./ModelPage.jsx";

// URL hash keeps the screen on reload and makes a result linkable: #/new, #/job/<id>, #/run/<run_id>
const readHash = () => {
  const [, kind, id] = (window.location.hash || "#/new").split("/");
  return { kind: kind || "new", id: id ? decodeURIComponent(id) : null };
};
const go = (kind, id) => {
  window.location.hash = id ? `#/${kind}/${encodeURIComponent(id)}` : `#/${kind}`;
};

export const DEFAULT_VISIBLE = {
  flood: true, change: false, before: false, after: false, model: false, rules: false,
  buildings: true, roads: true, broken: true, settlements: true, health: true, landing: false,   // landing: off until asked for (many symbols)
};

export default function App() {
  const mapEl = useRef(null);
  const ctl = useRef(null);
  const [route, setRoute] = useState(readHash());
  const [status, setStatus] = useState(null);
  const [runs, setRuns] = useState([]);
  const [R, setR] = useState(null);
  const [runError, setRunError] = useState(null);
  const [visible, setVisible] = useState(DEFAULT_VISIBLE);
  const [basemap, setBasemap] = useState("satellite");
  const [compare, setCompare] = useState(null); // null | {right, frac}
  const [panelOpen, setPanelOpen] = useState(true);
  const NARROW = 420;
  const [panelW, setPanelW] = useState(NARROW);
  const wideW = () => Math.round(Math.min(window.innerWidth * 0.6, window.innerWidth - 360));
  const isWide = panelW > NARROW + 40;
  const [layersTick, setLayersTick] = useState(0);
  const [modelOpen, setModelOpen] = useState(false);
  const closeModel = useCallback(() => setModelOpen(false), []);
  useEffect(() => {
    const open = () => setModelOpen(true);
    window.addEventListener("fs-model", open);
    return () => window.removeEventListener("fs-model", open);
  }, []);

  // ------------------------------------------------------------------------------------------- boot
  useEffect(() => {
    ctl.current = new MapCtl(mapEl.current);
    const onHash = () => setRoute(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refresh = useCallback(() => {
    api.status().then(setStatus).catch(() => setStatus({ offline: true }));
    api.runs().then(setRuns).catch(() => {});
  }, []);
  useEffect(refresh, [refresh]);

  // ------------------------------------------------------------------------------------------- open a run
  useEffect(() => {
    const m = ctl.current;
    if (route.kind !== "run" || !route.id) {
      setR(null);
      setCompare(null);
      m.clearRun();
      return;
    }
    let live = true;
    setRunError(null);
    api.run(route.id)
      .then((res) => {
        if (!live) return;
        setR(res);
        setCompare(null);
        m.setAoi(res.aoi.bbox, { fillOpacity: 0 });
        m.setRun(res, visible).then(() => live && setLayersTick((t) => t + 1));
        m.fitBbox(res.aoi.bbox); // after setRun starts, so the fit leaves room for the layer card
      })
      .catch((e) => live && setRunError(e.message));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route.kind, route.id]);

  useEffect(() => { ctl.current.applyVisibility(visible); }, [visible]);
  useEffect(() => { ctl.current.setBasemap(basemap); }, [basemap]);
  useEffect(() => {
    if (compare) ctl.current.compare(true, compare.right, compare.frac);
    else ctl.current.compare(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compare?.right, !!compare]);
  useEffect(() => { ctl.current.invalidate(); }, [panelOpen, panelW]);

  // drag the edge between the panel and the map to resize the panel (double-click: back to normal)
  const startResize = (e) => {
    e.preventDefault();
    const move = (ev) => {
      const x = ev.touches ? ev.touches[0].clientX : ev.clientX;
      setPanelW(Math.round(Math.min(Math.max(x, 320), window.innerWidth - 320)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      document.body.classList.remove("resizing");
      ctl.current.invalidate();
    };
    document.body.classList.add("resizing");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const openRun = (id) => { refresh(); go("run", id); };

  return (
    <div className={`app ${panelOpen ? "" : "panel-closed"}`} style={{ "--panel-w": `${panelW}px` }}>
      <header className="topbar">
        <a className="brand" href="#/new" aria-label="FloodSight home">
          <svg viewBox="0 0 32 32" aria-hidden="true"><path d="M2 22c4-4 8-4 12 0s8 4 16-2v10H2z" fill="var(--water)" /><path d="M4 18 12 6l6 8 3-4 7 8" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" /></svg>
          <span>FloodSight</span>
        </a>
        <p className="tagline">Where the flood hit, what it damaged, who is cut off</p>
        <nav className="topnav">
          <button className="btn ghost small mp-top" onClick={() => setModelOpen(true)}>How the AI was tested</button>
          {route.kind !== "new" && <a className="btn ghost small" href="#/new">New analysis</a>}
          {panelOpen && (
            <button className="btn ghost small panel-toggle" onClick={() => setPanelW(isWide ? NARROW : wideW())}>
              {isWide ? "Narrow panel" : "Wider panel"}
            </button>
          )}
          <button className="btn ghost small panel-toggle" onClick={() => setPanelOpen((v) => !v)} aria-expanded={panelOpen}>
            {panelOpen ? "Hide panel" : "Show panel"}
          </button>
        </nav>
      </header>

      <aside className="panel" aria-label="Analysis panel">
        {route.kind === "new" && (
          <NewPanel ctl={ctl} status={status} runs={runs}
            onJob={(job) => go("job", job.id)} onOpen={openRun} />
        )}
        {route.kind === "job" && (
          <JobPanel ctl={ctl} jobId={route.id} onDone={openRun} onBack={() => go("new")} />
        )}
        {route.kind === "run" && (
          runError ? (
            <div className="pad"><h2 className="h2">Can't open this analysis</h2><p className="error">{runError}</p>
              <a className="btn" href="#/new">Start a new analysis</a></div>
          ) : R ? (
            <RunPanel R={R} ctl={ctl} visible={visible} setVisible={setVisible}
              compare={compare} setCompare={setCompare} onRerun={() => go("new")} />
          ) : <div className="pad muted">Opening the analysis…</div>
        )}
      </aside>

      <main className="mapwrap">
        <div ref={mapEl} className="map" />
        {panelOpen && (
          <div className="resizer" onPointerDown={startResize} onDoubleClick={() => setPanelW(NARROW)}
            role="separator" aria-orientation="vertical" aria-label="Drag to resize the panel" title="Drag to resize the panel (double-click: normal width)" />
        )}
        {R && route.kind === "run" && !compare && (() => {
          const mode = visible.model && !visible.flood ? "ai" : visible.rules && !visible.flood ? "rules" : "final";
          const set = (m) => setVisible((v) => ({ ...v, flood: m === "final", model: m === "ai", rules: m === "rules" }));
          const opts = [["final", "Final map", true], ["ai", "AI alone", !!R.web?.model], ["rules", "Radar rules alone", !!R.web?.rules]];
          return (
            <div className="mapswitch" role="group" aria-label="What the map shows">
              {opts.filter((o) => o[2]).map(([k, label]) => (
                <button key={k} className={mode === k ? "on" : ""} aria-pressed={mode === k} onClick={() => set(k)}>{label}</button>
              ))}
              {R.web?.before && R.web?.after && (
                <button onClick={() => window.dispatchEvent(new CustomEvent("fs-tab", { detail: "compare" }))}>Before ↔ after</button>
              )}
            </div>
          );
        })()}
        {R && route.kind === "run" && !compare && (
          <div className="maptools">
            {R.helicopter_access && (
              <button className={`btn small heli-toggle ${visible.landing ? "on" : ""}`} aria-pressed={!!visible.landing}
                onClick={() => setVisible((v) => ({ ...v, landing: !v.landing }))}>
                <span className="heli-ico" aria-hidden="true">H</span>
                {visible.landing ? "Hide helicopter landing areas" : "Helicopter landing areas"}
              </button>
            )}
            <button className="btn small tool" onClick={() => window.dispatchEvent(new CustomEvent("fs-tab", { detail: "flow" }))}>
              Trace flood path from a point
            </button>
          </div>
        )}
        {R && route.kind === "run" && (
          <LayerCard key={`${R.run_id}-${layersTick}`} R={R} collapse={!!compare} visible={visible} setVisible={setVisible} basemap={basemap} setBasemap={setBasemap} ctl={ctl} />
        )}
        {route.kind !== "run" && (
          <div className="basemap-mini" role="group" aria-label="Background map">
            {["satellite", "streets", "topo"].map((k) => (
              <button key={k} className={basemap === k ? "on" : ""} onClick={() => setBasemap(k)}>
                {{ satellite: "Satellite", streets: "Streets", topo: "Terrain" }[k]}
              </button>
            ))}
          </div>
        )}
        {compare && <Swipe compare={compare} setCompare={setCompare} ctl={ctl} R={R} />}
      </main>
      <Lightbox />
      {modelOpen && <ModelPage onClose={closeModel} />}
    </div>
  );
}

function Swipe({ compare, setCompare, ctl, R }) {
  const ref = useRef(null);
  const [frac, setFrac] = useState(() => ctl.current.aoiCenterFrac() ?? compare.frac ?? 0.5);
  useEffect(() => { ctl.current.setSplit(frac); }, [frac, ctl]);
  const drag = (e) => {
    const box = ref.current.parentElement.getBoundingClientRect();
    const move = (ev) => {
      const x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - box.left;
      setFrac(Math.min(0.97, Math.max(0.03, x / box.width)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    e.preventDefault();
  };
  const key = (e) => {
    if (e.key === "ArrowLeft") setFrac((f) => Math.max(0.03, f - 0.03));
    if (e.key === "ArrowRight") setFrac((f) => Math.min(0.97, f + 0.03));
  };
  const pre = R?.inputs?.sentinel1_pairs?.[0];
  const rightLabel = { after: `After, ${fmt.date(pre?.post_date)}`, change: "Change after vs before", flood: "After, with detected damage" }[compare.right];
  return (
    <>
      <div ref={ref} className="swipe" style={{ left: `${frac * 100}%` }}>
        <div className="swipe-label left">Before, {fmt.date(pre?.pre_date)}</div>
        <div className="swipe-label right">{rightLabel}</div>
        <button className="swipe-handle" onPointerDown={drag} onKeyDown={key}
          role="slider" aria-label="Drag to compare before and after" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={Math.round(frac * 100)}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6 3 12l6 6M15 6l6 6-6 6" fill="none" stroke="currentColor" strokeWidth="2.2" /></svg>
        </button>
      </div>
      <button className="btn small swipe-close" onClick={() => setCompare(null)}>Close comparison</button>
    </>
  );
}
