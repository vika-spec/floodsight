import React, { useEffect, useMemo, useRef, useState } from "react";
import { zoom, ZoomImg } from "../Lightbox.jsx";
import { api, fileUrl, fmt } from "../api.js";
import RequestList from "../Requests.jsx";
import { C, STATUS } from "../theme.js";

const TABS = [
  ["summary", "Overview"],
  ["access", "Cut off"],
  ["compare", "Before / after"],
  ["ai", "AI model"],
  ["flow", "Flow path"],
  ["report", "Report"],
  ["check", "Validation"],
  ["details", "Method"],
];

export default function RunPanel({ R, ctl, visible, setVisible, compare, setCompare, onRerun }) {
  const [tab, setTab] = useState("summary");
  useEffect(() => setTab("summary"), [R.run_id]);
  useEffect(() => {          // the map's buttons (before/after, flow path) open a tab here
    const go = (e) => setTab(e.detail);
    window.addEventListener("fs-tab", go);
    return () => window.removeEventListener("fs-tab", go);
  }, []);
  const demo = (R.notes || []).some((n) => n.startsWith("DEMO MODE"));

  return (
    <div className="run">
      <Verdict R={R} demo={demo} />
      <KeyNumbers R={R} setTab={setTab} />
      <TrustStrip R={R} setTab={setTab} />
      <nav className="tabs" role="tablist" aria-label="Result sections">
        {TABS.map(([k, label]) => (
          <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>
            {label}
            {k === "access" && R.access?.settlements_cut_off > 0 && <span className="count">{R.access.settlements_cut_off}</span>}
          </button>
        ))}
      </nav>
      <div className="pad tabbody" role="tabpanel">
        {tab === "summary" && <><ReachFirst R={R} ctl={ctl} setTab={setTab} />
          <Summary R={R} setVisible={setVisible} visible={visible} setTab={setTab} setCompare={setCompare} /></>}
        {tab === "access" && <Access R={R} ctl={ctl} />}
        {tab === "compare" && <Compare R={R} compare={compare} setCompare={setCompare} />}
        {tab === "ai" && <AIModel R={R} visible={visible} setVisible={setVisible} />}
        {tab === "flow" && <Flow R={R} ctl={ctl} />}
        {tab === "report" && <Report R={R} />}
        {tab === "check" && <Check R={R} />}
        {tab === "details" && <Details R={R} onRerun={onRerun} />}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ verdict
function Verdict({ R, demo }) {
  const a = R.areas, acc = R.access || {};
  const v = R.verdict;                                     // flood_detected | possible_flood | no_significant_change
  const pair = R.inputs.sentinel1_pairs[0];
  const [lon, lat] = R.aoi.center;
  const total = (a.flood_km2 || 0) + (a.debris_km2 || 0);
  const confirmed = a.confirmed_km2 ?? ((a.flood_high_conf_km2 || 0) + (a.debris_high_conf_km2 || 0));
  const possible = Math.max(total - confirmed, 0);
  const cls = v === "flood_detected" ? "hit" : v === "possible_flood" ? "maybe" : "calm";
  return (
    <header className={`verdict ${cls}`}>
      {demo && <p className="demo-flag">Demo valley: synthetic data, not a real place</p>}
      <h1>{v === "flood_detected" ? "Flood damage found" : v === "possible_flood" ? "Possible flooding, not confirmed" : "No significant flood change"}</h1>
      <p className="headline">
        {v === "flood_detected" && (
          <>{fmt.km2(confirmed)} of new water and debris confirmed
            {possible >= 0.01 ? <>; {fmt.km2(possible)} more possible (seen by one source only)</> : ""}.{" "}
            {acc.settlements_cut_off > 0
              ? <>{fmt.n(acc.settlements_cut_off)} {acc.settlements_cut_off === 1 ? "settlement has" : "settlements have"} lost their road to a hospital
                {acc.buildings_in_cut_off_settlements ? ` (about ${fmt.n(acc.buildings_in_cut_off_settlements)} buildings)` : ""}
                {acc.settlements_likely_cut_off ? `, ${fmt.n(acc.settlements_likely_cut_off)} likely` : ""}
                {acc.settlements_detour ? `, ${fmt.n(acc.settlements_detour)} face a long detour` : ""}.</>
              : acc.settlements_likely_cut_off > 0
                ? <>{fmt.n(acc.settlements_likely_cut_off)} {acc.settlements_likely_cut_off === 1 ? "settlement is" : "settlements are"} likely cut off (low-confidence detections: check on the ground).</>
                : "No settlement has lost its road to a hospital."}</>
        )}
        {v === "possible_flood" && (
          <>{fmt.km2(possible)} changed, but every change was seen by one source only (one radar pass, the AI alone or
            Sentinel-2 alone). Treat it as a lead to check, not as confirmed damage.</>
        )}
        {v === "no_significant_change" && (
          <>Less than {fmt.km2(0.05)} changed between the radar images. Parts the radar could not see are shown in grey on the map.</>
        )}
      </p>
      <p className="meta">
        {lat.toFixed(3)}° N, {lon.toFixed(3)}° E, {fmt.n(R.aoi.area_km2)} km². Flood {fmt.date(R.flood_date)};
        radar {fmt.date(pair.pre_date)} and {fmt.date(pair.post_date)}.
      </p>
    </header>
  );
}

// ------------------------------------------------------------------------------------------------ summary
function Row({ sw, label, value, sub, onClick, active }) {
  const Tag = onClick ? "button" : "div";
  const toggle = onClick && active !== undefined;
  return (
    <Tag className={`ledger-row ${onClick ? "click" : ""} ${toggle && !active ? "off" : ""}`} onClick={onClick}
      aria-pressed={toggle ? !!active : undefined} title={toggle ? (active ? "Hide this layer on the map" : "Show this layer on the map") : undefined}>
      <i className="sw" style={{ background: sw }} aria-hidden="true" />
      <span className="l">{label}{sub && <small>{sub}</small>}</span>
      <span className="v">{value}</span>
    </Tag>
  );
}

// Four numbers a responder needs first; each opens the section that explains it.
function KeyNumbers({ R, setTab }) {
  const a = R.areas || {}, acc = R.access || {}, d = R.damage || {};
  if (R.verdict === "no_significant_change") return null;
  // confirmed (two sources agree) first; what only one source saw is shown as "+ possible"
  const fc = a.flood_high_conf_km2 ?? a.flood_km2 ?? 0, dc = a.debris_high_conf_km2 ?? a.debris_km2 ?? 0;
  const pw = Math.max((a.flood_km2 || 0) + (a.debris_km2 || 0) - fc - dc, 0);
  const rc = d.road_km_confirmed ?? d.road_km_affected ?? 0, rp = Math.max((d.road_km_affected || 0) - rc, 0);
  const bc = d.bridges_confirmed ?? d.bridges_hit ?? 0, bp = Math.max((d.bridges_hit || 0) - bc, 0);
  const plus = (txt, on) => (on ? ` · + ${txt} possible` : "");
  const tiles = [
    { c: STATUS.cut_off?.color || "#C8261C", shape: "dot", label: "Cut off from a hospital", v: fmt.n(acc.settlements_cut_off || 0),
      sub: `settlements${plus(fmt.n(acc.settlements_likely_cut_off || 0), acc.settlements_likely_cut_off)}`, tab: "access" },
    { c: C.building, shape: "sq", label: "Buildings hit", v: fmt.n(d.buildings_likely_hit || 0),
      sub: `confirmed${plus(fmt.n(d.buildings_possibly_hit || 0), d.buildings_possibly_hit)}`, tab: "summary" },
    { c: C.road, shape: "bar", label: "Roads affected", v: `${fmt.n(rc, 1)} km`,
      sub: `${fmt.n(bc)} bridge${bc === 1 ? "" : "s"} confirmed${plus([rp >= 0.05 ? `${fmt.n(rp, 1)} km` : "", bp ? `${fmt.n(bp)} bridge${bp === 1 ? "" : "s"}` : ""].filter(Boolean).join(", "), rp >= 0.05 || bp)}`, tab: "access" },
    { c: C.debris, shape: "sq", label: "Flood water + debris", v: fmt.km2(fc + dc),
      sub: `confirmed${plus(fmt.km2(pw), pw >= 0.005)}`, tab: "compare" },
  ];
  return (
    <section className="kpis" aria-label="Key numbers">
      {tiles.map((t) => (
        <button key={t.label} className="kpi" onClick={() => setTab(t.tab)}>
          <span className="kpi-l"><i className={t.shape} style={{ background: t.c }} aria-hidden="true" />{t.label}</span>
          <span className="kpi-v">{t.v}</span>
          <span className="kpi-s">{t.sub}</span>
        </button>
      ))}
    </section>
  );
}

// Is this map checked against an independent reference? Shown on every result, good or bad.
function TrustStrip({ R, setTab }) {
  const V = R.validation;
  const K = R.label_check;
  if (!V && K) {
    return (
      <button className="trust ok" onClick={() => setTab("check")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" strokeWidth="2.5" /></svg>
        <span><b>Checked against {K.reference}:</b>{" "}
          {pct(K.precision)} of what we flag is labelled flood (overlap IoU {(K.iou ?? 0).toFixed(2)})
          {K.model_role && /^(train|val)/i.test(K.model_role) ? `. This event was in the AI model's ${/^val/i.test(K.model_role) ? "validation" : "training"} set, so it is not an unseen test` : ""}.</span>
      </button>
    );
  }
  if (!V) {
    return (
      <button className="trust none" onClick={() => setTab("check")}>
        No official reference map is loaded for this area. <u>See the areas we checked</u>
      </button>
    );
  }
  const c = V.flood_extent || {};
  const latest = new Set(V.combined_from || []);
  let num = 0, den = 0;
  for (const p of V.products || []) {
    const r = p.roads || {};
    if ((latest.size === 0 || latest.has(p.product)) && r.closure_precision != null && r.our_closed_km_in_aoi) {
      num += r.closure_precision * r.our_closed_km_in_aoi; den += r.our_closed_km_in_aoi;
    }
  }
  return (
    <button className="trust ok" onClick={() => setTab("check")}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" strokeWidth="2.5" /></svg>
      <span><b>Checked against {String(V.reference || "a reference map").split(".")[0]}:</b>{" "}
        {pct(c.precision)} of what we flag is confirmed (overlap IoU {(c.iou ?? 0).toFixed(2)})
        {den > 0 ? `; ${pct(num / den)} of our road closures confirmed` : ""}. <u>See validation</u></span>
    </button>
  );
}

// Plain name for an EMS product key, e.g. "AOI01_GRA_MONIT01_v1" -> "EMS area 1, damage map, update 1".
const EMS_KIND = { GRA: "damage map", DEL: "flood outline", FEP: "first estimate" };
export function emsName(key) {
  const m = /AOI0*(\d+)_([A-Z]+)_(PRODUCT|MONIT0*(\d+))/.exec(String(key || ""));
  if (!m) return String(key || "").replace(/_/g, " ");
  const kind = EMS_KIND[m[2]] || m[2];
  return `EMS area ${m[1]}, ${kind}${m[4] ? `, update ${m[4]}` : ", first version"}`;
}

// The cut-off settlements to reach first, as cards; the full list is in the Cut off tab.
function ReachFirst({ R, ctl, setTab }) {
  const rows = (R.settlements || []).filter((r) => r.status === "cut_off" || r.status === "likely_cut_off");
  if (!rows.length) return null;
  const heli = new Map((R.helicopter_access?.per_settlement || []).map((p) => [p.name, p]));
  const top = rows.slice(0, 5);
  return (
    <section className="reach">
      <h2 className="h3">Who to reach first</h2>
      <p className="hint">Ranked by certainty of the cut, then size. Click to find it on the map.</p>
      {top.map((r, i) => {
        const h = heli.get(r.name);
        const air = h?.helipad ? `helipad ${fmt.n(h.helipad.km, 1)} km` : h?.zone ? `open ground ${fmt.n(h.zone.km, 1)} km` : null;
        return (
          <button key={`${r.name}-${i}`} className="reach-card" onClick={() => ctl.current.openSettlement(r.name)}>
            <i style={{ background: STATUS[r.status]?.color }} aria-hidden="true" />
            <span className="rc-main"><b>{r.name}</b>
              <small>{r.buildings ? `~${fmt.n(r.buildings)} buildings` : r.place}{r.target_before ? ` · usual hospital: ${r.target_before}` : ""}
                {air ? ` · ${air}` : ""}</small></span>
            <span className="rc-tag" style={{ color: STATUS[r.status]?.color }}>{r.status === "cut_off" ? "CUT OFF" : "LIKELY"}</span>
          </button>
        );
      })}
      {rows.length > top.length && (
        <button className="linkbtn" onClick={() => setTab("access")}>All {fmt.n(rows.length)} cut-off and likely cut-off settlements →</button>
      )}
    </section>
  );
}

function Summary({ R, visible, setVisible, setTab, setCompare }) {
  const a = R.areas, d = R.damage || {}, acc = R.access || {};
  const only = (k) => () => setVisible((v) => ({ ...v, [k]: !v[k] }));
  const notes = (R.notes || []).filter((n) => !n.startsWith("DEMO MODE") && !n.startsWith("OpenStreetMap snapshot"));
  return (
    <>
      <h2 className="h3">Where the flood hit</h2>
      <div className="ledger">
        <Row sw={C.water} label="New flood water, confirmed" value={fmt.km2(a.flood_high_conf_km2)} sub={`+ ${fmt.km2(Math.max((a.flood_km2 || 0) - (a.flood_high_conf_km2 || 0), 0))} possible (seen by one source only)`} onClick={only("flood")} active={visible.flood} />
        <Row sw={C.debris} label="Debris and mud, confirmed" value={fmt.km2(a.debris_high_conf_km2)} sub={`+ ${fmt.km2(Math.max((a.debris_km2 || 0) - (a.debris_high_conf_km2 || 0), 0))} possible (seen by one source only)`} onClick={only("flood")} active={visible.flood} />
        <Row sw={C.blind} label="Radar could not see" value={`${Math.round(100 * (a.nodata_fraction || 0))}%`} sub="steep slopes in radar shadow: unknown, not safe" />
        {R.inputs?.model?.vs_rules && (
          <Row sw="#0A28A0" label="AI model on this area" value={`+${fmt.km2(R.inputs.model.vs_rules.added_to_map_km2)}`}
            sub={`water added that the radar rules missed; ${fmt.km2(R.inputs.model.vs_rules.confidence_raised_km2)} confirmed. See the AI model tab`}
            onClick={() => setTab("ai")} />
        )}
      </div>

      <h2 className="h3">What was damaged</h2>
      <div className="ledger">
        <Row sw={C.building} label="Buildings likely hit" value={fmt.n(d.buildings_likely_hit)}
          sub={`${fmt.n(d.buildings_possibly_hit)} more possibly hit, of ${fmt.n(d.buildings_total)} mapped`} onClick={only("buildings")} active={visible.buildings} />
        <Row sw={C.road} label="Road affected, confirmed" value={`${fmt.n(d.road_km_confirmed ?? d.road_km_affected, 1)} km`} sub={`+ ${fmt.n(Math.max((d.road_km_affected || 0) - (d.road_km_confirmed ?? d.road_km_affected ?? 0), 0), 1)} km possible; ${fmt.n(d.road_segments_hit)} road segments in all`} onClick={only("roads")} active={visible.roads} />
        <Row sw={C.bridge} label="Bridges hit, confirmed" value={fmt.n(d.bridges_confirmed ?? d.bridges_hit)} sub={`+ ${fmt.n(Math.max((d.bridges_hit || 0) - (d.bridges_confirmed ?? d.bridges_hit ?? 0), 0))} possible, of ${fmt.n(d.bridges_total)} bridges in the area`} onClick={only("roads")} active={visible.roads} />
      </div>

      <h2 className="h3">Who is cut off</h2>
      <div className="ledger">
        <Row sw={STATUS.cut_off.color} label="Cut off from a hospital" value={fmt.n(acc.settlements_cut_off)}
          sub={acc.buildings_in_cut_off_settlements ? `about ${fmt.n(acc.buildings_in_cut_off_settlements)} buildings` : "blocked by confident detections"} onClick={() => setTab("access")} />
        <Row sw={STATUS.likely_cut_off.color} label="Likely cut off" value={fmt.n(acc.settlements_likely_cut_off || 0)} sub="blocked only by unconfirmed detections (e.g. debris seen by one radar pass)" onClick={() => setTab("access")} />
        <Row sw={STATUS.unknown.color} label="Unknown" value={fmt.n(acc.settlements_unknown || 0)} sub="last route crosses roads the radar could not see" onClick={() => setTab("access")} />
        <Row sw={STATUS.detour.color} label="Long detour" value={fmt.n(acc.settlements_detour)} onClick={() => setTab("access")} />
        <Row sw={STATUS.reachable.color} label="Road still fine" value={fmt.n(acc.settlements_reachable)} />
        <Row sw={STATUS.isolated_before.color} label="No mapped road even before" value={fmt.n((acc.settlements_isolated_before || 0) + (acc.settlements_no_road_data || 0))} sub="gaps in OpenStreetMap, not flood damage" />
      </div>

      {R.helicopter_access && (() => {
        const H = R.helicopter_access;
        const per = H.per_settlement || [];
        const reach = per.filter((x) => x.helipad || x.zone).length;
        return (
          <>
            <h2 className="h3">Helicopter access</h2>
            <button className={`btn small ${visible.landing ? "primary" : "secondary"}`} onClick={() => setVisible((v) => ({ ...v, landing: !v.landing }))}>
              {visible.landing ? "Hide helicopter landing areas on the map" : "See helicopter landing areas on the map"}
            </button>
            <div className="ledger">
              <Row sw="#0F4C75" label="Cut-off settlements with a landing option nearby" value={`${fmt.n(reach)} of ${fmt.n(per.length)}`}
                sub={`mapped helipad or flat open ground within ${H.criteria.max_distance_km} km`} onClick={() => setVisible((v) => ({ ...v, landing: !v.landing }))} active={visible.landing} />
              <Row sw="#0F4C75" label="Mapped helipads" value={fmt.n(H.helipads.length)} sub={`${fmt.n(H.helipads.filter((h) => h.usable).length)} with no damage detected`} />
              <Row sw="#0F4C75" label="Candidate landing grounds" value={H.zones_searched === false ? "not searched" : fmt.n(H.zones_total)}
                sub={H.zones_searched === false ? "No settlement needs air access here, so open ground was not searched (it is looked for only near cut-off settlements)." : `slope ≤ ${H.criteria.max_slope_deg}° (US Army landing guidance), ≥ ${H.criteria.clear_diameter_m} m clear, away from detected flood, mapped buildings and power lines. A place for a pilot to look at, not a certified site. Not checked: ${H.not_checked}`} />
            </div>
          </>
        );
      })()}

      <div className="row gap">
        <button className="btn primary" onClick={() => setTab("compare")}>Compare before and after</button>
        {R.reports?.en && <a className="btn secondary" href={fileUrl(R.run_id, R.reports.en)} target="_blank" rel="noreferrer">Open situation report</a>}
      </div>

      {notes.length > 0 && (
        <section className="notes">
          <h2 className="h3">Read before using these numbers</h2>
          <ul>{notes.map((n) => <li key={n}>{n}</li>)}</ul>
        </section>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------ access
function Access({ R, ctl }) {
  const rows = R.settlements || [];
  const counts = useMemo(() => rows.reduce((m, r) => ({ ...m, [r.status]: (m[r.status] || 0) + 1 }), {}), [rows]);
  const [pick, setPick] = useState(() => new Set(["cut_off", "likely_cut_off", "unknown", "detour"].filter((s) => counts[s])));
  const order = Object.keys(STATUS);
  // the server already ranks by response priority (status, then size, then delay)
  const shown = rows.filter((r) => pick.size === 0 || pick.has(r.status));
  const toggle = (s) => setPick((p) => { const n = new Set(p); n.has(s) ? n.delete(s) : n.add(s); return n; });
  const csv = () => {
    const head = ["priority", "name", "type", "status", "buildings", "hospital_before", "road_km_before", "road_km_now",
      "extra_km", "minutes_before", "minutes_now", "extra_minutes", "hospital_now"];
    const lines = [head.join(",")].concat(rows.map((r) =>
      [r.priority || "", r.name, r.place, STATUS[r.status]?.label || r.status, r.buildings, r.target_before, r.km_before,
        r.km_after, r.extra_km, r.min_before, r.min_after, r.extra_min, r.target_after]
        .map((v) => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")));
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url; a.download = `${R.run_id}_settlements.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  if (!rows.length) return <p className="muted">No settlements are mapped in OpenStreetMap for this area.</p>;
  return (
    <>
      <p className="lede small">Shortest road route from each settlement to the nearest {R.access?.target_kind || "hospital"}, on the road network
        as mapped before the flood, with flooded and debris-covered roads removed. Ranked by response priority: certainty of
        the cut, then settlement size (mapped buildings), then extra travel time (nominal speeds per road class).
        Click a name to find it on the map.</p>
      <div className="chips" role="group" aria-label="Filter by status">
        {order.filter((s) => counts[s]).map((s) => (
          <button key={s} className={pick.has(s) ? "on" : ""} style={{ "--c": STATUS[s].color }} onClick={() => toggle(s)} aria-pressed={pick.has(s)}>
            <i />{STATUS[s].label} <b>{counts[s]}</b>
          </button>
        ))}
      </div>
      <ul className="places">
        {shown.map((r, i) => (
          <li key={`${r.name}-${i}`}>
            <button onClick={() => ctl.current.openSettlement(r.name)}>
              <i style={{ background: STATUS[r.status]?.color }} aria-hidden="true" />
              <span className="p-main"><b>{r.priority ? `#${r.priority} ` : ""}{r.name}</b>
                <small>{r.place}{r.buildings ? `, ~${fmt.n(r.buildings)} buildings` : ""}{r.target_before ? `, nearest: ${r.target_before}` : ""}</small></span>
              <span className="p-km">
                {["cut_off", "likely_cut_off"].includes(r.status) ? <>{r.km_before ?? "–"} km <s aria-hidden="true" /><em>no route</em></>
                  : r.status === "detour" ? <>{r.km_before} → {r.km_after} km<em>+{r.extra_min != null ? `${r.extra_min} min` : `${r.extra_km} km`}</em></>
                    : r.status === "unknown" ? <>{r.km_before} km<em>not visible</em></>
                      : r.km_before != null ? <>{r.km_before} km</> : "–"}
              </span>
            </button>
          </li>
        ))}
      </ul>
      <button className="btn secondary small" onClick={csv}>Download the list (CSV)</button>
    </>
  );
}

// ------------------------------------------------------------------------------------------------ compare
function Compare({ R, compare, setCompare }) {
  const w = R.web || {};
  const z = w.zoom;
  const pair = R.inputs.sentinel1_pairs[0];
  const modes = [["change", "Change (easiest)"], ["after", "After"], ["flood", "Detected damage"]].filter(([k]) => w[k]);
  // the swipe is part of this tab: open it on arrival (change view: differences in colour), close it on leaving
  useEffect(() => {
    if (!compare && w.before && w.after) setCompare({ right: w.change ? "change" : "after", frac: 0.5 });
    return () => setCompare(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const st = z?.stats || {};
  const dB = (v) => `${v > 0 ? "+" : v < 0 ? "\u2212" : ""}${Math.abs(v)}\u00a0dB`;     // signed, true minus sign
  const lvl = (v) => `${v < 0 ? "\u2212" : ""}${Math.abs(v)}\u00a0dB`;
  const d = st.inside_change_db;
  const lb = [`Before, ${fmt.date(pair.pre_date)}`, `After, ${fmt.date(pair.post_date)}`];
  return (
    <>
      <div className="readhelp">
        <h2 className="h3">How to read radar images</h2>
        <ul>
          <li><b>Black / dark</b>: smooth surfaces that bounce the radar away, such as <b>water</b> or wet mud.</li>
          <li><b>White / bright</b>: rough surfaces, such as <b>boulders, debris, rubble</b>, buildings, or slopes facing the satellite.</li>
          <li>Both images come from the same orbit track ({pair.track}), so mountains, shadows and towns look the same in both.
            <b> Only what really changed looks different.</b></li>
          <li>Radar is not a photo: changes are often a few shades of grey. The colour <b>Change</b> view and <b>Flicker</b> make them easy to see.</li>
        </ul>
      </div>

      <h2 className="h3">Swipe on the map</h2>
      <p className="hint seg-lead">Left of the handle: before. Right of the handle:</p>
      <div className="seg wide">
        {modes.map(([k, label]) => (
          <button key={k} className={compare?.right === k ? "on" : ""} onClick={() => setCompare({ right: k, frac: 0.5 })}>{label}</button>
        ))}
      </div>
      {compare && <p className="hint">Drag the white handle on the map, and zoom in on the river.</p>}
      {w.change && (
        <p className="legend-inline">
          <span><i style={{ background: "#EE291F" }} />Red: darker after (new water, smoothing)</span>
          <span><i style={{ background: "#0DD9F2" }} />Cyan: brighter after (debris, rubble)</span>
          <span><i style={{ background: "#9AA3A8" }} />Grey: no real change</span>
        </p>
      )}

      {z && (
        <>
          <h2 className="h3">Close-up of the largest detected change</h2>
          <p className="hint">{z.size_km[0]} × {z.size_km[1]} km, north up, full 20 m detail. <b style={{ color: "#B58B00" }}>Yellow line</b>: what
            FloodSight detected, so you know where to look. Contrast is stretched on this close-up (same scale for both dates).</p>
          {d != null && (
            <p className="callout what-changed">
              Inside the yellow line the radar return changed by <b>{dB(d)}</b>
              {d <= -1.5 ? " (darker: typical of standing water or wet mud)" : d >= 1.5 ? " (brighter: typical of debris, boulders or rubble)" : ""};
              outside it by {dB(st.outside_change_db)} (no real change).
              {st.debris_px ? ` Where we found debris: ${lvl(st.debris_before_db)} before, ${lvl(st.debris_after_db)} after (brighter).` : ""}
              {st.flood_px ? ` Where we found new water: ${lvl(st.flood_before_db)} before, ${lvl(st.flood_after_db)} after (darker).` : ""}
              <small> (3 dB = twice / half as much radar echo.)</small>
            </p>
          )}
          <BeforeAfter before={fileUrl(R.run_id, z.before)} after={fileUrl(R.run_id, z.after)} labels={lb}
            caption="Radar (Sentinel-1). Dark = water or smooth ground, bright = rough ground or debris." />
          {z.opt_before && (
            <>
              <h2 className="h3">Same place in Sentinel-2 (optical)</h2>
              <p className="hint">False colour, closer to what the eye understands: <b>green</b> = vegetation, <b>pink-brown</b> = bare soil,
                mud or debris, <b>black</b> = water. Hatched grey: clouds, no clear view.</p>
              <BeforeAfter before={fileUrl(R.run_id, z.opt_before)} after={fileUrl(R.run_id, z.opt_after)} labels={lb}
                caption="Sentinel-2 false colour (SWIR, near-infrared, red). Earliest clear view after the flood." />
            </>
          )}
          <figure className="fig">
            <ZoomImg src={fileUrl(R.run_id, z.change)} alt="Radar change with FloodSight detection outlined in yellow"
              caption="Change between the two radar images. Red: darker after. Cyan: brighter after. Yellow line: what FloodSight detected." />
            <figcaption>Change in one picture: <b style={{ color: "#C8261C" }}>red</b> got darker (water),
              <b style={{ color: "#0B9FB3" }}> cyan</b> got brighter (debris), grey did not change. Yellow line: FloodSight detection.</figcaption>
          </figure>
        </>
      )}
    </>
  );
}

// Before/after viewer: slide, flicker (alternates automatically: changes jump out) or side by side.
function BeforeAfter({ before, after, labels, caption }) {
  const [mode, setMode] = useState("flicker");
  const [split, setSplit] = useState(50);
  const [showAfter, setShowAfter] = useState(false);
  useEffect(() => {
    if (mode !== "flicker") return undefined;
    const t = setInterval(() => setShowAfter((v) => !v), 900);
    return () => clearInterval(t);
  }, [mode]);
  return (
    <div className="ba">
      <div className="seg ba-modes" role="group" aria-label="How to compare">
        {[["flicker", "Flicker"], ["slide", "Slide"], ["side", "Side by side"]].map(([k, l]) => (
          <button key={k} className={mode === k ? "on" : ""} onClick={() => setMode(k)}>{l}</button>
        ))}
        <button onClick={() => zoom({ before, after, labels, caption })}>Enlarge</button>
      </div>
      {mode === "flicker" && (
        <div className="ba-flick" onClick={() => setMode("slide")} title="Click to stop and slide">
          <img src={before} alt={labels[0]} style={{ opacity: showAfter ? 0 : 1 }} />
          <img className="top" src={after} alt={labels[1]} style={{ opacity: showAfter ? 1 : 0 }} />
          <span className={`cap big ${showAfter ? "after" : ""}`}>{showAfter ? labels[1] : labels[0]}</span>
        </div>
      )}
      {mode === "slide" && (
        <div className="imgswipe" style={{ "--split": `${split}%` }}>
          <img src={before} alt={labels[0]} />
          <img className="top" src={after} alt={labels[1]} />
          <span className="cap l">{labels[0]}</span><span className="cap r">{labels[1]}</span>
          <input type="range" min="0" max="100" value={split} onChange={(e) => setSplit(Number(e.target.value))} aria-label="Slide between before and after" />
        </div>
      )}
      {mode === "side" && (
        <div className="ba-side">
          <figure><img src={before} alt={labels[0]} /><figcaption>{labels[0]}</figcaption></figure>
          <figure><img src={after} alt={labels[1]} /><figcaption>{labels[1]}</figcaption></figure>
        </div>
      )}
      <p className="hint">{mode === "flicker" ? "The picture switches between the two dates every second: whatever jumps is what changed. " : ""}{caption}</p>
    </div>
  );
}

// ------------------------------------------------------------------------------------------------ AI model
// What the AI did on THIS area and date. How the model was tested in general is on its own page (ModelPage).
export const openModelPage = () => window.dispatchEvent(new Event("fs-model"));

function AIModel({ R, visible, setVisible }) {
  const [card, setCard] = useState(null);
  useEffect(() => { api.modelCard().then(setCard).catch(() => setCard(false)); }, []);
  const m = R.inputs?.model || {};
  const v = m.vs_rules;
  const hasLayer = !!R.web?.model;
  const pctf = (x) => (x == null ? "–" : `${Math.round(100 * x)}%`);
  const nTests = (card?.tests?.length || 0) + (card?.second_dataset ? 1 : 0);
  let verdict = null;
  if (v) {
    const tot = v.both_km2 + v.model_only_km2 + v.rules_only_km2;
    if (tot < 0.01) verdict = "Neither the AI nor the radar rules found open flood water here. Damage on this area, if any, is debris, which the AI was not trained for (the debris map comes from the radar rules and Sentinel-2).";
    else if (v.agreement_iou >= 0.5) verdict = `The AI and the radar rules largely agree here (overlap ${pctf(v.agreement_iou)}): two independent methods see the same water, so those detections are more reliable.`;
    else if (v.model_only_km2 > v.rules_only_km2) verdict = "The AI sees more water than the radar rules here. Only the part on low, flat ground where water can pond is added to the map, at low confidence.";
    else verdict = "The radar rules find more water than the AI here (common in mountains and on small rivers, which the AI saw little of in training). The AI then mainly confirms or questions the rules' detections.";
  }
  const views = [["model", "AI alone"], ["rules", "Radar rules alone"], ["flood", "Final map"]].filter(([k]) => k !== "rules" || R.web?.rules);
  const only = (k) => setVisible((s) => ({ ...s, model: k === "model", rules: k === "rules", flood: k === "flood" }));
  const shown = views.find(([k]) => visible[k] && views.every(([o]) => o === k || !visible[o]))?.[0];

  return (
    <>
      {!m.used ? (
        <p className="callout">The AI model was not used in this run{(R.notes || []).find((x) => x.includes("model")) ? `: ${(R.notes || []).find((x) => x.includes("model"))}` : "."}</p>
      ) : v ? (
        <p className="ai-head">The AI marked <b>{fmt.km2(v.both_km2 + v.model_only_km2)}</b> of this area as new flood water
          (probability ≥ {v.threshold ?? m.threshold ?? 0.8}).</p>
      ) : (
        <p className="hint">This run was made before the AI comparison was recorded; re-run it (downloads are reused) to see what the AI did here.</p>
      )}

      {v && (
        <>
          <h2 className="h3">How the AI changed the final map</h2>
          <section className="kpis ai-kpis" aria-label="How the AI changed the final map">
            <div className="kpi static">
              <span className="kpi-l"><i style={{ background: "#0A28A0" }} aria-hidden="true" />Agreed with radar rules</span>
              <span className="kpi-v">{fmt.km2(v.both_km2)}</span>
              <span className="kpi-s">confidence raised on {fmt.km2(v.confidence_raised_km2)}</span>
            </div>
            <div className="kpi static">
              <span className="kpi-l"><i style={{ background: "#3C64D2" }} aria-hidden="true" />Added by the AI</span>
              <span className="kpi-v">{fmt.km2(v.added_to_map_km2)}</span>
              <span className="kpi-s">only on low, flat ground; low confidence</span>
            </div>
            <div className="kpi static">
              <span className="kpi-l"><i style={{ background: C.water }} aria-hidden="true" />Questioned by the AI</span>
              <span className="kpi-v">{fmt.km2(v.confidence_lowered_km2)}</span>
              <span className="kpi-s">radar-rule water the AI disagrees with; confidence lowered</span>
            </div>
          </section>
          {verdict && <p className="callout">{verdict}</p>}
        </>
      )}

      {hasLayer && (
        <>
          <h2 className="h3">See it on the map</h2>
          <div className="seg" role="group" aria-label="Map view">
            {views.map(([k, l]) => (
              <button key={k} className={shown === k ? "on" : ""} aria-pressed={shown === k} onClick={() => only(k)}>{l}</button>
            ))}
          </div>
          <p className="hint">AI alone: blue shading is the AI's flood probability before it is combined with anything (light = maybe, dark = very likely).</p>
        </>
      )}

      <FusionDiagram />

      {v?.note && <p className="fine">{v.note} Threshold chosen on the validation event only.</p>}

      <button className="trust ok ai-tested" onClick={openModelPage}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" strokeWidth="2.5" /></svg>
        <span><b>How the model was tested</b>{nTests ? `: ${nTests} tests on data it never trained on, incl. a Himalayan flash flood and a second dataset, and ${Math.max((card?.comparison?.rows?.length || 1) - 1, 0)} alternative models.` : "."} <u>Open</u></span>
      </button>
    </>
  );
}

// How the parts combine: why "AI alone" is not the final map.
function FusionDiagram() {
  const ins = [["AI model", "flood-water probability", "#0A28A0"], ["Radar rules", "water and debris", C.water],
    ["Sentinel-2", "second sensor, clear sky", "#2E7D32"], ["Terrain (DEM)", "where water and debris can be", "#7A5C2E"]];
  return (
    <figure className="fusion" aria-label="How FloodSight combines its parts">
      <svg viewBox="0 0 580 190" role="img">
        <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="var(--ink)" /></marker></defs>
        {ins.map(([t, s, c], i) => {
          const y = 8 + i * 45;
          return (
            <g key={t}>
              <path d={`M200 ${y + 18} C 245 ${y + 18}, 245 95, 290 95`} fill="none" stroke={c} strokeWidth="2" />
              <rect y={y} width="200" height="36" rx="6" fill="var(--card)" stroke={c} strokeWidth="2" />
              <text x="12" y={y + 16} fontSize="13" fontWeight="700" fill="var(--ink)">{t}</text>
              <text x="12" y={y + 30} fontSize="11" fill="var(--ink-3)">{s}</text>
            </g>
          );
        })}
        <rect x="290" y="68" width="130" height="54" rx="6" fill="var(--ink)" />
        <text x="355" y="90" fontSize="13" fontWeight="700" fill="var(--paper)" textAnchor="middle">Combine</text>
        <text x="355" y="107" fontSize="10.5" fill="var(--paper)" textAnchor="middle">agree = more confident</text>
        <path d="M420 95 L 455 95" stroke="var(--ink)" strokeWidth="2.5" markerEnd="url(#arr)" />
        <rect x="460" y="68" width="115" height="54" rx="6" fill="var(--card)" stroke="var(--ink)" strokeWidth="2" />
        <text x="517" y="90" fontSize="13" fontWeight="700" fill="var(--ink)" textAnchor="middle">Final map</text>
        <text x="517" y="107" fontSize="10.5" fill="var(--ink-3)" textAnchor="middle">with confidence</text>
      </svg>
      <figcaption>The AI is one of four sources. Where it agrees with the radar rules, confidence goes up; it adds water only where
        water can pond; debris comes from the radar rules and Sentinel-2. That is why "AI alone" and the final map differ.</figcaption>
    </figure>
  );
}

// ------------------------------------------------------------------------------------------------ flow path
function Profile({ fp }) {
  const km = fp.profile?.km || [], z = fp.profile?.elev_m || [];
  if (km.length < 2) return null;
  const W = 340, H = 120, P = 26;
  const k1 = km[km.length - 1] || 1, zmin = Math.min(...z), zmax = Math.max(...z), zr = Math.max(zmax - zmin, 1);
  const X = (k) => P + (k / k1) * (W - P - 6), Y = (e) => 6 + (1 - (e - zmin) / zr) * (H - P - 6);
  const d = km.map((k, i) => `${i ? "L" : "M"}${X(k).toFixed(1)},${Y(z[i]).toFixed(1)}`).join("");
  return (
    <figure className="profile">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Elevation along the flood path">
        <path d={`${d}L${X(k1)},${H - P}L${P},${H - P}Z`} fill={C.flow} opacity="0.12" />
        <path d={d} fill="none" stroke={C.flow} strokeWidth="2" />
        {(fp.settlements || []).map((s, i) => <line key={i} x1={X(s.km_downstream)} x2={X(s.km_downstream)} y1={H - P} y2={H - P - 8} stroke="#17242E" />)}
        <text x={P} y={H - 8} fontSize="10">0 km</text>
        <text x={W - 6} y={H - 8} fontSize="10" textAnchor="end">{fmt.n(k1, 0)} km</text>
        <text x={2} y={12} fontSize="10">{fmt.n(zmax)} m</text>
        <text x={2} y={H - P} fontSize="10">{fmt.n(zmin)} m</text>
      </svg>
      <figcaption>Elevation along the path; ticks mark settlements. Where the line flattens, a debris flow slows and deposits.</figcaption>
    </figure>
  );
}

function Flow({ R, ctl }) {
  const [start, setStart] = useState(null);
  const [km, setKm] = useState(80);
  const [fp, setFp] = useState(null);
  const [err, setErr] = useState(null);
  const [stage, setStage] = useState(null);
  const kmRef = useRef(km);
  kmRef.current = km;
  useEffect(() => {
    const el = ctl.current.map.getContainer();
    el.classList.add("is-picking");
    let alive = true;
    const off = ctl.current.onClick(async (ll) => {
      setStart(ll); setErr(null); setFp(null); setStage("Starting…");
      ctl.current.setFlowPath(null, ll);
      try {
        const st = await api.flowStart(R.run_id, ll.lng, ll.lat, kmRef.current);
        if (!st.reused) {
          for (;;) {
            await new Promise((res) => setTimeout(res, 1500));
            const j = await api.job(st.job.id);
            if (!alive) return;
            setStage(j.stage);
            if (j.state === "failed") throw new Error(j.error || "The path could not be traced.");
            if (j.state === "done") break;
          }
        }
        const out = await api.flow(st.key);
        if (!alive) return;
        setFp(out);
        ctl.current.setFlowPath(out, ll);
      } catch (e) { if (alive) setErr(e.message); }
      finally { if (alive) setStage(null); }
    });
    return () => { alive = false; off(); el.classList.remove("is-picking"); ctl.current.setFlowPath(null); };
  }, [R.run_id, ctl]);
  const s = fp?.settlements || [];
  const csv = () => {
    const head = ["km_downstream", "name", "type", "m_above_river", "m_from_river", "band", "damage_nearby", "lat", "lon"];
    const rows = [...s.map((x) => ({ ...x, type: x.place || "settlement" })), ...(fp.facilities || []).map((x) => ({ ...x, type: "health facility" })),
      ...(fp.bridges || []).map((x) => ({ ...x, type: "bridge" }))].sort((a, b) => a.km_downstream - b.km_downstream);
    const lines = [head.join(",")].concat(rows.map((x) => head.map((h) => `"${String(x[h] ?? "").replace(/"/g, '""')}"`).join(",")));
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = `${R.run_id}_flood_path.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <>
      <p className="lede small">Where would water or debris released here go? Click <b>any point on the map</b>, also outside the analysed box,
        for example a glacial lake or landslide dam upstream. The path follows the steepest descent on the Copernicus 30 m elevation model and
        continues down the valley; places along it come from OpenStreetMap as it was before the event.</p>
      <label className="field inline"><span>Trace up to</span>
        <select value={km} onChange={(e) => setKm(Number(e.target.value))}>
          {[25, 50, 80, 120].map((k) => <option key={k} value={k}>{k} km</option>)}
        </select></label>
      {!start && <p className="callout">Click anywhere on the map to start.</p>}
      {stage && <p className="muted">{stage}</p>}
      {err && <p className="error">{err}</p>}
      {fp && (
        <>
          <div className="ledger">
            <Row sw={C.flow} label="Path length" value={`${fmt.n(fp.length_km, 1)} km`} sub={`drops ${fmt.n(fp.drop_m)} m; ${fp.stop_reason}`} />
            <Row sw={C.flow} label="Settlements near the river" value={fmt.n(s.filter((x) => !String(x.band).startsWith(">")).length)}
              sub={`within ${fmt.n(fp.max_distance_m)} m and ${fp.height_bands_m?.slice(-1)[0]} m above it; ${fmt.n(s.filter((x) => String(x.band).startsWith(">")).length)} more on higher ground are listed too`} />
            <Row sw={C.health} label="Health facilities" value={fmt.n(fp.facilities?.length || 0)} />
            <Row sw={C.bridge} label="Bridges over the path" value={fmt.n(fp.bridges?.length || 0)} />
            {fp.path_on_mapped_river != null && <Row sw={C.ok} label="Check: path on a mapped river" value={`${Math.round(100 * fp.path_on_mapped_river)}%`}
              sub={`of its length within 150 m of a river mapped in OpenStreetMap (independent of the elevation model; dashed on the map where not)${
                fp.rivers_followed?.length ? ". Follows: " + fp.rivers_followed.map((r) => `${r.name} (km ${r.from_km}-${r.to_km})`).join(", ") : ""}`} />}
          </div>
          <Profile fp={fp} />
          {s.length > 0 && (
            <ul className="places">
              {s.map((f, i) => (
                <li key={i}><button onClick={() => ctl.current.flyTo(f.lat, f.lon)}>
                  <i style={{ background: C.flow, opacity: f.band === "<= 10 m" ? 1 : f.band === "<= 25 m" ? 0.7 : 0.45 }} />
                  <span className="p-main"><b>{f.name}{f.name_en && f.name_en !== f.name ? ` (${f.name_en})` : ""}</b>
                    <small>{f.m_above_river ?? "?"} m above the river, {f.m_from_river} m from it{f.damage_nearby ? ` · ${f.damage_nearby}` : ""}</small></span>
                  <span className="p-km">{f.km_downstream} km<em>{f.band}</em></span>
                </button></li>
              ))}
            </ul>
          )}
          <p className="hint">Shading on the map: valley floor up to 10, 25 and 50 m above the river (exposure levels, not a flood-depth
            prediction). {fp.note}</p>
          <div className="row gap">
            <button className="btn secondary small" onClick={csv}>Download the list (CSV)</button>
            <a className="btn secondary small" href={`/api/flows/${fp.key || ""}`} target="_blank" rel="noreferrer" hidden={!fp.key}>JSON</a>
          </div>
          <p className="hint">Click again to try another start point.</p>
        </>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------------ report
function Report({ R }) {
  const rep = R.reports || {};
  const f = (rel) => fileUrl(R.run_id, rel);
  const vec = R.vectors || {};
  return (
    <>
      <p className="lede small">A one-page situation report for responders, generated from this analysis. Every number is copied from the
        results by a fixed template; no language model writes or changes numbers.</p>
      <div className="row gap">
        {rep.en && <a className="btn primary" href={f(rep.en)} target="_blank" rel="noreferrer">English report (PDF)</a>}
        {rep.ne && <a className="btn secondary" href={f(rep.ne)} target="_blank" rel="noreferrer" lang="ne">नेपाली प्रतिवेदन (PDF)</a>}
      </div>
      {rep.en && (
        <object className="pdf" data={f(rep.en)} type="application/pdf" aria-label="Situation report preview">
          {rep.map && <ZoomImg className="pdf-fallback" src={f(rep.map)} alt="Map from the situation report" caption="Map from the situation report" />}
        </object>
      )}
      {(vec.health_csv || vec.priority_csv) && (
        <>
          <h2 className="h3">Lists for responders (open in Excel)</h2>
          <div className="row gap">
            {vec.priority_csv && <a className="btn secondary" href={f(vec.priority_csv)} download>Settlements and road access (CSV)</a>}
            {vec.health_csv && <a className="btn secondary" href={f(vec.health_csv)} download>Hospitals and clinics with locations (CSV)</a>}
          </div>
          <p className="hint">The report map shows only the hospitals that settlements with lost or changed access depend on{R.health_facilities_total ? ` (the area has ${fmt.n(R.health_facilities_total)})` : ""}; the full list is in the CSV.</p>
          {!rep.ne && <p className="fine">Nepali report: made only for areas in Nepal.</p>}
        </>
      )}
      <h2 className="h3">Data for GIS</h2>
      <ul className="files">
        <li><a href={f("results.json")} download>results.json</a><small>every number, inputs, caveats, timings</small></li>
        <li><a href={f("layers/flood_class.tif")} download>flood_class.tif</a><small>0 none, 1 water, 2 debris, 3 river, 255 not visible</small></li>
        <li><a href={f("layers/confidence.tif")} download>confidence.tif</a><small>1 low to 3 high</small></li>
        {Object.entries(vec).map(([k, rel]) => <li key={k}><a href={f(rel)} download>{rel.split("/").pop()}</a><small>GeoJSON</small></li>)}
      </ul>
    </>
  );
}

// ------------------------------------------------------------------------------------------------ validation
const pct = (x) => (x == null ? "–" : `${Math.round(100 * x)}%`);

function Check({ R }) {
  const V = R.validation;
  const K = R.label_check;
  if (!V && K) {
    return (
      <button className="trust ok" onClick={() => setTab("check")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" strokeWidth="2.5" /></svg>
        <span><b>Checked against {K.reference}:</b>{" "}
          {pct(K.precision)} of what we flag is labelled flood (overlap IoU {(K.iou ?? 0).toFixed(2)})
          {K.model_role && /^(train|val)/i.test(K.model_role) ? `. This event was in the AI model's ${/^val/i.test(K.model_role) ? "validation" : "training"} set, so it is not an unseen test` : ""}.</span>
      </button>
    );
  }
  if (!V) {
    return (
      <>
        <p className="lede small">Compare this analysis with an independent damage map, such as a Copernicus Emergency Management Service
          activation. Reference maps are only used for checking; they never feed the analysis.</p>
        <pre className="cmd">python -m floodsight.validate.ems --run runs/{R.run_id} --ems validation_data/EMSR927</pre>
        <p className="hint">Then reopen this analysis.</p>
      </>
    );
  }
  const c = V.flood_extent || {};
  const prods = (V.products || []).filter((p) => p.extent);
  return (
    <>
      <p className="lede small">{V.reference}. {V.what_ems_mapped ? `${V.what_ems_mapped}.` : ""}</p>
      <div className="metrics">
        <div><b>{pct(c.precision)}</b><span>of what we mapped, EMS mapped too (precision)</span></div>
        <div><b>{pct(c.recall)}</b><span>of what EMS mapped, we found (recall)</span></div>
        <div><b>{(c.iou ?? 0).toFixed(2)}</b><span>overlap, IoU</span></div>
        <div><b>{(c.f1 ?? 0).toFixed(2)}</b><span>F1</span></div>
      </div>
      {V.combined_from && <p className="hint">Combined over {V.combined_from.map(emsName).join("; ")} (latest map per area). Pixels the radar could not see are left out.</p>}
      <div className="tablewrap">
        <table>
          <thead><tr><th>EMS area</th><th>Precision</th><th>Recall</th><th>IoU</th><th>Found as debris</th><th>Buildings found</th><th>Damaged road flagged</th><th>Our road closures confirmed</th><th>Precision, confident / low</th></tr></thead>
          <tbody>
            {prods.map((p) => (
              <tr key={p.product}>
                <td>{emsName(p.product)}</td>
                <td>{pct(p.extent.precision)}</td><td>{pct(p.extent.recall)}</td><td>{p.extent.iou.toFixed(2)}</td>
                <td>{pct(Object.values(p.ems_event_types?.types || {})[0]?.detected_as_debris)}</td>
                <td>{pct(p.buildings?.recall_damaged)}</td><td>{pct(p.roads?.share_we_flagged)}</td>
                <td>{pct(p.roads?.closure_precision)}</td>
                <td>{pct(p.by_confidence?.high_medium?.precision)} / {pct(p.by_confidence?.low?.precision)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">"Our road closures confirmed": share of the road length we closed that EMS shows as damaged or inside its
        mapped flood/debris area (this is what the cut-off analysis rests on). "Precision, confident / low": how often
        high/medium-confidence and low-confidence detections agree with EMS, which shows whether the confidence levels mean something.</p>
      {V.figure && <figure className="fig"><ZoomImg src={fileUrl(R.run_id, V.figure)} alt="Agreement map per EMS area" caption="Agreement with the EMS map per area" />
        <figcaption>Green: both. Red: only FloodSight. Blue: only EMS. Grey: radar could not see.</figcaption></figure>}
      <p className="fine">{V.note}</p>
    </>
  );
}

// ------------------------------------------------------------------------------------------------ details
function Details({ R, onRerun }) {
  const t = Object.entries(R.timings || {}).filter(([k]) => k !== "total");
  const max = Math.max(...t.map(([, v]) => v), 1);
  const inp = R.inputs || {};
  return (
    <>
      <h2 className="h3">Satellite images used</h2>
      <div className="tablewrap">
        <table>
          <thead><tr><th>Track</th><th>Before</th><th>After</th><th>Gap</th><th>Coverage</th></tr></thead>
          <tbody>
            {inp.sentinel1_pairs.map((p) => (
              <tr key={p.track}><td>{p.track}</td><td>{p.pre_date}{p.pre_stack_dates?.length > 1 ? ` (reference: median of ${p.pre_stack_dates.length} images, back to ${p.pre_stack_dates[p.pre_stack_dates.length - 1]})` : ""}</td><td>{p.post_date}</td><td>{p.gap_days} days</td><td>{pct(p.coverage)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="facts">
        <li>Radar processing: {{ sentinelhub: "Sentinel Hub on Copernicus Data Space", openeo: "openEO on Copernicus Data Space",
          local: "on this computer from the Level-1 GRD file", synthetic: "synthetic demo data" }[inp.sentinel1_backend] || inp.sentinel1_backend}
          {inp.sentinel1_calibration ? `, calibration ${inp.sentinel1_calibration}` : ""}, Copernicus 30 m DEM
          {inp.sentinelhub_processing_units ? `, ${fmt.n(inp.sentinelhub_processing_units)} processing units` : ""}</li>
        <li>Sentinel-2 optical: {inp.sentinel2?.used
          ? `second sensor where clear (${Math.round(100 * (inp.sentinel2.clear_fraction || 0))}% of the area)` +
            (inp.sentinel2.added_flood_km2 != null ? `: added ${fmt.km2(inp.sentinel2.added_flood_km2)} water and ${fmt.km2(inp.sentinel2.added_debris_km2)} debris the radar did not show (low confidence), confirmed ${fmt.km2((inp.sentinel2.confirmed_flood_km2 || 0) + (inp.sentinel2.confirmed_debris_km2 || 0))} of radar detections` : "")
          : "not used (clouds or unavailable)"}</li>
        {inp.debris_terrain?.applied && <li>Debris class only in mountain terrain (UNEP-WCMC definition): {Math.round(100 * (inp.debris_terrain.mountain_fraction || 0))}% of this area</li>}
        <li>OpenStreetMap: snapshot of {fmt.date(inp.osm_snapshot)}, before the flood</li>
        <li>Segmentation model: {inp.model?.used ? `used${inp.model.threshold ? `, threshold ${inp.model.threshold}` : ""}` : "not used"}
          {inp.model?.input_speckle_matching?.window ? ` (input smoothed to the training data's graininess: ${inp.model.input_speckle_matching.before_db} → ${inp.model.input_speckle_matching.after_db} dB, Lee ${inp.model.input_speckle_matching.window}×${inp.model.input_speckle_matching.window})` : ""}</li>
        {(inp.detection_stats || []).map((d) => (
          <li key={d.track}>Water threshold {d.track}: {d.water_threshold_db} dB ({d.water_threshold_method}), found from this scene, not preset</li>
        ))}
      </ul>
      <h2 className="h3">Time per step</h2>
      <ul className="bars">
        {t.map(([k, v]) => (
          <li key={k}><span>{k}</span><i style={{ width: `${(100 * v) / max}%` }} /><b>{fmt.dur(v)}</b></li>
        ))}
      </ul>
      <p className="hint">Total {fmt.dur(R.timings?.total || 0)}. Re-running the same area reuses the downloaded images.</p>
      {R.data_requests?.length > 0 && (
        <RequestList title="Where the download time went" note items={R.data_requests.map((r, i) => ({ ...r, id: i, start: r.start_s }))} />
      )}
      <h2 className="h3">Sources and credits</h2>
      <ul className="credits">{(R.attributions || []).map((a) => <li key={a}>{a}</li>)}</ul>
      <p className="fine">Run {R.run_id}, FloodSight {R.floodsight_version}, generated {R.generated_utc}. Educational prototype, not an operational emergency service.</p>
      <button className="btn secondary small" onClick={onRerun}>Analyse another area or date</button>
    </>
  );
}
