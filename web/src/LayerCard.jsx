import React, { useEffect, useState } from "react";
import { BASEMAP_KEYS } from "./mapctl.js";
import { C, STATUS } from "./theme.js";

// The layer switches ARE the legend: each switch shows the colours that layer draws.
const GROUPS = [
  ["Satellite evidence", [
    ["flood", "Flood and debris", [[C.water, "New water"], [C.debris, "Debris, mud"], [C.permanent, "River, lake"], [C.blind, "Radar can't see"]]],
    ["change", "Radar change", [["#EE291F", "Darker after"], ["#0DD9F2", "Brighter after"]]],
    ["before", "Radar before", null],
    ["after", "Radar after", null],
    ["rules", "Radar rules alone (before AI and Sentinel-2)", [[C.water, "Water"], [C.debris, "Debris"]]],
    ["model", "AI model alone: flood water", [["#AAD2FF", "Maybe"], ["#3C64D2", "Likely"], ["#0A28A0", "Very likely"]]],
  ]],
  ["Damage", [
    ["buildings", "Buildings hit", [[C.building, "Likely"], [C.buildingMaybe, "Possibly"]]],
    ["roads", "Roads and bridges hit", [[C.road, "Road"], [C.bridge, "Bridge"]]],
    ["broken", "Road links cut", [[C.cut, "Cut"]]],
  ]],
  ["Access to hospitals", [
    ["settlements", "Settlements", Object.values(STATUS).slice(0, 4).map((s) => [s.color, s.label])],
    ["health", "Hospitals and clinics", [[C.health, "Facility"]]],
    ["landing", "Helicopter landing", [["#0F4C75", "Helipad (map)"], ["#0F4C75", "Flat open ground (candidate)"]]],
  ]],
];

export default function LayerCard({ visible, setVisible, basemap, setBasemap, ctl, collapse }) {
  const [open, setOpen] = useState(window.innerWidth > 900 && !collapse);
  useEffect(() => { if (collapse) setOpen(false); }, [collapse]);   // keep the swipe area clear
  const flip = (k) => setVisible((v) => ({ ...v, [k]: !v[k] }));
  return (
    <section className={`layercard ${open ? "open" : ""}`} aria-label="Map layers and legend">
      <button className="layercard-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span>Map layers</span>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d={open ? "M6 15l6-6 6 6" : "M6 9l6 6 6-6"} fill="none" stroke="currentColor" strokeWidth="2" /></svg>
      </button>
      {open && (
        <div className="layercard-body">
          {GROUPS.map(([title, items]) => (
            <fieldset key={title}>
              <legend>{title}</legend>
              {items.filter(([k]) => ctl.current?.hasLayer(k)).map(([k, label, sw]) => (
                <label key={k} className="lyr">
                  <input type="checkbox" checked={!!visible[k]} onChange={() => flip(k)} />
                  <span className="lyr-text">
                    <span className="lyr-name">{label}</span>
                    {sw && (
                      <span className="lyr-sw">
                        {sw.map(([c, t]) => <span key={t}><i style={{ background: c }} />{t}</span>)}
                      </span>
                    )}
                  </span>
                </label>
              ))}
            </fieldset>
          ))}
          <fieldset>
            <legend>Background</legend>
            <div className="seg">
              {BASEMAP_KEYS.map(([k, label]) => (
                <button key={k} className={basemap === k ? "on" : ""} onClick={() => setBasemap(k)}>{label}</button>
              ))}
            </div>
            <p className="fine">Background maps show today, not the flood date. Damage layers come from the radar images.</p>
          </fieldset>
        </div>
      )}
    </section>
  );
}
