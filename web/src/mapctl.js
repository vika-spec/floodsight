// Imperative Leaflet controller. React decides WHAT is shown; this class knows HOW to show it on the map.
import L from "leaflet";
import { api, esc, fileUrl, fmt } from "./api.js";
import { C, STATUS } from "./theme.js";

const BASEMAPS = {
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Imagery © Esri, Maxar, Earthstar Geographics (today's imagery, not the event date)",
    maxZoom: 19,
  },
  streets: {
    label: "Streets",
    url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attribution: "© OpenStreetMap contributors (current map)",
    maxZoom: 19,
  },
  topo: {
    label: "Terrain",
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attribution: "© OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors",
    maxZoom: 17,
  },
};
export const BASEMAP_KEYS = Object.keys(BASEMAPS).map((k) => [k, BASEMAPS[k].label]);

const RASTERS = { flood: 0.95, change: 0.9, before: 0.92, after: 0.92, model: 1, rules: 0.95 };

export class MapCtl {
  constructor(el) {
    this.map = L.map(el, { zoomControl: false, attributionControl: true, preferCanvas: true, minZoom: 3 })
      .setView([28.0, 84.5], 7);
    L.control.zoom({ position: "bottomright" }).addTo(this.map);
    L.control.scale({ position: "bottomleft", imperial: false }).addTo(this.map);
    this.map.attributionControl.setPrefix(false);
    for (const [name, z] of [["rasters", 350], ["cmpLeft", 352], ["cmpRight", 353], ["vectors", 420], ["points", 640]]) {
      this.map.createPane(name).style.zIndex = z;
    }
    this.renderer = L.canvas({ pane: "vectors", padding: 0.3 });
    this.layers = {};
    this.visible = {};
    this.listeners = new Set();
    this.setBasemap("satellite");
    this.map.on("click", (e) => this.listeners.forEach((f) => f(e.latlng)));
    this.map.on("move zoom resize", () => this._clip());
  }

  onClick(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  setBasemap(key) {
    const b = BASEMAPS[key];
    if (this.base) this.map.removeLayer(this.base);
    this.base = L.tileLayer(b.url, { attribution: b.attribution, maxZoom: b.maxZoom, crossOrigin: true }).addTo(this.map);
    this.baseKey = key;
  }

  invalidate() {
    setTimeout(() => this.map.invalidateSize(), 50);
  }

  fitBbox(b, opts = {}) {
    if (!b) return;
    const wide = this.map.getSize().x > 700;
    this.map.fitBounds([[b[1], b[0]], [b[3], b[2]]], {
      paddingTopLeft: [30, 30], paddingBottomRight: [wide && this.run ? 320 : 30, 30], maxZoom: 13, ...opts,
    });
  }

  flyTo(lat, lon, zoom = 14) {
    this.map.flyTo([lat, lon], Math.max(zoom, this.map.getZoom()), { duration: 0.8 });
  }

  // --------------------------------------------------------------------------------------- area of interest
  setAoi(b, style = {}) {
    if (this.aoi) this.map.removeLayer(this.aoi);
    this.aoi = null;
    if (!b) return;
    this.aoi = L.rectangle([[b[1], b[0]], [b[3], b[2]]], {
      color: C.aoi, weight: 2, dashArray: "7 5", fill: true, fillOpacity: 0.04, interactive: false, ...style,
    }).addTo(this.map);
  }

  startDraw(onDone) {
    this.stopDraw();
    const m = this.map;
    const el = m.getContainer();
    el.classList.add("is-drawing");
    m.dragging.disable();
    let start = null;
    let rect = null;
    const down = (e) => {
      start = e.latlng;
      rect = L.rectangle([start, start], { color: C.water, weight: 2, fillOpacity: 0.08, interactive: false }).addTo(m);
    };
    const move = (e) => start && rect.setBounds(L.latLngBounds(start, e.latlng));
    const up = (e) => {
      if (!start) return;
      const bb = L.latLngBounds(start, e.latlng);
      m.removeLayer(rect);
      this.stopDraw();
      if (Math.abs(bb.getEast() - bb.getWest()) > 1e-4 && Math.abs(bb.getNorth() - bb.getSouth()) > 1e-4) {
        onDone([bb.getWest(), bb.getSouth(), bb.getEast(), bb.getNorth()]);
      }
    };
    m.on("mousedown", down).on("mousemove", move).on("mouseup", up);
    this._drawOff = () => {
      m.off("mousedown", down).off("mousemove", move).off("mouseup", up);
      if (rect && m.hasLayer(rect)) m.removeLayer(rect);
      m.dragging.enable();
      el.classList.remove("is-drawing");
    };
  }

  stopDraw() {
    if (this._drawOff) this._drawOff();
    this._drawOff = null;
  }

  // --------------------------------------------------------------------------------------- run layers
  clearRun() {
    Object.values(this.layers).forEach((l) => l && this.map.removeLayer(l));
    this.layers = {};
    if (this.routeLayer) this.map.removeLayer(this.routeLayer);
    this.routeLayer = null;
    this._routes = null;
    this.compare(false);
    this.setFlowPath(null);
    this.run = null;
  }

  async setRun(R, visible) {
    this.clearRun();
    this.run = R;
    const id = R.run_id;
    for (const k of Object.keys(RASTERS)) {
      const w = R.web?.[k];
      if (w) {
        this.layers[k] = L.imageOverlay(fileUrl(id, w.png), w.bounds, { opacity: RASTERS[k], pane: "rasters", className: `ov-${k}` });
      }
    }
    const vec = R.vectors || {};
    const [bld, roads, broken, setl, health, landing] = await Promise.all(
      ["buildings", "roads", "broken_roads", "settlements", "health", "landing"].map((k) => api.geojson(id, vec[k]).catch(() => null)),
    );
    if (this.run !== R) return; // a newer run was opened meanwhile
    const r = this.renderer;
    const bb = R.aoi.bbox;
    const inAoi = (ll) => ll.lng >= bb[0] && ll.lng <= bb[2] && ll.lat >= bb[1] && ll.lat <= bb[3];
    if (bld?.features?.length) {
      this.layers.buildings = L.geoJSON(bld, {
        renderer: r, pane: "vectors",
        style: (f) => ({ color: f.properties.status === "likely_hit" ? C.building : C.buildingMaybe, weight: 1, fillOpacity: 0.85 }),
        pointToLayer: (f, ll) => L.circleMarker(ll, { radius: 3, renderer: r }),
        onEachFeature: (f, l) => l.bindTooltip(`Building ${f.properties.status === "likely_hit" ? "likely hit" : "possibly hit"} · ${f.properties.overlap_pct ?? "–"}% of footprint`),
      });
    }
    if (roads?.features?.length) {
      const isBridge = (f) => String(f.properties.is_bridge).toLowerCase() === "true";
      this.layers.roads = L.geoJSON(roads, {
        renderer: r, pane: "vectors",
        style: (f) => (isBridge(f) ? { color: C.bridge, weight: 7, opacity: 0.95 } : { color: C.road, weight: 3.5, opacity: 0.95 }),
        onEachFeature: (f, l) => l.bindTooltip(`${isBridge(f) ? "Bridge" : "Road"} ${esc(f.properties.name || f.properties.highway || "")} · ${fmt.n(f.properties.affected_m)} m affected`),
      });
    }
    if (broken?.features?.length) {
      this.layers.broken = L.geoJSON(broken, { renderer: r, pane: "vectors", style: { color: C.cut, weight: 5, opacity: 0.95, dashArray: "2 6", lineCap: "round" } });
    }
    if (health?.features?.length) {
      this.layers.health = L.geoJSON(health, {
        pane: "points",
        pointToLayer: (f, ll) => {
          const inside = inAoi(ll);
          return L.marker(ll, { pane: "points", icon: L.divIcon({ className: `pin-health${inside ? "" : " out"}`,
            html: "<span>+</span>", iconSize: inside ? [18, 18] : [11, 11] }) });
        },
        onEachFeature: (f, l) => l.bindTooltip(esc(f.properties.name || "Health facility") +
          (inAoi(l.getLatLng()) ? "" : " (outside the analysed area; used as a route destination)")),
        filter: (f) => f.geometry?.type === "Point",
      });
    }
    if (landing?.features?.length) {
      this.layers.landing = L.geoJSON(landing, {
        pane: "points",
        pointToLayer: (f, ll) => L.marker(ll, { pane: "points", icon: L.divIcon({
          className: f.properties.kind === "mapped helipad" ? "pin-heli heli-pad" : "pin-heli heli-zone",
          html: "<span>H</span>", iconSize: f.properties.kind === "mapped helipad" ? [18, 18] : [14, 14] }) }),
        onEachFeature: (f, l) => {
          const p = f.properties;
          l.bindTooltip(p.kind === "mapped helipad"
            ? `Helipad (pre-event map): ${esc(p.name)} · ${esc(p.state)}`
            : `Candidate landing ground · slope ${p.slope_deg}° · ~${p.clear_radius_m} m clear radius · candidate only: not checked for trees, unmapped wires or buildings, soft ground, or debris our map missed`);
        },
      });
    }
    const heli = {};
    for (const h of R.helicopter_access?.per_settlement || []) heli[h.name] = h;
    if (setl?.features?.length) {
      this.settlementIndex = {};
      this.layers.settlements = L.geoJSON(setl, {
        pane: "points",
        pointToLayer: (f, ll) => {
          const s = STATUS[f.properties.status] || STATUS.no_road_data;
          const big = ["cut_off", "likely_cut_off", "unknown", "detour"].includes(f.properties.status);
          return L.circleMarker(ll, { pane: "points", radius: big ? 8 : 5.5, color: "#fff", weight: 2, fillColor: s.color, fillOpacity: 1 });
        },
        onEachFeature: (f, l) => {
          const p = f.properties;
          const s = STATUS[p.status] || STATUS.no_road_data;
          this.settlementIndex[p.name] = l;
          l.bindTooltip(`${esc(p.name)} · ${s.label}`);
          l.bindPopup(
            `<div class="pop"><b>${esc(p.name)}</b><span class="pop-status" style="--c:${s.color}">${s.label}</span>` +
              `<p>${s.long}.</p><dl><dt>Hospital before</dt><dd>${esc(p.target_before || "–")}</dd>` +
              `<dt>Road distance</dt><dd>${p.km_before ?? "–"} km → ${p.km_after ?? "no route"}${p.km_after != null ? " km" : ""}</dd>` +
              `<dt>Travel time</dt><dd>${p.min_before ?? "–"} min → ${p.min_after ?? "no route"}${p.min_after != null ? " min" : ""}</dd>` +
              (p.buildings ? `<dt>Buildings</dt><dd>${p.buildings}</dd>` : "") +
              (p.priority ? `<dt>Priority</dt><dd>#${p.priority}</dd>` : "") +
              (p.target_after && p.target_after !== p.target_before ? `<dt>Hospital now</dt><dd>${esc(p.target_after)}</dd>` : "") +
              (heli[p.name] ? `<dt>Helipad</dt><dd>${heli[p.name].helipad ? `${esc(heli[p.name].helipad.name)}, ${heli[p.name].helipad.km} km` : "none mapped nearby"}</dd>` +
                `<dt>Flat open ground</dt><dd>${heli[p.name].zone ? `${heli[p.name].zone.km} km (${heli[p.name].zone.lat_lon.join(", ")})` : "none nearby"}</dd>` : "") +
              `</dl>` +
              (["cut_off", "likely_cut_off", "unknown", "detour"].includes(p.status)
                ? `<p class="pop-key">On the map: <span style="color:#5B6770">grey dashes</span> = usual route, ` +
                  `<span style="color:${C.ok}">green</span> = route now, <span style="color:${C.cut}">red dots</span> = where it is blocked.</p>`
                : "") +
              `</div>`,
          );
          l.on("click", () => this.showRoute(p.name));
        },
      });
    }
    this.applyVisibility(visible);
  }

  applyVisibility(visible) {
    this.visible = { ...visible };
    for (const [k, l] of Object.entries(this.layers)) {
      if (!l) continue;
      const on = !!visible[k] && !(this.cmp && (k === "before" || k === "after" || k === this.cmp.right));
      if (on && !this.map.hasLayer(l)) l.addTo(this.map);
      if (!on && this.map.hasLayer(l)) this.map.removeLayer(l);
    }
  }

  hasLayer(k) {
    return !!this.layers[k];
  }

  openSettlement(name) {
    const l = this.settlementIndex?.[name];
    if (!l) return;
    if (!this.map.hasLayer(this.layers.settlements)) this.layers.settlements.addTo(this.map);
    this.showRoute(name).then((bounds) => {
      if (bounds) this.map.flyToBounds(bounds.pad(0.15), { duration: 0.7, maxZoom: 15 });
      else this.map.flyTo(l.getLatLng(), Math.max(13, this.map.getZoom()), { duration: 0.7 });
      setTimeout(() => l.openPopup(), 750);
    });
  }

  // route to the nearest hospital before the flood (grey dashes), now (green), and where it is blocked (red)
  async showRoute(name) {
    if (this.routeLayer) this.map.removeLayer(this.routeLayer);
    this.routeLayer = null;
    const R = this.run;
    const vec = R?.vectors || {};
    if (!vec.routes) return null;
    if (!this._routes) {
      this._routes = Promise.all([api.geojson(R.run_id, vec.routes).catch(() => null),
        api.geojson(R.run_id, vec.route_blocks).catch(() => null)]);
    }
    const [routes, blocks] = await this._routes;
    if (this.run !== R) return null;
    const mine = (fc) => (fc?.features || []).filter((f) => f.properties.name === name);
    const rf = mine(routes);
    if (!rf.length) return null;
    const g = L.layerGroup();
    for (const f of rf) {
      const before = f.properties.route === "before";
      L.geoJSON(f, { pane: "vectors", style: { color: "#fff", weight: before ? 6 : 8, opacity: 0.9 } }).addTo(g);
      L.geoJSON(f, { pane: "vectors", style: before ? { color: "#5B6770", weight: 3.5, dashArray: "6 7", opacity: 1 }
        : { color: C.ok, weight: 5, opacity: 1 } }).addTo(g);
    }
    for (const f of mine(blocks)) {
      const [lon, lat] = f.geometry.coordinates;
      L.circleMarker([lat, lon], { pane: "points", radius: 7, color: "#fff", weight: 2, fillColor: C.cut, fillOpacity: 1 })
        .bindTooltip("Usual route blocked here").addTo(g);
    }
    g.addTo(this.map);
    this.routeLayer = g;
    return L.geoJSON({ type: "FeatureCollection", features: rf }).getBounds();
  }

  // --------------------------------------------------------------------------------------- swipe compare
  compare(on, right = "after", frac = 0.5) {
    const m = this.map;
    for (const k of ["cmpL", "cmpR", "cmpR2"]) if (this[k]) m.removeLayer(this[k]);
    this.cmpL = this.cmpR = this.cmpR2 = null;
    this.cmp = null;
    const panes = [m.getPane("cmpLeft"), m.getPane("cmpRight")];
    panes.forEach((p) => (p.style.clip = ""));
    if (on && this.run?.web?.before && this.run.web[right]) {
      const id = this.run.run_id;
      const w = this.run.web;
      this.cmpL = L.imageOverlay(fileUrl(id, w.before.png), w.before.bounds, { pane: "cmpLeft", opacity: 1 }).addTo(m);
      if (right === "flood" && w.after) {
        // result drawn on top of the after-image, so the right half reads as "after, with what we detected"
        this.cmpR2 = L.imageOverlay(fileUrl(id, w.after.png), w.after.bounds, { pane: "cmpRight", opacity: 1 }).addTo(m);
      }
      this.cmpR = L.imageOverlay(fileUrl(id, w[right].png), w[right].bounds, { pane: "cmpRight", opacity: 1 }).addTo(m);
      this.cmp = { right, frac };
    }
    this.applyVisibility(this.visible);
    this._clip();
  }

  aoiCenterFrac() {
    const b = this.run?.aoi?.bbox;
    if (!b) return null;
    const p = this.map.latLngToContainerPoint([(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]);
    const f = p.x / this.map.getSize().x;
    return f > 0.05 && f < 0.95 ? f : null;
  }

  setSplit(frac) {
    if (!this.cmp) return;
    this.cmp.frac = frac;
    this._clip();
  }

  _clip() {
    const m = this.map;
    const [lp, rp] = [m.getPane("cmpLeft"), m.getPane("cmpRight")];
    if (!this.cmp) {
      lp.style.clip = rp.style.clip = "";
      return;
    }
    const size = m.getSize();
    const nw = m.containerPointToLayerPoint([0, 0]);
    const se = m.containerPointToLayerPoint(size);
    const x = m.containerPointToLayerPoint([size.x * this.cmp.frac, 0]).x;
    lp.style.clip = `rect(${nw.y}px, ${x}px, ${se.y}px, ${nw.x}px)`;
    rp.style.clip = `rect(${nw.y}px, ${se.x}px, ${se.y}px, ${x}px)`;
  }

  // --------------------------------------------------------------------------------------- flow path
  // flood path from any point: height bands along the valley, the path, and places along it
  setFlowPath(fp, start) {
    if (this.flow) this.map.removeLayer(this.flow);
    this.flow = null;
    if (!fp && !start) return;
    const g = L.layerGroup().addTo(this.map);
    if (fp?.corridor?.features?.length) {
      const op = { 50: 0.14, 25: 0.2, 10: 0.3 };
      L.geoJSON(fp.corridor, {
        pane: "vectors",
        style: (f) => ({ color: C.flow, weight: 0, fillColor: C.flow, fillOpacity: op[f.properties.band_m] ?? 0.15 }),
        onEachFeature: (f, l) => l.bindTooltip(`Up to ${f.properties.band_m} m above the river`, { sticky: true }),
      }).addTo(g);
    }
    if (fp?.path) {
      L.geoJSON(fp.path, { pane: "vectors", style: { color: "#fff", weight: 8, opacity: 0.9 } }).addTo(g);
      if (fp.path_check?.features?.length) {
        // independent check: solid where the DEM path lies on a river mapped in OpenStreetMap, dashed where not
        L.geoJSON(fp.path_check, {
          pane: "vectors",
          style: (f) => ({ color: C.flow, weight: 4.5, opacity: 1, dashArray: f.properties.on_mapped_river ? null : "6 7" }),
          onEachFeature: (f, l) => l.bindTooltip(
            `km ${f.properties.from_km}-${f.properties.to_km}: ${f.properties.on_mapped_river
              ? "on a river mapped in OpenStreetMap" : "not near a mapped river (check here)"}`, { sticky: true }),
        }).addTo(g);
      } else {
        L.geoJSON(fp.path, { pane: "vectors", style: { color: C.flow, weight: 4.5, opacity: 1 } }).addTo(g);
      }
    }
    const mark = (list, opts, label) => (list || []).forEach((p) =>
      L.circleMarker([p.lat, p.lon], { pane: "points", color: "#fff", weight: 2, fillOpacity: 1, ...opts })
        .bindTooltip(`${esc(p.name)} · ${label(p)}`).addTo(g));
    mark(fp?.settlements, { radius: 7, fillColor: C.flow },
      (p) => `${p.km_downstream} km downstream, ${p.m_above_river ?? "?"} m above the river`);
    mark(fp?.facilities, { radius: 7, fillColor: C.health }, (p) => `health facility, ${p.km_downstream} km downstream`);
    mark(fp?.bridges, { radius: 5, fillColor: C.bridge }, (p) => `bridge, ${p.km_downstream} km downstream`);
    if (start) {
      L.circleMarker(start, { pane: "points", radius: 8, color: "#fff", weight: 3, fillColor: C.flow, fillOpacity: 1 })
        .bindTooltip("Start point").addTo(g);
    }
    this.flow = g;
    if (fp?.path) this.map.flyToBounds(L.geoJSON(fp.path).getBounds().pad(0.1), { duration: 0.7 });
  }
}
