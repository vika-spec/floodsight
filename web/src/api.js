// Thin client for the FloodSight server. Every error comes back as {error: "plain sentence"}.
async function call(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
  } catch {
    throw new Error("The FloodSight server is not reachable. Is `python serve.py` still running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
  return data;
}

export const api = {
  status: () => call("/api/status"),
  places: (q) => call(`/api/places?q=${encodeURIComponent(q)}`),
  runs: () => call("/api/runs"),
  modelCard: () => call("/api/model-card"),
  run: (id) => call(`/api/runs/${encodeURIComponent(id)}`),
  start: (body) => call("/api/jobs", { method: "POST", body }),
  job: (id) => call(`/api/jobs/${id}`),
  flowStart: (id, lon, lat, km) => call(`/api/runs/${encodeURIComponent(id)}/flow`, { method: "POST", body: { lon, lat, km } }),
  flow: (key) => call(`/api/flows/${encodeURIComponent(key)}`),
  geojson: async (id, rel) => (rel ? call(fileUrl(id, rel)) : null),
};

export const fileUrl = (id, rel) => `/api/runs/${encodeURIComponent(id)}/files/${rel}`;

// ----------------------------------------------------------------------------------------------- helpers
export function parseBbox(text) {
  const v = String(text || "").split(",").map((s) => parseFloat(s.trim()));
  if (v.length !== 4 || v.some((x) => !Number.isFinite(x))) return null;
  const [w, s, e, n] = v;
  if (!(w < e && s < n) || Math.abs(s) > 85 || Math.abs(n) > 85 || Math.abs(w) > 180 || Math.abs(e) > 180) return null;
  return [w, s, e, n];
}

export function bboxText(b) {
  return b.map((x) => x.toFixed(4)).join(", ");
}

export function areaKm2([w, s, e, n]) {
  const R = 6371.0088;
  const rad = Math.PI / 180;
  return Math.abs(R * R * (e - w) * rad * (Math.sin(n * rad) - Math.sin(s * rad)));
}

export const fmt = {
  n: (x, d = 0) => (x == null || Number.isNaN(x) ? "–" : Number(x).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d })),
  km2: (x) => (x == null ? "–" : x < 0.1 ? `${(x * 100).toFixed(1)} ha` : `${Number(x).toFixed(2)} km²`),
  date: (s) => (s ? new Date(s + "T00:00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "–"),
  dur: (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, "0")} s`),
};

export const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
