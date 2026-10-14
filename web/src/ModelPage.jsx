import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";

// "How the AI model was tested": the model card as its own full-width page (opened over the map, closed with
// Esc or the button). It holds every recorded test of the model; the per-run AI tab only shows what the AI did
// on that area. All numbers come from docs/model_card.json (recorded runs, see docs/tuning_log.md).

const n2 = (x) => (x == null ? "–" : x < 0.1 ? x.toFixed(3) : x.toFixed(2));        // small values unrounded
const pct = (x) => (x == null ? "–" : x < 0.1 ? `${(100 * x).toFixed(1)}%` : `${Math.round(100 * x)}%`);

export default function ModelPage({ onClose }) {
  const [card, setCard] = useState(null);
  const [runIds, setRunIds] = useState(new Set());
  const closeRef = useRef(null);
  useEffect(() => {
    api.modelCard().then(setCard).catch(() => setCard(false));
    api.runs().then((rs) => setRunIds(new Set(rs.map((r) => r.run_id)))).catch(() => {});
    closeRef.current?.focus();
    const esc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const openRun = (id) => { onClose(); window.location.hash = `#/run/${encodeURIComponent(id)}`; };
  const cmp = card?.comparison;
  const prod = cmp?.rows?.find((r) => r.chosen);
  const rules = cmp?.rows?.find((r) => /rules/i.test(r.model));
  const noBefore = cmp?.rows?.find((r) => /after-only/i.test(r.model));
  const himalaya = (card?.tests || []).filter((t) => /Buner/.test(t.name));
  const sd = card?.second_dataset;

  return (
    <div className="modelpage" role="dialog" aria-modal="true" aria-labelledby="mp-title">
      <div className="mp-bar">
        <h1 id="mp-title">How the AI model was tested</h1>
        <button ref={closeRef} className="btn secondary small" onClick={onClose}>Close <kbd>Esc</kbd></button>
      </div>
      <div className="mp-body">
        {card === null && <p className="hint">Loading…</p>}
        {card === false && <p className="callout bad">Model details are unavailable on this server.</p>}
        {card && (
          <>
            <p className="lede">{card.name}. It compares the radar image before and after the flood and gives every pixel a
              probability of being <b>new flood water</b>. Trained only on the listed training data; every score below comes
              from data it never trained on, and every setting was chosen on a separate validation event.</p>

            <section className="mp-heads" aria-label="Headline results">
              {prod && (
                <div className="mp-head">
                  <span className="mp-l">Kuro Siwo held-out test events</span>
                  <span className="mp-v">IoU {n2(prod.test_iou)}</span>
                  <span className="mp-s">precision {pct(prod.test_precision)}, recall {pct(prod.test_recall)}</span>
                </div>
              )}
              {rules && prod && (
                <div className="mp-head">
                  <span className="mp-l">Gain over radar rules alone</span>
                  <span className="mp-v">+{(prod.test_iou - rules.test_iou).toFixed(2)} IoU</span>
                  <span className="mp-s">same test tiles; without the before-image: {noBefore ? `−${(prod.test_iou - noBefore.test_iou).toFixed(2)}` : "–"}</span>
                </div>
              )}
              {sd && (
                <div className="mp-head">
                  <span className="mp-l">Second dataset, never seen (Sen1Floods11)</span>
                  <span className="mp-v">IoU {sd.rows.map((r) => n2(r.iou)).join(" / ")}</span>
                  <span className="mp-s">{sd.rows.map((r) => r.split.split(" (")[0]).join(" / ")}; after-only model</span>
                </div>
              )}
              {himalaya.length > 0 && (
                <div className="mp-head warn">
                  <span className="mp-l">Himalayan flash flood (Buner), never seen</span>
                  <span className="mp-v">{himalaya.map((t) => pct(t.model_alone_precision)).join(" / ")} precise</span>
                  <span className="mp-s">but recall {himalaya.map((t) => pct(t.model_alone_recall)).join(" / ")}: debris, not open water</span>
                </div>
              )}
            </section>

            <nav className="mp-toc" aria-label="Sections">
              <a href="#mp-scenes">Unseen scenes</a><a href="#mp-second">Second dataset</a>
              <a href="#mp-compare">Other models</a><a href="#mp-train">Training and limits</a>
            </nav>

            {card.tests && (
              <section id="mp-scenes">
                <h2 className="h2">1. Real floods it never saw</h2>
                <p className="hint">Full FloodSight runs from raw Sentinel-1 data, compared with an independent reference map. Each area
                  was decided before it was run and reported as it came out. "AI alone" is the model's own map; "final map" adds the
                  radar rules, Sentinel-2 and terrain.</p>
                <div className="tablewrap">
                  <table className="aitests">
                    <thead><tr><th>Flood</th><th>AI alone: IoU</th><th>precision</th><th>recall</th><th>Final map: IoU</th><th>precision</th></tr></thead>
                    <tbody>
                      {card.tests.map((t) => (
                        <tr key={t.name}>
                          <td className="tname"><b>{t.name}</b><br /><small>{t.role}; reference: {t.reference}</small>
                            {t.run && runIds.has(t.run) && <><br /><button className="linkbtn" onClick={() => openRun(t.run)}>Open this run</button></>}</td>
                          {t.model_alone_na ? <td colSpan={3} className="na">{t.model_alone_na}</td> : (
                            <><td>{n2(t.model_alone_iou)}</td><td>{pct(t.model_alone_precision)}</td><td>{pct(t.model_alone_recall)}</td></>)}
                          <td>{n2(t.final_map_iou)}</td><td>{pct(t.final_map_precision)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <ul className="facts">{card.tests.filter((t) => t.note).map((t) => <li key={t.name}><b>{t.name}:</b> {t.note}</li>)}</ul>
              </section>
            )}

            {sd && (
              <section id="mp-second">
                <h2 className="h2">2. A second listed dataset it never saw: {sd.name}</h2>
                <p className="hint">{sd.model} {sd.why_not_production_model}</p>
                <div className="tablewrap">
                  <table className="aitests">
                    <thead><tr><th>Split</th><th>Chips</th><th>IoU</th><th>precision</th><th>recall</th><th>Dataset's per-chip Otsu labels: IoU</th></tr></thead>
                    <tbody>
                      {sd.rows.map((r) => (
                        <tr key={r.split}><td className="tname"><b>{r.split}</b></td><td>{r.chips}</td><td>{n2(r.iou)}</td>
                          <td>{pct(r.precision)}</td><td>{pct(r.recall)}</td><td>{n2(r.otsu_iou)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="fine">{sd.what_is_scored} {sd.reading}</p>
              </section>
            )}

            {cmp && (
              <section id="mp-compare">
                <h2 className="h2">3. Is this the right model? {cmp.rows.length - 1} alternatives, same data and budget</h2>
                <p className="hint">{cmp.source}. Flood IoU; the model used by FloodSight is highlighted.</p>
                <div className="tablewrap">
                  <table className="aitests cmp">
                    <thead><tr><th>Method</th><th>Validation (chooses)</th><th>Test: IoU</th><th>precision</th><th>recall</th>
                      <th>Unseen event 205</th><th>ms per tile</th></tr></thead>
                    <tbody>
                      {cmp.rows.map((r) => (
                        <tr key={r.model} className={r.chosen ? "chosen" : ""}>
                          <td className="tname">{r.chosen ? <b>{r.model}</b> : r.model}</td><td>{n2(r.val_iou)}</td>
                          <td>{n2(r.test_iou)}</td><td>{pct(r.test_precision)}</td><td>{pct(r.test_recall)}</td>
                          <td>{n2(r.unseen_iou)}</td><td>{r.ms_per_tile != null ? Math.round(r.ms_per_tile) : "–"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {cmp.reading && <p className="callout">{cmp.reading}</p>}
                {cmp.rule && <p className="fine">{cmp.rule}</p>}
              </section>
            )}

            <section id="mp-train">
              <h2 className="h2">4. Training data and limits</h2>
              <ul className="facts">
                <li><b>Inputs:</b> {card.inputs}</li>
                <li><b>Output:</b> {card.output}</li>
                <li><b>Training data:</b> {card.training_data}</li>
                <li><b>Split:</b> {card.split}</li>
                <li><b>Threshold:</b> {card.threshold}</li>
                <li><b>How FloodSight uses it:</b> {card.how_used}</li>
                {(card.limitations || []).map((x) => <li key={x}><b>Limit:</b> {x}</li>)}
              </ul>
              <p className="fine">Every setting, result and decision is dated in docs/tuning_log.md in the code repository; the comparison
                protocol, written before any comparison model was trained, is in docs/model_comparison.md.</p>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
