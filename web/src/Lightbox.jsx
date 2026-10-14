import React, { useEffect, useState } from "react";

// Full-screen viewer for figures. Any component calls zoom({src, alt, caption}) for one image, or
// zoom({before, after, labels: [l, r], caption}) for a before/after pair with a slider.
export const zoom = (item) => window.dispatchEvent(new CustomEvent("fs-zoom", { detail: item }));

export default function Lightbox() {
  const [item, setItem] = useState(null);
  const [actual, setActual] = useState(false);
  const [split, setSplit] = useState(50);
  const [flick, setFlick] = useState(false);
  const [showAfter, setShowAfter] = useState(false);
  useEffect(() => {
    if (!flick) return undefined;
    const t = setInterval(() => setShowAfter((v) => !v), 900);
    return () => clearInterval(t);
  }, [flick]);
  useEffect(() => {
    const open = (e) => { setItem(e.detail); setActual(false); setSplit(50); setFlick(false); };
    const key = (e) => { if (e.key === "Escape") setItem(null); };
    window.addEventListener("fs-zoom", open);
    window.addEventListener("keydown", key);
    return () => { window.removeEventListener("fs-zoom", open); window.removeEventListener("keydown", key); };
  }, []);
  if (!item) return null;
  const close = () => setItem(null);
  return (
    <div className="lightbox" role="dialog" aria-modal="true" aria-label={item.alt || "Enlarged figure"} onClick={close}>
      <div className="lb-bar" onClick={(e) => e.stopPropagation()}>
        {item.src && (
          <button className="btn small" onClick={() => setActual((v) => !v)}>
            {actual ? "Fit to screen" : "Actual size (scroll to pan)"}
          </button>
        )}
        {item.src && <a className="btn small" href={item.src} target="_blank" rel="noreferrer">Open in new tab</a>}
        {!item.src && <button className="btn small" onClick={() => setFlick((v) => !v)}>{flick ? "Stop flicker (slide)" : "Flicker"}</button>}
        <button className="btn small primary" onClick={close} autoFocus>Close (Esc)</button>
      </div>
      <div className={`lb-body ${actual ? "actual" : ""}`} onClick={(e) => e.stopPropagation()}>
        {item.src ? (
          <img src={item.src} alt={item.alt || ""} onClick={() => setActual((v) => !v)} />
        ) : (
          <div className="imgswipe lb-swipe" style={{ "--split": flick ? (showAfter ? "0%" : "100%") : `${split}%` }}>
            <img src={item.before} alt={item.labels?.[0] || "Before"} />
            <img className="top" src={item.after} alt={item.labels?.[1] || "After"} />
            {flick ? <span className="cap l big">{showAfter ? item.labels?.[1] : item.labels?.[0]}</span>
              : <><span className="cap l">{item.labels?.[0]}</span><span className="cap r">{item.labels?.[1]}</span></>}
            <input type="range" min="0" max="100" value={split} onChange={(e) => setSplit(Number(e.target.value))}
              aria-label="Slide between before and after" />
          </div>
        )}
      </div>
      {item.caption && <p className="lb-cap" onClick={(e) => e.stopPropagation()}>{item.caption}</p>}
    </div>
  );
}

// An image in the side panel that opens the viewer on click.
export function ZoomImg({ src, alt, caption, className }) {
  return (
    <button className={`zoomable ${className || ""}`} onClick={() => zoom({ src, alt, caption })} title="Click to enlarge">
      <img src={src} alt={alt} />
      <span className="zoom-hint" aria-hidden="true">Click to enlarge</span>
    </button>
  );
}
