# FloodSight — Report skeleton (max 6 pages)

Fill every [bracket] from real runs. Keep figures large; judges skim.

## 1. Problem and approach (½ page)
- Rescuers need: where the flood hit, what was damaged, who is cut off. Ground sensors and roads fail; satellites don't.
- One sentence per stage of the pipeline + the architecture figure (README "How it works").
- Multimodal: SAR (Sentinel-1) + optical (Sentinel-2) + elevation (Copernicus DEM) + maps (OSM) + learned model.

## 2. Data and rules (½ page)
- Table of inputs, licences and how pre-event OSM / no-leakage is enforced (README "Data rules").
- Same-orbit pair selection: show the pair chosen for Trishuli: [track, dates, gap].

## 3. Flood and debris mapping (1½ pages)  ← 30% of the score
- Water: split-based adaptive threshold + change; debris: VV/VH change + texture, river-corridor constraint.
- Mountain handling: HAND, slope, layover/shadow mask (figure: blind-spot map for Trishuli, [x]% of area).
- Fusion with second track / Sentinel-2 / model; confidence levels.
- Case study vs EMSR927: IoU [ ], F1 [ ], precision [ ], recall [ ]; agreement map figure; discuss main
  disagreements honestly (blind spots, receded water, debris vs landslide).

## 4. AI component: segmentation model (1 page)  ← 25%
- Architecture (Siamese U-Net, shared encoder, |post−pre| fusion), training on Kuro Siwo, event-level split.
- Held-out Kuro Siwo test events: flood IoU [ ], F1 [ ].
- Unseen Himalayan scenes: Trishuli (+ Teesta / Melamchi / Beas) rule-based vs model vs fused, same metrics.
- What the model adds and where it fails (domain gap: steep terrain).

## 5. Damage and cut-off analysis (1 page)  ← 20%
- Buildings/roads/bridges statuses and thresholds; bridge handling.
- Road graph before/after, hospital targets, buffer beyond area, detour rule.
- Trishuli results table: settlements cut off [ ], bridges hit [ ]; map figure.
- Bonus: flow path from [upstream point], settlements listed.

## 6. Dashboard and situation report (½ page)  ← 10%
- Screenshot of the dashboard + the one-page PDF (EN and NE side by side).
- "Every number comes from the maps": template design, no LLM in the path.

## 7. Limitations (½ page)  ← part of 15%
- Copy and extend README "Limitations" with numbers from your runs.

## 8. Attributions and citations
- The three required lines verbatim + Kuro Siwo citation + EMS credit.
