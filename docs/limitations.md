# What FloodSight can and cannot do

Kept up to date during development; every number comes from a run recorded in `docs/tuning_log.md`.
The report's limitations section is written from this file. Last updated: 5 Oct 2026.

## What it can do (with evidence)

| Capability | Evidence |
|---|---|
| Map flood water on plains reliably | Sindh, Kuro Siwo 1111009 (a training event): IoU 0.85. Sri Lanka, Kuro Siwo 205 (never used): IoU 0.47 at 99% precision |
| Flag damage that is real (few false alarms) | Trishuli vs EMSR927: precision 96%. Buner AOI11: 94% for confident detections |
| Close roads correctly for the cut-off analysis | Trishuli AOI03: 98% of the road length we closed is damaged in EMS |
| Run for any area and date, hills or plains, from raw Sentinel data | Same code for Trishuli, Buner, Slovenia, Sindh, Sri Lanka; nothing set per area |
| Say "unknown" instead of "safe" where the radar cannot see | Trishuli: 0.6% of the area unseen by both orbit tracks, marked unknown (9% before a track-merge bug was fixed on 10 Oct) |
| Confidence labels that mean something (since 5 Oct) | Buner AOI10: confident 60% vs low 30% correct; AOI11: 94% vs 54% |
| Use only allowed inputs | Sentinel-1, Sentinel-2, Copernicus DEM, pre-event OpenStreetMap; EMS only for checking (not even on the public server) |
| Trace a flood path from any point and list places along it | Checked against OpenStreetMap rivers (solid/dashed on the map); `scripts/check_flowpaths.py` |
| Situation report in English and Nepali, numbers only from the maps | Fixed template; no language model writes numbers |

## What it cannot do

### Timing
- **No early warning.** Sentinel-1 passes the same track every 12 days (6 with two satellites), so the
  system maps the impact after the fact. It cannot warn of a glacier collapse or lake outburst minutes ahead.
- **Fast water may be gone.** Flash-flood water drains within hours. By the first radar pass mostly debris is
  left (Buner: cloudburst on 15 Aug, after-images on 16 and 20 Aug).

### Mountains
- **Much of the mud and debris is missed.** Recall in the Himalaya is low: Trishuli 48%, Buner 34% and 10%.
  In Buner, 42% / 30% of the EMS area showed no radar change at all, and 43% / 53% only weak change. Fresh
  debris on an already rough riverbed looks almost the same to the radar as before.
- **Narrow valleys are close to the pixel size.** Analysis is at 20 m; a damaged strip 20–60 m wide is 1–3
  pixels. Meža (Slovenia), radar only: IoU 0.014. Mapping at 10 m did not fix it (0.020).
- **Radar blind spots.** Steep slopes facing toward or away from the satellite (layover/shadow) cannot be
  measured by one track; where the other track cannot see them either they are shown as unknown (Trishuli 0.6%).
- **Debris confidence is not fully calibrated.** In the Meža valley, confident debris is still slightly less
  precise than low-confidence debris (43% vs 52%, small sample).
- **Other change can look like debris.** Fresh landslides, construction or harvested fields near the river.

### Clouds
- **Sentinel-2 helps only when the sky is clear.** Where it saw the ground it made a big difference (Buner AOI10:
  IoU 0.03 radar-only to 0.197), but monsoon clouds often hide it.

### AI model
- **Detects water, not debris.** The training data (Kuro Siwo) labels water only.
- **Never learned mountains.** Training events have 0–2.5% steep ground; Trishuli has 82%. In Buner the model
  alone is precise (95–99%) but finds little, because the damage was mostly debris.
- **Optical is not inside the network.** Kuro Siwo has no Sentinel-2, so optical evidence is combined after
  the model, at the decision stage.

### Maps of roads and buildings
- **Only as good as OpenStreetMap.** Remote valleys are often incompletely mapped. Cut-off status depends on the
  mapped road network; footpaths are excluded by default.
- **"Cut off" means no road route,** not that people are in danger or that no footpath exists.

### Helicopter landing options
- **Candidates, not certified sites.** Checked: slope at most 7° (US Army FM 3-21.38 landing guidance), 60 m
  clear, radar-visible, away from detected flood/debris and from buildings and power lines mapped in OpenStreetMap.
  Not checked: trees, wires or buildings missing from OpenStreetMap, soft ground, wind, the approach path, and
  debris our map missed (much of it in mountain valleys). Not validated against any reference.

### Flow path (bonus)
- **Steepest-descent on a 30 m elevation model.** Follows mountain valleys well; on flat plains the direction of
  water is hard to read from elevation and the path can leave the real river (shown dashed where it does).
- **Height bands are exposure levels,** not a prediction of how high this flood reaches.

### Public demo site
- **New analyses need an access code** and are capped per day (processing budget of the free Copernicus tier).
- **Slower than a laptop** for large areas (no graphics card on the free server).
- **Validation only where a reference map was loaded offline.** For a new area there is no EMS comparison;
  the checked example runs are linked instead.

### Own raw-data (GRD) processing chain
- **Simpler geocoding than ESA SNAP.** Our Level-1 GRD chain (thermal-noise removal, calibration, geocoding,
  terrain parallax correction) uses the product's GCP grid plus a DEM shift instead of precise-orbit range-Doppler
  terrain correction, and downloads whole measurement files, so it is slower than the server route.

## Tested and not adopted

- **Several before-images as the change reference** (temporal median of the 4 latest same-track images, after
  Handwerger et al. 2022). On the diagnosis events it made the map slightly worse (Meža IoU 0.112 to 0.106,
  Mura 0.278 to 0.276; Sindh unchanged): with only weeks of history, seasonal vegetation change shows up as
  false debris. It would need a longer, season-matched history (e.g. the same months of earlier years).
