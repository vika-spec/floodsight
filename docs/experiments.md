# Experiments (kept honest: what we tried, what happened, what we decided)

## 1. Learned "normal variation" change test — REJECTED (3 Oct 2026)

**Idea.** Instead of one fixed 3 dB change threshold, learn per scene how much each kind of place normally
changes between two passes (small CNN trained on 3 earlier images of the same track, heteroscedastic
Gaussian, split-conformal calibration at a 1% false-alarm rate). Motivation: the miss breakdown on the
diagnosis event Slovenia EMSR680 showed faint missed damage in the Meža valley and harvested fields flagged
as debris on the Mura plains.

**Synthetic check (passed).** Farm fields vs forest: field false alarms 34–39% (fixed rule) -> 0.3–0.5%.

**Real check (failed), Slovenia EMSR680, same before/after images:**

| | Meža valley (AOI15) old -> new | Mura plains (AOI04, 10–13) old -> new |
|---|---|---|
| our flood + debris area | 0.13 -> 0.21 km² | 23.5 -> 58.3 km² |
| precision vs EMS | 0.158 -> 0.137 | 0.034 -> 0.018 |
| recall vs EMS | 0.016 -> 0.023 | 0.309 -> 0.399 |
| cut-off settlements | — | 4 -> 25 |

**Why (most likely).** 4 Aug 2023 was a day of extreme rain over all of Slovenia: wet soil and crops
brighten the radar everywhere, which really is unusual compared with the dry June–July history, but it is
weather, not flood damage. "Unusual" is not the same as "flooded". On riverbanks the natural variability
(water level changes) is high, so faint damage stayed "normal".

**Decision.** Removed from the pipeline; fixed physical rules kept. Instead, debris (a mountain process) is
only allowed in mountain terrain, defined by the UNEP-WCMC mountain classes (Kapos et al. 2000).

## 2. Debris only in mountain terrain — ADOPTED (3 Oct 2026)

Debris flows need steep catchments; on plains a flood shows up as water. Mountain terrain follows the
UNEP-WCMC definition (Kapos et al., 2000, "Developing a map of the world's mountain forests"): elevation
>= 2500 m; 1500–2500 m with slope >= 2 deg; 1000–1500 m with slope >= 5 deg or local elevation range >= 300 m;
300–1000 m with local elevation range >= 300 m (range within a 7 km radius). An external definition: not
fitted to any event. Results on the diagnosis runs are recorded in tuning_log.md.
