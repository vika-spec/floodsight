# FloodSight results: the single source of truth

Every number in the report, the model card, the dashboard and the defence guide must come from this table.
Each row says **what was evaluated** (final fused map or AI alone), **against what**, and **what kind of test** it is.
Source runs and logs: `docs/tuning_log.md`, `models/model_comparison.json`, `models/sen1floods11_eval.json`,
`runs/<run>/validation/`. Last updated 11 Oct 2026, before the code freeze; the case-study row is re-run once
after the freeze and only that row is then updated.

## Test types

| Type | Meaning |
|---|---|
| **Blind** | Decided before it was run, never used to choose any setting, reported as it came out |
| **Case study** | Viewed during development, but no setting was ever chosen from it: **not blind** |
| **Held-out (same dataset)** | Flood events of the training dataset kept out of training and selection |
| **Second dataset** | A listed dataset never used for training or selection |
| **Diagnosis** | Used to find weaknesses; design may change because of it, no threshold fitted; not a test |
| **Training event** | Used in training; reported only as a chain check, never as a test |

"Confirmed" in FloodSight means two independent evidence sources inside the system agree (both radar tracks,
radar + AI, or radar + Sentinel-2). Reference maps (Copernicus EMS) are never used by the system; they are only
used afterwards to compute the scores below.

## 1. Final fused map vs reference maps (full pipeline from raw Sentinel data)

| Flood | Reference | Type | IoU | Precision | Recall | Other verified numbers |
|---|---|---|---|---|---|---|
| Trishuli, Nepal, Aug 2026 | EMS EMSR927, all AOIs combined (latest map per AOI) | Case study (not blind) | 0.466 | 0.958 | 0.476 | 28 settlements cut off, 5 likely, 4 unknown; AOI03: 98% of the road length we close is damaged in EMS; radar-unknown 0.6% of the area |
| Lasithi plateau, Crete, Oct 2026 | EMS EMSR933 AOI03, latest map (MONIT02) | Blind | 0.424 | 0.467 | 0.822 | Confirmed detections: precision 0.718 (3.66 km²); single-source: 0.156 (2.95 km²). Other EMS products of the same AOI: first delineation IoU 0.476 (P 0.479, R 0.986; confirmed P 0.846), grading map 0.440 (P 0.493, R 0.801; confirmed P 0.749). Roads (grading map): 85% of EMS-damaged road length flagged; 48% of our closed length damaged in EMS |
| Buner, Pakistan, Aug 2025 | EMS EMSR838 AOI10 | Blind (Himalaya, run once) | 0.197 | 0.318 | 0.34* | Confirmed precision 0.595 vs single-source 0.299 |
| Buner, Pakistan, Aug 2025 | EMS EMSR838 AOI11 | Blind (Himalaya, run once) | 0.089 | 0.584 | 0.10* | Confirmed precision 0.938 vs single-source 0.535 |
| Sri Lanka, May 2017 | Kuro Siwo 205 labels | Held-out event, never used (blind) | 0.468 | 0.993 | 0.469 | — |
| Sindh, Pakistan, 2022 | Kuro Siwo 1111009 labels | Training event (not a test) | 0.848 | — | — | — |
| Meža valley, Slovenia | EMS EMSR680 AOI15 | Diagnosis | 0.112 | 0.516 | — | Narrow valley at pixel scale |
| Mura plains, Slovenia | EMS EMSR680 AOI04/10-13 | Diagnosis | 0.278 | 0.527 | — | — |

*Buner recall as recorded in docs/limitations.md (34% and 10%).

## 2. AI model alone (Siamese U-Net, production weights `siamese_unet.pt`, threshold 0.8)

| Test | Type | IoU | Precision | Recall | Note |
|---|---|---|---|---|---|
| Kuro Siwo events 427 + 1111006, same 600 tiles per event as all compared models | Held-out (same dataset) | 0.653 | 0.86 | 0.73 | F1 = 2·IoU/(1+IoU) = 0.79 |
| Kuro Siwo event 205, Kuro Siwo's own images (130 tiles) | Held-out, never used (blind) | 0.457 | 0.96 | 0.47 | Same-tile comparison |
| Kuro Siwo event 205, FloodSight's own processing of the same area | Held-out, never used (blind) | 0.442 | — | — | On Kuro Siwo's own images of that area: 0.578 (processing-chain gap) |
| Lasithi plateau vs EMS EMSR933, latest map (MONIT02) | Blind | 0.549 | 0.744 | 0.677 | Radar rules alone: 0.513 |
| Lasithi plateau vs EMS EMSR933, first delineation | Blind | 0.842 | 0.886 | 0.944 | Radar rules alone: 0.624 |
| Buner AOI10 / AOI11 vs EMS flood areas | Blind (Himalaya) | 0.006 / 0.006 | 0.945 / 0.994 | 0.006 / 0.006 | Damage was debris, which the model was not trained on |
| Trishuli vs EMSR927 | Case study | not scored | — | — | EMS mapped mass movement only; no flood-water reference |

After-only variant (`cmp_post_only.pt`, same Kuro Siwo training; the only variant usable without a before-image):

| Test | Type | IoU | Precision | Recall | Note |
|---|---|---|---|---|---|
| Sen1Floods11 test split, 90 chips, all surface water | Second dataset | 0.479 | 0.592 | 0.715 | Dataset's per-chip Otsu labels: 0.546 |
| Sen1Floods11 Bolivia hold-out, 15 chips | Second dataset | 0.684 | 0.835 | 0.791 | Otsu labels: 0.697 |

## 3. Model comparison (same 600 tiles per event; threshold per model chosen on validation 1111013)

All comparison models (`cmp_*`) were trained with the same event split, budget (15 epochs, batch 8, lr 3e-4,
AdamW, cross-entropy + Dice, same augmentation, ImageNet start, 2,000 tiles per event). The **production model
was trained earlier in its own run** (best epoch 6 by validation); a same-budget retrain of the same architecture
is included so the architecture is compared fairly.

| Method | Validation IoU | Test IoU | Test P | Test R | Event 205 IoU | ms/tile |
|---|---|---|---|---|---|---|
| Siamese U-Net R34, production | 0.758 | 0.653 | 0.86 | 0.73 | 0.457 | 49 |
| Siamese U-Net R34, same-budget retrain | 0.703 | 0.686 | 0.84 | 0.79 | 0.492 | 53 |
| Early fusion U-Net R50 | 0.724 | 0.677 | 0.81 | 0.80 | 0.625 | 73 |
| Siamese DeepLabV3+ R34 | 0.726 | 0.673 | 0.83 | 0.78 | 0.577 | 83 |
| Early fusion U-Net R34 | 0.690 | 0.677 | 0.80 | 0.81 | 0.608 | 68 |
| Siamese SegFormer MiT-B2 | 0.610 | 0.618 | 0.82 | 0.72 | 0.584 | 146 |
| After-only U-Net R34 | 0.549 | 0.472 | 0.62 | 0.66 | 0.197 | 65 |
| Gradient-boosted trees | 0.531 | 0.537 | 0.76 | 0.65 | 0.404 | 279 |
| Physical radar rules (no learning) | 0.658 | 0.479 | 0.88 | 0.51 | 0.288 | 134 |

Derived numbers used in text: **+0.17 IoU** = production 0.653 − radar rules 0.479 (test, same tiles);
**+0.12** = 0.653 − gradient-boosted trees 0.537; **before-image worth about 0.18** = 0.653 − after-only 0.472
(0.21 for the same-budget retrain). Decision rule (fixed before training): replace the production model only if
validation IoU is at least 0.02 higher; best challenger 0.726 < 0.758, so it stays.
