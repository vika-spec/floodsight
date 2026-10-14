# Model comparison: protocol (written before any comparison model was trained)

Question: is the Siamese U-Net the right model, compared with the alternatives a reviewer would expect?

## Models

| Name (`--arch`) | What it is | Why it is in the comparison |
|---|---|---|
| `siamese` | **Ours.** Shared ResNet-34 encoder per date; decoder sees [after, abs(after - before)] at every scale | Production model |
| `early_fusion` | Before and after stacked as 6 channels into one U-Net (FC-EF, Daudt et al. 2018) | Standard change-detection baseline: does the shared encoder help? |
| `post_only` | After image only (single-image segmentation, like Sen1Floods11 baselines) | Does the before image help? |
| `siamese_unetpp` | U-Net++ nested skip connections, two-date fusion as ours | Idea behind SNUNet-CD (Fang et al. 2021); finer boundaries? |
| `siamese_attn_unet` | U-Net with scSE attention in the decoder (Roy et al. 2018), two-date fusion as ours | Attention variant of U-Net |
| `siamese_deeplabv3p` | DeepLabV3+ (atrous spatial pyramid pooling), two-date fusion as ours | Multi-scale context (small streams to large plains) |
| `siamese_segformer` | SegFormer MiT-B2 transformer encoder with the temporal difference | The ChangeFormer idea (Bandara & Patel 2022): long-range context |
| `segformer_early` | Plain SegFormer MiT-B2, dates stacked | Transformer single-network baseline (reported strong on Sentinel-1 floods) |
| `siamese` + `--terrain-loss 0.5` | Ours with a physics penalty on flood predicted on slopes steeper than the water rule (12°) | Physics-informed variant; weight 0.5 fixed in advance, not tuned |
| gradient-boosted trees | Per-pixel classical machine learning on before, after, change, slope and 5x5 context | Is deep learning needed at all? |
| physical rules | FloodSight's water rule alone (threshold + 3 dB darkening), no learning | Non-learned reference |

All library models come from segmentation-models-pytorch and are made two-date change models with exactly the
same fusion as ours, so the comparison isolates the network, not the fusion.

## Fairness rules (fixed now, before results)

1. One event split for every model: validation 1111013; test 427 and 1111006; events 205, 421 and 498 left
   out completely (kept unseen); every other event in the dataset folder is training. `compare_models.py`
   refuses a checkpoint trained on a different split.
2. Same training budget for every deep model: same epochs, tiles per event, learning rate, loss, augmentation
   and ImageNet initialisation (`--pretrained`). Only `--batch` may be lowered if a model does not fit in
   8 GB; that is reported.
3. Best epoch and flood threshold are chosen on the **validation** event only.
4. Test events are scored once per model, on the same tiles (fixed seed). Kuro Siwo 205 (never used for
   training or selection) is added as an extra unseen event, labelled as such.
5. **Rule for changing the production model:** a model replaces the Siamese U-Net only if its VALIDATION flood
   IoU is higher by at least 0.02 and it runs inside the pipeline. Test results are reported for all models
   but never used to choose. Whatever the outcome, all rows go into the report.

## Considered but not run

- **Mamba-based models (e.g. FM-Mamba):** need the `mamba-ssm` CUDA kernels, which are not supported on
  Windows; noted as future work.
- **Multimodal network with HAND, DEM and Sentinel-2 inputs:** Kuro Siwo has only Sentinel-1 and a DEM per tile
  (no Sentinel-2, and HAND cannot be computed on isolated 2.2 km tiles), so such a model cannot be trained on
  the allowed data. FloodSight fuses these sources at the decision level instead (terrain limits, Sentinel-2
  confirmation, model + rules), which is evaluated end to end on EMS maps.

## Set actually trained (decided 10 Oct 2026, BEFORE any comparison model was trained)

Time allows six deep models. The set follows the Kuro Siwo benchmark (Bountos et al., NeurIPS 2024, Table 2), where
U-Nets with the dates stacked as channels scored best (UNet-ResNet50, flood F1 80.1) and dedicated change-detection
networks lower (ChangeFormer 77.0, FC-EF 75.1, SNUNet-CD 74.3):

| `--arch` / encoder | Why |
|---|---|
| `siamese` / ResNet-34 | ours (same budget as the others, retrained for a fair comparison) |
| `early_fusion` / ResNet-34 | the benchmark's best family (dates stacked), same backbone as ours |
| `early_fusion` / ResNet-50 | the benchmark's best model (UNet-ResNet50); does a bigger encoder help? |
| `post_only` / ResNet-34 | does the before-image help? Also the only model that can be scored on Sen1Floods11 (one image per chip) |
| `siamese_segformer` | ChangeFormer-style transformer change model |
| `siamese_deeplabv3p` | DeepLab family from the benchmark, with our two-date fusion |
| gradient-boosted trees, physical rules | non-deep baselines |

Not trained (time): U-Net++ (the SNUNet-CD idea scored lowest in the benchmark), attention U-Net, SegFormer with
stacked dates, terrain-loss variant. They stay in the code and are listed as not run.

Second listed dataset: the after-only model is also scored on **Sen1Floods11** (Bonafilia et al. 2020) hand-labelled
test split and its Bolivia hold-out (`scripts/eval_sen1floods11.py`), which it never saw. The two-date models cannot
be scored there because Sen1Floods11 has no before-image.

## Commands (Windows PowerShell, from the floodsight folder)

```
pip install segmentation-models-pytorch scikit-learn
$common = "--root D:\data\KuroSiwo\GRD\data --epochs 15 --batch 8 --pretrained --val-events 1111013 --test-events 427 1111006 --exclude-events 205 421 498 --max-per-event 2000"
Invoke-Expression "python -m floodsight.model.train $common --arch siamese --encoder resnet34 --out models\cmp_siamese.pt"
Invoke-Expression "python -m floodsight.model.train $common --arch post_only --encoder resnet34 --out models\cmp_post_only.pt"
Invoke-Expression "python -m floodsight.model.train $common --arch early_fusion --encoder resnet34 --out models\cmp_early_fusion.pt"
Invoke-Expression "python -m floodsight.model.train $common --arch early_fusion --encoder resnet50 --out models\cmp_early_fusion_r50.pt"
Invoke-Expression "python -m floodsight.model.train $common --arch siamese_segformer --out models\cmp_siamese_segformer.pt"
Invoke-Expression "python -m floodsight.model.train $common --arch siamese_deeplabv3p --out models\cmp_siamese_deeplabv3p.pt"

python scripts\compare_models.py --root D:\data\KuroSiwo\GRD\data --gbm models\cmp_gbm.joblib --train-gbm --extra-events 205 `
  --weights models\cmp_siamese.pt models\cmp_post_only.pt models\cmp_early_fusion.pt models\cmp_early_fusion_r50.pt `
  models\cmp_siamese_segformer.pt models\cmp_siamese_deeplabv3p.pt

python scripts\eval_sen1floods11.py --root D:\data\Sen1Floods11 --weights models\cmp_post_only.pt
```
An interrupted model continues with the same command plus `--resume`. If a model does not fit in memory, lower
`--batch` for that model only (reported). The production model (`models/siamese_unet.pt`) is not touched.

## Results

(filled in from `models/model_comparison.json` after the runs; models added as they finish, numbers as printed)

### Kuro Siwo held-out test events (printed once at the end of each training run)

| Model | Val 1111013 flood IoU (best epoch) | Test 1111006 IoU | Test 427 IoU | Test all: IoU / F1 |
|---|---|---|---|---|
| Siamese U-Net R34 (ours, same budget) | 0.700 (10 of 15) | 0.725 | 0.225 | **0.643 / 0.782** |
| early fusion U-Net R34 | 0.690 (14 of 15) | 0.714 | 0.254 | 0.619 / 0.765 |
| early fusion U-Net R50 (benchmark's best family) | 0.715 (14 of 15) | 0.712 | 0.226 | 0.627 / 0.771 |
| Siamese DeepLabV3+ R34 | **0.721** (15 of 15) | 0.710 | 0.233 | 0.626 / 0.770 |
| Siamese SegFormer MiT-B2 | 0.600 (10 of 15) | 0.679 | 0.261 | 0.590 / 0.742 |
| after-only U-Net R34 (no before-image) | 0.513 (11 of 15) | 0.504 | 0.115 | 0.398 / 0.570 |

All trained 10-11 Oct 2026 with the same split and budget (15 epochs, batch 8, ImageNet start, 2000 tiles per
event); no batch size had to be lowered. Test rows were printed once at the end of each run and never used to
choose anything. The same-tile comparison with gradient-boosted trees, the physical rules and the unseen event
205 comes from `compare_models.py` (below, when run).

What the table shows (test, combined): the two-date models are all close (0.59-0.64); dropping the before-image
costs about 0.24 IoU (0.643 -> 0.398), which supports the change-detection design. Validation scores move by up
to +-0.07 from one epoch to the next, so differences of a few hundredths between the two-date models are within
that noise.

427 has few flood pixels (0.45 M vs 2.7 M in 1111006), so its score swings a lot; the combined row is the fairer number.

### Second listed dataset: Sen1Floods11 (after-only model only; never used for training or selection)

| Split | Chips | IoU | Precision | Recall | Dataset's per-chip Otsu labels, IoU |
|---|---|---|---|---|---|
| Test (flood_test_data.csv) | 90 | 0.479 | 0.592 | 0.715 | 0.546 |
| Bolivia hold-out | 15 | 0.684 | 0.835 | 0.791 | 0.697 |

Water = permanent water or flood (Sen1Floods11 labels all surface water). Slope input set to 0 (no DEM in the dataset).
The Otsu labels pick a threshold per chip; the model uses one fixed setting for all chips.

### Same-tile comparison (`compare_models.py`, 11 Oct 2026; `models/model_comparison.json`)

Every method scored on the same 600 tiles per event (130 for event 205); each model's flood threshold chosen on
validation 1111013 only. Flood IoU (precision / recall):

| Method | Validation 1111013 | Test 1111006 | Test 427 | **Test all** | Unseen 205 | ms per tile |
|---|---|---|---|---|---|---|
| **Siamese U-Net R34, production** (`siamese_unet.pt`) | **0.758** (0.89/0.84) | 0.685 | 0.183 | 0.653 (0.86/0.73) | 0.457 (0.96/0.47) | 49 |
| Siamese U-Net R34, retrained same budget | 0.703 (0.89/0.77) | 0.714 | 0.161 | **0.686** (0.84/0.79) | 0.492 (0.96/0.50) | 53 |
| Early fusion U-Net R50 | 0.724 (0.87/0.81) | 0.708 | 0.158 | 0.677 (0.81/0.80) | **0.625** (0.84/0.71) | 73 |
| Siamese DeepLabV3+ R34 | 0.726 (0.86/0.82) | 0.704 | 0.178 | 0.673 (0.83/0.78) | 0.577 (0.90/0.62) | 83 |
| Early fusion U-Net R34 | 0.690 (0.87/0.77) | 0.710 | 0.204 | 0.677 (0.80/0.81) | 0.608 (0.83/0.69) | 68 |
| Siamese SegFormer MiT-B2 | 0.610 (0.80/0.72) | 0.646 | 0.216 | 0.618 (0.82/0.72) | 0.584 (0.87/0.64) | 146 |
| After-only U-Net R34 | 0.549 (0.64/0.80) | 0.515 | 0.083 | 0.472 (0.62/0.66) | 0.197 (0.61/0.23) | 65 |
| Gradient-boosted trees | 0.531 (0.69/0.70) | 0.572 | 0.152 | 0.537 (0.76/0.65) | 0.404 (0.90/0.42) | 279 |
| Physical radar rules | 0.658 (0.96/0.68) | 0.502 | 0.078 | 0.479 (0.88/0.51) | 0.288 (0.99/0.29) | 134 |

These differ slightly from the end-of-training numbers above because the tiles are a fixed sample of 600 per
event and the threshold is tuned on validation instead of taking the most likely class.

**Decision (rule 5, decided on validation only).** The decision rule for this case was written down on 11 Oct
BEFORE the production model's validation score was looked up: a challenger must beat the production model's
validation IoU by at least 0.02 as well as the same-budget retrain. Production model validation flood IoU: 0.723
at training time (epoch 6), 0.758 in the same-tile comparison. Best challenger (DeepLabV3+): 0.721 / 0.726. Not
higher, so **the production Siamese U-Net stays**. No threshold or setting changed.

**What the comparison shows.**
- Deep learning helps: +0.17 test IoU over the physical rules and +0.12 over gradient-boosted trees.
- The before-image is essential: removing it costs 0.18-0.21 test IoU.
- The two-date deep models are within 0.07 of each other on test; the production model is in that group.
- The physical rules have the highest precision (0.88-0.99) but miss half the flood: this is why FloodSight uses
  the rules and the model together rather than either alone.
- On the never-used event 205 (Sri Lanka) the early-fusion models generalise better (0.61-0.63) than the Siamese
  models (0.46-0.49). This is reported as found; it was not used to choose, because 205 is a test event. It is
  a candidate for future work, not a change made now.
- Event 427 has few flood pixels and every method scores low there (0.08-0.22).
