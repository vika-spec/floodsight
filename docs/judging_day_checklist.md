# Judging-day checklist

## Day before
- [ ] `python scripts/check_setup.py --login` → all OK (openEO token cached, CDSE .env set)
- [ ] `python scripts/rehearse.py` → PASS on all presets incl. the dry-season control
- [ ] `pytest` → all pass
- [ ] `models/siamese_unet.pt` present; the web app's start page shows every "Ready for live satellite data?" line green
- [ ] `python serve.py` opens the app; open one finished run on the laptop AND on the projector
- [ ] Trishuli run cached in `runs/` (instant fallback if the network is slow)
- [ ] Laptop on power, sleep disabled, phone hotspot as backup internet

## During the live evaluation
1. `python serve.py` → Draw a box on the map (or type the judges' corners) → flood date → Analyse this area.
2. Narrate the numbered progress list: catalogue search → same-track pair → download → detection → damage → cut-off.
3. Show: verdict + numbers → Compare tab (swipe before/after on the map) → Cut off tab (click a village) →
   Flow path (click upstream) → Report tab (EN/NE PDF) → Validation tab on the Trishuli run.
4. If openEO is slow: open the cached Trishuli run from "Earlier analyses" meanwhile; or More options →
   Radar processing → On this computer only.
5. If "No usable satellite data": explain why (no same-track pass yet) — that is the correct, honest answer.

## Questions to be ready for
- Why same orbit track? · Why HAND? · How do you know a village is cut off and not just outside the map?
- How is post-event OSM excluded? · Where can the model be wrong? · Why not just use an LLM for the report?
