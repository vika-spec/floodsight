# Deploying FloodSight for free so the judges can use it

**Host: Hugging Face Spaces (Docker), free "CPU basic" tier** — 2 vCPU, 16 GB RAM, public HTTPS link,
encrypted secrets, no credit card. Why this host: the pipeline needs the full Python geo stack and PyTorch
(~3 GB image) and several GB of RAM for a mountain valley; the free tiers of Render/Railway/Vercel are too
small or cannot run long jobs, and Cloud Run needs a billing account.

What judges get at `https://<user>-floodsight.hf.space`:
- the finished analyses (showcase: Trishuli case study, diagnosis events, demo) — open to everyone, instant;
- the offline demo valley — open to everyone;
- **new analyses of any area and date** — with the access code you give them (protects the team's Copernicus
  quota: daily caps on runs and processing units, area cap; see `server/public.py`).

Things to know (tell the judges):
- A new analysis takes about **5–10 minutes** on the free CPU (no GPU); one runs at a time, others queue.
- A free Space **sleeps after ~48 h without visitors**; the first visit wakes it (1–3 min). Open it yourself
  the morning of judging.
- The disk is not persistent: new analyses disappear when the Space restarts; the showcase is rebuilt each time.

## One-time setup (about 30 minutes, all from the command prompt)

Run these from the inner `floodsight` folder (the one with `run.py`). Replace `<user>` with your Hugging Face
user name.

1. **Account and token:** sign up at https://huggingface.co (free), then create a token with *write* access at
   https://huggingface.co/settings/tokens.
   ```
   pip install -U huggingface_hub
   hf auth login
   ```
2. **Create the Space** (Docker, free CPU, public):
   ```
   hf repos create <user>/floodsight --type space --sdk docker
   ```
3. **Secrets:** add one line to your `.env` (a long code you give only to the judges), e.g.
   `FLOODSIGHT_ACCESS_CODE=floods-judge-7Kq2`, then send all secrets from `.env` to the Space:
   ```
   hf spaces secrets add <user>/floodsight --secrets-file .env
   ```
   Secrets are encrypted on Hugging Face and never appear in the code, the logs or the web page.
   Optional limits (also as secrets/variables): `FLOODSIGHT_MAX_KM2` (1600), `FLOODSIGHT_MAX_RUNS_DAY` (20),
   `FLOODSIGHT_PU_DAY` (600; the free Copernicus quota is ~10,000 a month for everything).
4. **Showcase runs** (after your final runs):
   ```
   python scripts/make_showcase.py trishuli_shub slo_meza slo_mura demo_2026-08-26
   ```
   This copies them to `showcase/` without the raw satellite inputs.
5. **Upload the code, the model and the showcase** (never `.env`, raw runs or EMS files):
   ```
   hf upload <user>/floodsight . . --repo-type space --exclude ".env" --exclude "runs/*" --exclude "validation_data/*" --exclude "data/*" --exclude "web/node_modules/*" --exclude "_backups/*" --exclude "**/__pycache__/*" --exclude ".git/*"
   ```
   Large files (the model `models/siamese_unet.pt`) are stored with Git LFS automatically.
6. **Watch the build** (~10 minutes the first time), then open `https://<user>-floodsight.hf.space`:
   ```
   hf spaces logs <user>/floodsight
   ```
   Check: the showcase runs open, the demo runs, and a small new area runs with the access code.

## Updating
Re-run step 5 after any code change (and step 4 if the showcase changed); the Space rebuilds automatically.

## Same container on your own computer (optional)
```
docker build -t floodsight .
docker run -p 7860:7860 --env-file .env -e FLOODSIGHT_ACCESS_CODE=test floodsight
```
Then open http://localhost:7860.

## Data and licences on a public server
Sentinel data: Copernicus free, full and open policy. OSM: ODbL, attributed on the map and in reports.
EMS reference results shown in validation: "European Union, Copernicus Emergency Management Service data".
The raw EMS files are not uploaded (`validation_data/` is excluded).
