# Deploying FloodSight on Azure (Azure for Students credit), at the lowest cost

**Host: Azure Container Apps, Consumption plan, scale to zero.** You pay only while someone is using the app;
when nobody is, it shuts down and costs nothing. Every subscription gets a free monthly grant of
180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests, which covers light judging use.

What costs money, and how we keep it low:

| Item | Setting | Why |
|---|---|---|
| The app | min 0 replicas, max 1 | No cost when idle. One replica, because running analyses are kept in memory |
| Size | 4 vCPU, 8 GiB while running | A mountain valley needs several GB of RAM; faster runs also finish sooner |
| Logs | `--logs-destination none` | No Log Analytics workspace (it bills per GB of logs) |
| Image registry | Azure Container Registry, Basic | The only fixed cost (a few dollars a month). Delete everything after judging |

Check current prices on the Azure pricing page; set a budget alert (step 8) so nothing surprises you.

## One-time setup (Windows PowerShell, from the folder that contains `run.py`)

1. **Azure CLI** (once): `winget install -e --id Microsoft.AzureCLI`, close and reopen PowerShell, then:
   ```
   az login
   az extension add --name containerapp --upgrade
   az provider register --namespace Microsoft.App
   az provider register --namespace Microsoft.OperationalInsights
   az provider register --namespace Microsoft.ContainerRegistry
   ```
2. **Check before building:** `models\siamese_unet.pt` exists (the build refuses to continue without it),
   `web\dist\index.html` exists, and `.env` has `FLOODSIGHT_ACCESS_CODE=<a long code>`.
3. **Showcase runs** (finished analyses judges can open instantly; copies results without raw satellite inputs
   or EMS files):
   ```
   python scripts/make_showcase.py trishuli_shub demo_2026-08-26
   ```
   Use your own run folder names from `runs\`.
4. **Resource group and environment without paid logging.** Students subscriptions may allow only some
   regions; if `centralindia` is refused, try `southindia`, `southeastasia` or `eastus`.
   ```
   az group create -n floodsight-rg -l centralindia
   az containerapp env create -n floodsight-env -g floodsight-rg -l centralindia --logs-destination none
   ```
5. **Build in the cloud and deploy** (no Docker needed on your PC; takes 10-20 minutes the first time):
   ```
   az containerapp up -n floodsight -g floodsight-rg --environment floodsight-env --source . --ingress external --target-port 7860
   ```
6. **Size and scale to zero:**
   ```
   az containerapp update -n floodsight -g floodsight-rg --cpu 4 --memory 8Gi --min-replicas 0 --max-replicas 1
   ```
7. **Secrets from your .env** (values are never typed or shown):
   ```
   powershell -ExecutionPolicy Bypass -File deploy\azure_secrets.ps1
   ```
8. **Link and budget alert:**
   ```
   az containerapp show -n floodsight -g floodsight-rg --query properties.configuration.ingress.fqdn -o tsv
   ```
   Open `https://<that address>`. In the Azure portal: Cost Management, Budgets, add a budget (for example
   10 USD) with an email alert.

## Test it like a judge
On your phone: open the link, open the Trishuli showcase, run a small new area with the access code,
open the situation report and download the CSV files. The first visit after idle takes 1-3 minutes
(the app starts from zero).

## On judging day
Avoid the cold start by keeping one replica warm for the day, then go back to zero:
```
az containerapp update -n floodsight -g floodsight-rg --min-replicas 1
az containerapp update -n floodsight -g floodsight-rg --min-replicas 0
```

## After a code change
Run step 5 again (it rebuilds and redeploys). Secrets and size stay.

## After the hackathon: stop all costs
```
az group delete -n floodsight-rg --yes
```
This deletes the app, the environment and the image registry.

## Never
Never put `.env`, keys, `validation_data` or raw runs in the image (`.dockerignore` excludes them) or in Git.
