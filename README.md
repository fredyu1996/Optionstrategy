# Optionstrategy

## Position Tracking Setup (Google Sheets)

The **📋 My Positions** view persists positions in a Google Sheet so they survive
Streamlit Cloud redeploys. One-time setup:

1. In [Google Cloud Console](https://console.cloud.google.com/), create (or pick) a project.
2. Enable **Google Sheets API** and **Google Drive API**.
3. Create a **service account**, then create a **JSON key** for it and download it.
4. Create a new Google Sheet. In row 1, add these headers (columns A–H):
   `id`, `ticker`, `strategy`, `strike`, `expiry`, `entry_premium`, `contracts`, `entry_date`
5. Click **Share** on the sheet and give **Editor** access to the service
   account's `client_email` (from the JSON).
6. Copy the sheet ID from its URL (`.../d/<SHEET_ID>/edit`).
7. Add to Streamlit secrets (`.streamlit/secrets.toml` locally, or the app's
   **Secrets** on Streamlit Cloud):

   ```toml
   positions_sheet_key = "<SHEET_ID>"

   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "...@....iam.gserviceaccount.com"
   client_id = "..."
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "..."
   ```

Until secrets are set, the My Positions view shows a setup prompt instead of crashing.
## Signal Alerts (Telegram)

A GitHub Actions job (`notify.py`, hourly during US market hours) scans the
S&P 500 for fresh 🟢 Enter signals and your held positions for TRIM/SELL, then
sends new signals to Telegram. De-duplicated via an `alert_state` tab in the
positions sheet (auto-created).

One-time setup:

1. In Telegram, message **@BotFather** → `/newbot` → copy the **bot token**.
2. Get your **chat id**: message **@userinfobot**, or message your new bot and
   open `https://api.telegram.org/bot<token>/getUpdates` to read `chat.id`.
3. In the GitHub repo → **Settings → Secrets and variables → Actions → New
   repository secret**, add:
   - `TELEGRAM_BOT_TOKEN` — the bot token
   - `TELEGRAM_CHAT_ID` — your chat id
   - `POSITIONS_SHEET_KEY` — the same Google Sheet id used by the app
   - `GCP_SERVICE_ACCOUNT` — the full service-account JSON (paste as the value)
4. The service account already shares the sheet; the `alert_state` tab is created
   automatically on the first run.
5. Trigger a manual test: repo → **Actions → Signal Alerts → Run workflow**.
