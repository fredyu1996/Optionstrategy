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