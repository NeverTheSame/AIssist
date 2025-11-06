# Simple Token Authentication Guide

**No enterprise app creation needed!** This is the easiest way to authenticate with Azure/Kusto.

## Quick Start

1. **Get a token on your local machine:**
   ```bash
   python3 get_azure_token.py
   ```
   Or if you don't have the script:
   ```bash
   az login
   az account get-access-token --resource https://icmcluster.kusto.windows.net/.default
   ```
   Copy the `accessToken` value from the output.

2. **Paste token into Streamlit Settings:**
   - Open your Streamlit app
   - Go to ⚙️ Settings page
   - Scroll to "Azure Authentication"
   - Paste the token into "Azure Access Token" field
   - Click "Save Configuration"

3. **Done!** You can now fetch incidents from the database.

## How It Works

- You authenticate on your **local machine** using Azure CLI (which uses your existing Azure credentials)
- Azure CLI gives you a temporary access token (valid for ~1 hour)
- You paste that token into the Streamlit app
- The app uses the token to authenticate with Kusto
- When the token expires, get a new one and paste it again

## Token Expiration

- Tokens typically expire after **1 hour**
- When you get an authentication error, it usually means the token expired
- Simply run `python3 get_azure_token.py` again to get a new token
- Paste the new token into Settings

## Helper Scripts

### Python Script (Recommended)
```bash
python3 get_azure_token.py
```

This will:
- Use Azure CLI to get a token
- Display the token clearly
- Show expiry information
- Provide copy-paste instructions

### Bash Script
```bash
./get_azure_token.sh
```

Same functionality as Python script, but uses bash.

### Manual Azure CLI
```bash
az login
az account get-access-token --resource https://icmcluster.kusto.windows.net/.default
```

Copy the `accessToken` value from the JSON output.

## Advantages

✅ **No enterprise app needed** - Uses your existing Azure credentials  
✅ **Simple** - Just copy and paste  
✅ **Secure** - Token is stored in session state, not on disk  
✅ **Quick setup** - Takes 30 seconds  
✅ **No special permissions required** - Works with standard Azure accounts  

## Troubleshooting

### "Token expired" error
- Get a new token: `python3 get_azure_token.py`
- Paste it into Settings page

### "Azure CLI not found"
- Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
- Then run: `az login`

### "Not logged in" error
- Run: `az login`
- Authenticate in your browser
- Then get the token again

### Token not working
- Make sure you copied the entire token (it's a very long string)
- Check there are no extra spaces before/after
- Verify the token scope matches your Kusto cluster URL

## Alternative: Service Principal

If you **can** create enterprise applications and want longer-lived credentials:
- See `AZURE_AUTH_SETUP.md` for Service Principal setup
- Enter Client ID, Secret, and Tenant ID in Settings page
- Tokens refresh automatically

But for most users, **direct token input is easier and sufficient**.

