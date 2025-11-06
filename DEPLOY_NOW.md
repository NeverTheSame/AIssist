# 🚀 Deploy to Streamlit Cloud Now!

## Quick Start Guide

### Step 1: Add prompts.json (REQUIRED)

Your `prompts.json` file is currently ignored by git. You need to add it to the repository:

```bash
# Check if prompts.json exists
ls -la prompts.json

# If it exists, add it to the repository (force add even though it's in .gitignore)
git add -f prompts.json

# Commit it
git commit -m "Add prompts.json for Streamlit Cloud deployment"
```

**⚠️ Important**: If `prompts.json` doesn't exist, you need to create it first. Check your local `.env` setup or create it based on the README.md examples.

### Step 2: Commit All Changes

```bash
# Review what's staged
git status

# Commit all Streamlit files
git commit -m "Add Streamlit web interface for team access

- Add streamlit_app.py with web UI
- Add Streamlit configuration
- Add deployment documentation
- Update requirements.txt with streamlit"

# Push to GitHub
git push origin streamlit-interface
```

### Step 3: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**: https://streamlit.io/cloud
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the form**:
   - Repository: Select your repository
   - Branch: `streamlit-interface`
   - Main file path: `streamlit_app.py`
   - App URL: Choose a name (optional)
5. **Click "Deploy"**

### Step 4: Configure Secrets

After deployment, go to your app settings:

1. Click on your app in the dashboard
2. Go to **"Settings"** → **"Secrets"**
3. Add all environment variables from your `.env` file:

```toml
AI_SERVICE_API_KEY = "your-key-here"
AI_SERVICE_ENDPOINT = "your-endpoint-here"
AI_SERVICE_API_VERSION = "your-version"
AI_SERVICE_DEPLOYMENT_NAME = "your-deployment"
AI_SERVICE_MODEL_NAME = "your-model"
DATABASE_CLUSTER = "your-cluster"
DATABASE_NAME = "your-database"
DATABASE_TOKEN_SCOPE = "your-scope"
```

4. **Save** and the app will restart automatically

### Step 5: Test Your App

1. Open your app URL (e.g., `https://your-app-name.streamlit.app`)
2. Try processing a test incident
3. Check the "Logs" tab in Streamlit Cloud if there are any errors

## ⚠️ Important Notes

### VPN Access Limitation
Streamlit Cloud **cannot access resources behind your VPN**. If your Kusto database requires VPN:
- The Kusto fetcher may not work directly from Streamlit Cloud
- Consider pre-processing data locally and uploading results
- Or use Azure App Service with VPN connection instead

### prompts.json Requirement
- Must be in the repository for Streamlit Cloud
- Currently in `.gitignore`, so use `git add -f prompts.json`
- If sensitive, consider using Streamlit Secrets instead

### Environment Variables
- All `.env` variables must be added to Streamlit Cloud Secrets
- Variables are case-sensitive
- App restarts automatically after saving secrets

## Troubleshooting

### App won't start
- Check logs in Streamlit Cloud dashboard
- Verify `prompts.json` is in repository
- Ensure all secrets are configured

### prompts.json not found
- Run: `git add -f prompts.json && git commit -m "Add prompts.json"`
- Push changes: `git push origin streamlit-interface`
- Redeploy in Streamlit Cloud

### Database connection errors
- Streamlit Cloud cannot access VPN-protected resources
- Consider alternative deployment (Azure App Service) or API gateway

## Next Steps

1. ✅ Deploy the app
2. ✅ Configure secrets
3. ✅ Test with a sample incident
4. ✅ Share URL with your team: `https://your-app-name.streamlit.app`
5. ✅ Monitor usage and logs

## Need Help?

- Check `STREAMLIT_CLOUD_SETUP.md` for detailed instructions
- Review `DEPLOY_CHECKLIST.md` for complete checklist
- Check Streamlit Cloud logs for error messages
- Review Streamlit docs: https://docs.streamlit.io/streamlit-community-cloud

