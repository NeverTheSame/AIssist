# Streamlit Cloud Deployment Guide

## Step-by-Step Setup Instructions

### 1. Commit and Push Your Changes

```bash
# Add all new files
git add streamlit_app.py .streamlit/ Dockerfile .dockerignore STREAMLIT_*.md
git add requirements.txt

# Commit
git commit -m "Add Streamlit web interface for team access"

# Push to GitHub
git push origin streamlit-interface
```

### 2. Sign Up for Streamlit Cloud

1. Go to **https://streamlit.io/cloud**
2. Click **"Sign up"** or **"Get started"**
3. Sign in with your **GitHub account**
4. Authorize Streamlit Cloud to access your repositories

### 3. Deploy Your App

1. In Streamlit Cloud dashboard, click **"New app"**
2. Fill in the deployment form:
   - **Repository**: Select your GitHub repository
   - **Branch**: Select `streamlit-interface`
   - **Main file path**: Enter `streamlit_app.py`
   - **App URL** (optional): Choose a custom subdomain
3. Click **"Deploy"**

### 4. Configure Secrets (Environment Variables)

**Important**: Your `.env` file contains sensitive credentials. Add them to Streamlit Cloud Secrets.

1. In your app's Streamlit Cloud dashboard, go to **"Settings"**
2. Click on **"Secrets"** in the left sidebar
3. Add all environment variables from your `.env` file:

```toml
# AI Service Configuration (Required)
AI_SERVICE_API_KEY = "your-api-key-here"
AI_SERVICE_ENDPOINT = "your-endpoint-here"
AI_SERVICE_API_VERSION = "your-api-version"
AI_SERVICE_DEPLOYMENT_NAME = "your-deployment-name"
AI_SERVICE_MODEL_NAME = "your-model-name"

# Database Configuration (If using Kusto fetcher)
DATABASE_CLUSTER = "https://your-cluster.kusto.windows.net"
DATABASE_NAME = "YourDatabase"
DATABASE_TOKEN_SCOPE = "https://your-cluster.kusto.windows.net/.default"

# Cost Configuration (Optional)
AI_SERVICE_INPUT_COST = "0.01"
AI_SERVICE_OUTPUT_COST = "0.03"

# Vision Service (Optional)
VISION_API_KEY = "your-vision-api-key"
VISION_ENDPOINT = "your-vision-endpoint"

# Article Search (Optional)
DEFAULT_ARTICLES_EMBEDDINGS_PATH = "path/to/your/embeddings"
```

**Note**: In Streamlit Cloud, secrets are accessed via `st.secrets`, but your code uses `os.environ`. We need to make sure the app can read both. Streamlit Cloud automatically loads secrets as environment variables, so your existing code should work!

### 5. Required Files Check

Make sure these files exist in your repository:
- ✅ `streamlit_app.py` - Main app file
- ✅ `prompts.json` - Prompt templates (if not in repo, you'll need to add it)
- ✅ `.streamlit/config.toml` - Streamlit config
- ✅ `requirements.txt` - Dependencies (already updated with streamlit)

**Important**: If `prompts.json` is not in your repository:
- Add it to the repo (even if it's a private file)
- Or modify the code to handle missing prompts gracefully
- Or create a `prompts.json.example` and document how to set it up

### 6. Access Your App

Once deployed:
- Your app will be available at: `https://your-app-name.streamlit.app`
- Share this URL with your team members
- The app will automatically redeploy when you push changes to the branch

### 7. Monitoring and Logs

- **View logs**: In Streamlit Cloud dashboard, click on your app → "Logs"
- **Check status**: Dashboard shows app status and resource usage
- **Restart app**: Use "Reboot app" button if needed

## Troubleshooting

### App Won't Start
- Check logs in Streamlit Cloud dashboard
- Verify all secrets are set correctly
- Ensure `prompts.json` exists in the repository
- Check that Python version is compatible (3.8+)

### Environment Variables Not Working
- Secrets in Streamlit Cloud are case-sensitive
- Make sure variable names match exactly
- Restart the app after adding secrets

### Missing Files Error
- Ensure `prompts.json` is in the repository
- Check that all required files are committed
- Verify file paths in the code match your repo structure

### Database Connection Issues
- Streamlit Cloud cannot access your internal VPN
- Kusto fetcher may not work directly from Streamlit Cloud
- Consider:
  - Using a proxy/API gateway
  - Processing data locally and uploading results
  - Using Azure App Service with VPN instead

## Important Notes

1. **VPN Access**: Streamlit Cloud runs on public servers and cannot access resources behind your VPN. If your Kusto database requires VPN, you may need to:
   - Use Azure App Service instead (can connect to VPN)
   - Set up a proxy/API gateway
   - Pre-process data and upload to cloud storage

2. **Private Repositories**: 
   - Free tier: Public repos only
   - Paid tier: Private repos supported

3. **Resource Limits**:
   - Free tier has CPU/memory limits
   - Apps may sleep after inactivity
   - Consider paid tier for production use

4. **File Storage**:
   - Files created during runtime are ephemeral
   - Use cloud storage (S3, Azure Blob) for persistent data
   - Consider using a database for storing results

## Next Steps After Deployment

1. Test the app with a sample incident
2. Share the URL with your team
3. Monitor usage and performance
4. Set up alerts if needed
5. Consider adding authentication for sensitive data

## Support

- Streamlit Cloud docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit forums: https://discuss.streamlit.io
- Your app logs: Check in Streamlit Cloud dashboard

