# Streamlit Cloud Deployment Checklist

## Pre-Deployment Checklist

### ✅ Required Files
- [ ] `streamlit_app.py` - Main application file
- [ ] `requirements.txt` - Updated with streamlit
- [ ] `.streamlit/config.toml` - Streamlit configuration
- [ ] `prompts.json` - **MUST be in repository** (currently in .gitignore)
- [ ] `config.py` - Configuration loader

### ✅ Files That Should Be in Repository
- [ ] `processor.py` - Core processing logic
- [ ] `transformer.py` - Data transformation
- [ ] `kusto_fetcher.py` - Database fetcher (if using)
- [ ] `article_searcher.py` - Article search (if using)
- [ ] Other supporting Python files

### ⚠️ Important: prompts.json

**CRITICAL**: `prompts.json` is currently in `.gitignore` but is **REQUIRED** for Streamlit Cloud deployment.

**Options:**
1. **Add prompts.json to repository** (recommended for deployment):
   ```bash
   git add -f prompts.json
   git commit -m "Add prompts.json for Streamlit Cloud deployment"
   ```

2. **Or remove from .gitignore** (if it's safe to commit):
   ```bash
   # Remove prompts.json from .gitignore
   git add prompts.json
   git commit -m "Add prompts.json to repository"
   ```

3. **Or create prompts.json.example** and document setup:
   - Create example file
   - Document in README
   - Users must create prompts.json manually

### ✅ Environment Variables Setup

Prepare all environment variables from your `.env` file to add to Streamlit Cloud Secrets:

#### Required:
- [ ] `AI_SERVICE_API_KEY`
- [ ] `AI_SERVICE_ENDPOINT`
- [ ] `AI_SERVICE_API_VERSION`
- [ ] `AI_SERVICE_DEPLOYMENT_NAME`
- [ ] `AI_SERVICE_MODEL_NAME`

#### Optional (if using):
- [ ] `DATABASE_CLUSTER`
- [ ] `DATABASE_NAME`
- [ ] `DATABASE_TOKEN_SCOPE`
- [ ] `VISION_API_KEY`
- [ ] `VISION_ENDPOINT`
- [ ] `DEFAULT_ARTICLES_EMBEDDINGS_PATH`

## Deployment Steps

### Step 1: Prepare Repository
```bash
# Make sure prompts.json is added
git add -f prompts.json  # Force add even if in .gitignore

# Add Streamlit files
git add streamlit_app.py .streamlit/ requirements.txt
git add STREAMLIT_*.md DEPLOY_CHECKLIST.md

# Commit
git commit -m "Add Streamlit interface for team access"

# Push
git push origin streamlit-interface
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select repository and `streamlit-interface` branch
5. Set main file: `streamlit_app.py`
6. Click "Deploy"

### Step 3: Configure Secrets
1. Go to app settings → Secrets
2. Add all environment variables
3. Save and restart app

### Step 4: Test
1. Open your app URL
2. Try processing a test incident
3. Check logs for any errors
4. Verify all features work

## Post-Deployment

- [ ] Test app functionality
- [ ] Share URL with team
- [ ] Monitor logs for errors
- [ ] Set up alerts (if needed)
- [ ] Document access for team

## Troubleshooting

### prompts.json Missing
**Error**: "prompts.json not found"
**Solution**: Add prompts.json to repository (see above)

### Environment Variables Not Working
**Error**: "Configuration is incomplete"
**Solution**: 
- Check Secrets in Streamlit Cloud
- Verify variable names match exactly (case-sensitive)
- Restart app after adding secrets

### Database Connection Issues
**Error**: "Could not connect to database"
**Solution**: 
- Streamlit Cloud cannot access VPN-protected resources
- Consider using Azure App Service with VPN instead
- Or set up API gateway/proxy

## Quick Commands

```bash
# Check what needs to be committed
git status

# Add prompts.json (force, even if in .gitignore)
git add -f prompts.json

# Add all Streamlit files
git add streamlit_app.py .streamlit/ requirements.txt *.md

# Commit and push
git commit -m "Deploy Streamlit interface"
git push origin streamlit-interface
```

