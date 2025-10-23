# 🚀 Deployment Guide

## Deploy to Streamlit Cloud (Recommended)

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: India Olympics Medal Predictor"

# Add your GitHub repository
git remote add origin https://github.com/Aryanchhabra/Olympic.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. **Go to:** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure deployment:**
   - Repository: `Aryanchhabra/Olympic`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

5. **Advanced settings (if needed):**
   - Python version: 3.8 or higher
   - No secrets needed for this app

6. **Click "Deploy"**

7. **Wait 2-5 minutes** for deployment to complete

8. **Your app will be live at:**
   - `https://olympic-india-medals.streamlit.app`
   - Or a custom URL assigned by Streamlit

### Step 3: Update README

Once deployed, update the dashboard link in README.md with your actual Streamlit URL.

---

## Alternative: Local Deployment

### Option 1: Run with Python

```bash
cd app
streamlit run streamlit_app.py
```

### Option 2: Run with Docker

```bash
# Build image
docker build -t olympic-dashboard .

# Run container
docker run -p 8501:8501 olympic-dashboard
```

---

## Troubleshooting Deployment

### Issue: "No module named 'streamlit'"
**Solution:** Ensure `requirements.txt` is in the root directory

### Issue: "File not found" errors
**Solution:** Check that all paths in `streamlit_app.py` are relative to the app directory

### Issue: Large repository size
**Solution:** Ensure `.gitignore` excludes large files like datasets (if needed)

---

## Environment Variables (if needed)

If you add any API keys or secrets later:

1. Create `.streamlit/secrets.toml`
2. Add secrets:
   ```toml
   api_key = "your_key_here"
   ```
3. Access in app:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

---

## Post-Deployment Checklist

- [ ] App loads without errors
- [ ] All 4 tabs work correctly
- [ ] Medal counts show 23 (not 140)
- [ ] Charts display properly
- [ ] Predictions work
- [ ] Data download works
- [ ] No broken links
- [ ] Update README with live URL
- [ ] Share on LinkedIn/portfolio

---

**Your dashboard is now live! 🎉**

