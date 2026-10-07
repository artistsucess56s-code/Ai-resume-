# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking Systems (ATS) and gives
specific, prioritized suggestions to improve it. Powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for targeted keyword matching
- **ATS score (0-100)** combining:
  - 70% AI evaluation (keywords, content quality, structure) from Gemini
  - 30% rule-based format checks (contact info, sections, length, bullets, action verbs, metrics)
- Matched vs. missing keywords
- Prioritized improvements with rewritten example lines
- Resume text is processed in memory and never stored by the app

## Run locally
```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey and either paste it into the
sidebar, or set it once:

```bash
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

Or create `.streamlit/secrets.toml` (already git-ignored):

```toml
GEMINI_API_KEY = "your-key"
```

To use a different Gemini model, change it in the sidebar or set `GEMINI_MODEL`.
The default is `gemini-2.5-flash`.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** -> select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings** -> **Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Project structure
```
app.py             # Streamlit app
requirements.txt   # Python dependencies
README.md
.gitignore
```

## Limitations
- Scanned/image-only resumes can't be read; use a text-based PDF or DOCX.
- The score is an estimate. Real ATS products differ, so use it as guidance, not a guarantee.
- Multi-column or heavily designed templates may extract out of order, which is itself a useful signal.
