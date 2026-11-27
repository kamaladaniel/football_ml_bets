# Deploy this app so you can open it on your phone

You already have a working Streamlit frontend (`app.py`).  
Below is the practical path to free hosting + phone access.

---

## 1. Recommended free host: Streamlit Community Cloud

**Why this one**
- Made for Streamlit
- Free for public apps
- Auto-redeploys when you push to GitHub
- Supports secrets (put your API token there, not in code)
- Works well on mobile browsers

**Limits (important)**
- Free apps **sleep** after ~1 hour of inactivity. First visit after sleep can take 30–60 seconds to wake.
- Resource limits are modest. Training many leagues at once can time out or get killed.
- **Mitigation**: disable most leagues in `config/leagues.yaml` and only enable the 5–10 you actually care about.

### Steps

1. Create a free GitHub account (if you don’t have one).
2. Create a new repository (public is easiest for the free tier).
3. Push this whole `football_ml_project` folder to the repo root.
4. Go to https://share.streamlit.io and sign in with GitHub.
5. Click **New app** → select your repo → set Main file path to `app.py`.
6. Under **Advanced settings → Secrets**, add:

```toml
BSD_API_TOKEN = "your_real_token_here"
```

7. Deploy. After a minute you get a public URL like  
   `https://yourname-football-ml-xxxxx.streamlit.app`

Open that URL on your phone. Add it to your home screen for an app-like feel.

### Important code change before sharing

In `src/utils.py` the function `get_api_token()` still has a hardcoded fallback token.  
Before you push to a **public** repo, change it to:

```python
def get_api_token():
    import os
    token = os.environ.get("BSD_API_TOKEN")
    if not token:
        # Streamlit Cloud injects secrets as env vars automatically
        try:
            import streamlit as st
            token = st.secrets.get("BSD_API_TOKEN")
        except Exception:
            pass
    if not token:
        raise RuntimeError(
            "BSD_API_TOKEN not found. Set the environment variable or Streamlit secret."
        )
    return token
```

(The Streamlit Cloud secrets system automatically exposes them as environment variables, so the original `os.environ.get` already works once the secret is set. Just remove the hardcoded fallback string.)

---

## 2. Alternative free hosts

| Host | Pros | Cons |
|------|------|------|
| **Hugging Face Spaces** | Free, good for ML, Docker or Streamlit template | Slightly more setup; also sleeps |
| **Render** | Free tier available | Needs credit card sometimes; limited free hours |
| **Railway** | Easy | Free tier is now very limited |
| **Local + Cloudflare Tunnel / ngrok** | Full control, free | Your computer must stay on |

For “open on my phone when I’m not home”, **Streamlit Community Cloud** or **Hugging Face Spaces** are the realistic free options.

---

## 3. Make it usable on free tiers

1. **Reduce leagues**  
   Edit `config/leagues.yaml` and set `enabled: false` on everything you don’t need.  
   Aim for ≤ 8–10 enabled leagues. Each league does download + walk-forward validation + fit.

2. **Reuse data when possible**  
   In the sidebar turn off “Download fresh data” after the first successful run of the day. Cached CSVs in `data/raw/` are reused (note: on Streamlit Cloud the filesystem is ephemeral, so data disappears when the app sleeps unless you use a persistent volume or external storage).

3. **Odds**  
   Upload `odds.csv` from the sidebar each session, or keep using the Fair odds columns and compare manually with your bookmaker.

4. **Timeouts**  
   If a run is killed, reduce leagues further or set `history_seasons: 2` in the settings block of `leagues.yaml`.

---

## 4. Local phone access (quick test, no hosting)

If you just want to try the phone UI while at home:

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

Then open `http://YOUR_PC_LAN_IP:8501` on your phone (same Wi-Fi).

For access from outside your house without a public host, use a tunnel:

```bash
# one-time
npx localtunnel --port 8501
# or
ngrok http 8501
```

The tunnel URL works from anywhere while your PC is on.

---

## 5. Checklist before first deploy

- [ ] Remove or blank the hardcoded API token in `src/utils.py`
- [ ] Put the real token in Streamlit Secrets / environment variable
- [ ] Disable unused leagues in `config/leagues.yaml`
- [ ] `requirements.txt` includes `streamlit>=1.32.0` (already done)
- [ ] Main file is `app.py`
- [ ] You understand free apps sleep and the first open can be slow

---

## 6. What the Streamlit app gives you

- Mobile-friendly dark UI
- Date picker + “only today’s fixtures” toggle
- Optional fresh download toggle
- Odds.csv uploader
- Big “Value bets” cards (the only thing that matters)
- Full predictions table + CSV download
- Last-run results are kept in session (and written to `data/predictions/` when possible)

Run it the same way you used to run `main.py` — just through the browser instead of the terminal.
