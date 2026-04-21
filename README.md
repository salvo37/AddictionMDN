# Craving Trajectory Simulator — MDN v10

## Files needed
- `app.py` — Streamlit app
- `mdn_model_v10.keras` — trained MDN model
- `encoder_mdn_v10.keras` — trained encoder
- `mdn_params.json` — normalization parameters
- `requirements.txt` — dependencies

## Deploy on Streamlit Cloud
1. Create a GitHub repository
2. Upload all files above
3. Go to https://share.streamlit.io
4. Connect your GitHub repo
5. Set main file: `app.py`
6. Deploy → get public link

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reference
Bonfiglio, Renati, Penna (2026). Craving Dynamics as a Latent-State System in Addiction. AIRS 2026.
