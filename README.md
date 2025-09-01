<<<<<<< HEAD
# Welcome to streamlit

This is the app you get when you run `streamlit hello`, extracted as its own app.

Edit [Hello.py](./Hello.py) to customize this app to your heart's desire. ❤️

Check it out on [Streamlit Community Cloud](https://st-hello-app.streamlit.app/)
=======
# TACO Streamlit — Step 1 (Local Model in Same Folder)

## Files
- streamlit_app.py
- requirements.txt
- best.pt  ← put your YOLO weights file here (rename your file to best.pt)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate       # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
streamlit run streamlit_app.py
```

This opens http://localhost:8501 — upload an image or use webcam, then click "Run detection".

## Notes
- If you name the weights differently, set an env var: `export LOCAL_MODEL=myweights.pt`
- If PyTorch installation is slow, be patient. On Apple Silicon: `pip install 'torch==2.4.*' --extra-index-url https://download.pytorch.org/whl/cpu`
- If you see OpenCV display errors, we already use `opencv-python-headless` to avoid GUI deps.
>>>>>>> b_note
