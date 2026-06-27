# Sonic Signatures — Audio Fingerprint Identifier

EE200 Course Project, Q3B ("Signals to Softwares"). A Shazam-style song
identifier: spectrogram → constellation of peaks → paired (f1, f2, Δt)
hashes → offset-histogram matching against an indexed song database,
wrapped in a Streamlit app with single-clip and batch modes.

## Repo layout

```
.
├── app.py              Streamlit app (UI, plotting, both modes)
├── fingerprint.py       Core Q3A logic: load/fingerprint/match
├── build_database.py    One-time script to index your song folder
├── song_database.pkl    Pre-built database (commit this!)
├── requirements.txt      Python deps
├── packages.txt          System deps (ffmpeg, for Streamlit Cloud)
└── songs/                Your reference song library (mp3/wav/etc.)
```

## 1. Run it locally

```bash
# create a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# ffmpeg is required by pydub to decode mp3/m4a — install it system-wide
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt-get install ffmpeg
# Windows: download from ffmpeg.org and add to PATH
```

## 2. Build the song database

Put your reference songs (the ones provided for the assignment) into a
folder, e.g. `songs/`, then run:

```bash
python build_database.py songs/
```

This creates `song_database.pkl` in the current directory — **commit
this file to git**. The assignment requires the database to ship with
the deployed app so it works immediately, with no re-indexing step.

## 3. Run the app locally

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`. Try both modes:
- **Single-clip mode** — upload one query clip, see the spectrogram,
  constellation, offset histogram, and the identified song.
- **Batch mode** — upload several clips, download `results.csv`.

## 4. Push to a fresh GitHub repo

```bash
git init
git add .
git commit -m "Initial commit: Sonic Signatures audio fingerprint app"

# create a new EMPTY repo on github.com first, then:
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

If `song_database.pkl` is large (tens of MB), GitHub's normal limit is
100 MB per file, so it should be fine unless your song library is huge.
If it's over 100 MB, look into [Git LFS](https://git-lfs.github.com/).

## 5. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **"New app"**.
3. Pick your repo, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**. Streamlit Cloud will:
   - read `requirements.txt` and `packages.txt` automatically,
   - apt-install `ffmpeg`,
   - pip-install everything else,
   - launch the app.
5. Wait for the build to finish (a couple of minutes the first time).
   You'll get a public URL like `https://<something>.streamlit.app`.

**Test the live URL yourself** before submitting — upload a clip in
single-clip mode and run a small batch — since a broken submission link
scores zero on this part per the assignment instructions.

## 6. What to submit

- The live Streamlit Cloud **app URL**
- A link to this **GitHub repo** (source code)
- Both placed in the same PDF as your Q3A report
- A `.zip` of all your code, per the submission instructions

## Notes on the fingerprinting approach

- Audio is decoded to mono and resampled to 11025 Hz; query clips are
  capped at 20 seconds, since a few seconds of audio is already enough
  for a confident match (and it keeps the app responsive).
- Peaks are local maxima in a sliding 2D window over the spectrogram,
  picked above an adaptive threshold relative to that clip's mean energy.
- Hashes pair each peak with up to 8 nearby later peaks (key = `(f1, f2,
  Δt)`, value = the anchor time `t1`). A true match produces a sharp
  spike in the offset histogram (`song_t1 − query_t1`); a wrong song
  only gets scattered, low coincidental matches.
- `build_database.py` indexes full-length songs (no truncation), so the
  reference fingerprints are as complete as possible.
