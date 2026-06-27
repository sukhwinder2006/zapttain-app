"""
EE200 Course Project — Q3B: Signals to Softwares ("Zapptain America")
Streamlit app wrapping the Shazam-style audio fingerprinting identifier
built in Q3A.

Two modes (selectable in the sidebar):
  1. Single-clip mode — upload one query clip, see the spectrogram,
     constellation of peaks, offset histogram, and the identified song.
  2. Batch mode — upload multiple query clips, get a results.csv with
     columns: filename, prediction (matched song's filename, no extension).

Run locally with:  streamlit run app.py
"""

import gzip
import io
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from fingerprint import (
    load_audio,
    fingerprint_audio,
    match_query,
)

st.set_page_config(page_title="Sonic Signatures — Song Identifier", layout="wide", page_icon="📡")

DB_PATH = os.path.join(os.path.dirname(__file__), "song_database.pkl")
DB_PATH_GZ = os.path.join(os.path.dirname(__file__), "song_database.pkl.gz")

# ---------------------------------------------------------------------------
# Light visual theming — signal/oscilloscope palette. Dark panel background,
# a "signal green" accent for matches, monospace for anything data-flavored.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg: #0B0F14;
    --panel: #121822;
    --line: #243140;
    --signal: #39FF8A;
    --amber: #FFB347;
    --text: #E8EDF2;
    --muted: #7C8B9C;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
}

.stApp {
    background-color: var(--bg);
}

h1, h2, h3 {
    color: var(--text) !important;
    letter-spacing: 0.02em;
}

.eyebrow {
    color: var(--signal);
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}

.scope-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}

.match-readout {
    background: var(--panel);
    border: 1px solid var(--signal);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 0 18px rgba(57, 255, 138, 0.12);
}

.match-readout .label {
    color: var(--muted);
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.match-readout .value {
    color: var(--signal);
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 0.1rem;
}

[data-testid="stMetricValue"] {
    color: var(--signal) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

hr {
    border-color: var(--line) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_database():
    """
    Loads song_database.pkl.gz if present (gzip-compressed, to stay under
    GitHub's 25 MB web-upload limit), otherwise falls back to a plain
    song_database.pkl.
    """
    if os.path.exists(DB_PATH_GZ):
        with gzip.open(DB_PATH_GZ, "rb") as f:
            data = pickle.load(f)
    elif os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            data = pickle.load(f)
    else:
        st.error(
            "No song database found. Expected `song_database.pkl` or "
            "`song_database.pkl.gz` in the same folder as app.py. "
            "Run build_database.py and commit the resulting file to the repo."
        )
        st.stop()
    return data


def save_uploaded_to_tempfile(uploaded_file):
    """Streamlit's UploadedFile has no real path on disk; write it to a temp
    file so ffmpeg (used inside fingerprint.load_audio) can read it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def plot_spectrogram(fp):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#0B0F14")
    ax.pcolormesh(fp["t"], fp["f"], fp["Sxx_db"], shading="gouraud", cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)", color="#E8EDF2")
    ax.set_ylabel("Frequency (Hz)", color="#E8EDF2")
    ax.set_title("Spectrogram", color="#E8EDF2")
    ax.tick_params(colors="#7C8B9C")
    for spine in ax.spines.values():
        spine.set_color("#243140")
    fig.tight_layout()
    return fig


def plot_constellation(fp):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0B0F14")
    ax.set_facecolor("#0B0F14")
    ax.pcolormesh(fp["t"], fp["f"], fp["Sxx_db"], shading="gouraud", cmap="gray_r", alpha=0.35)
    ax.scatter(
        fp["t"][fp["time_idx"]], fp["f"][fp["freq_idx"]],
        s=14, c="#39FF8A", marker="o", edgecolors="black", linewidths=0.3,
    )
    ax.set_ylim(0, 4000)
    ax.set_xlabel("Time (s)", color="#E8EDF2")
    ax.set_ylabel("Frequency (Hz)", color="#E8EDF2")
    ax.set_title(f"Constellation map ({len(fp['freq_idx'])} peaks)", color="#E8EDF2")
    ax.tick_params(colors="#7C8B9C")
    for spine in ax.spines.values():
        spine.set_color("#243140")
    fig.tight_layout()
    return fig


def plot_offset_histogram(offset_counts, best_offsets, top_songs):
    fig, axes = plt.subplots(1, min(3, len(top_songs)), figsize=(13, 4), squeeze=False)
    fig.patch.set_facecolor("#0B0F14")
    axes = axes[0]
    for ax, song in zip(axes, top_songs):
        ax.set_facecolor("#0B0F14")
        counter = offset_counts[song]
        best_off = best_offsets[song]
        window = 60
        offs = np.arange(best_off - window, best_off + window)
        counts = [counter.get(o, 0) for o in offs]
        ax.bar(offs, counts, width=1.0, color="#FFB347")
        ax.set_title(f"{song}\npeak={counter[best_off]}", fontsize=9, color="#E8EDF2")
        ax.set_xlabel("Offset (time bins)", color="#E8EDF2")
        ax.set_ylabel("Count", color="#E8EDF2")
        ax.tick_params(colors="#7C8B9C")
        for spine in ax.spines.values():
            spine.set_color("#243140")
    fig.suptitle("Offset histograms — top candidate songs", color="#E8EDF2")
    fig.tight_layout()
    return fig


st.markdown('<div class="eyebrow">EE200 · Q3B · Audio Fingerprinting</div>', unsafe_allow_html=True)
st.title("📡 Sonic Signatures")
st.caption(
    "A Shazam-style identifier: spectrogram → sparse constellation of peaks → "
    "paired (f1,f2,Δt) hashes → offset-histogram matching against an indexed song database."
)

data = load_database()
db = data["db"]
song_list = [os.path.splitext(s)[0] for s in data["song_list"]]

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Single-clip mode", "Batch mode"])
    st.markdown("---")
    st.write(f"**Indexed songs:** {len(song_list)}")
    with st.expander("Show song list"):
        for s in song_list:
            st.write("- " + s)

if mode == "Single-clip mode":
    st.subheader("Single-clip identification")
    uploaded = st.file_uploader(
        "Upload a query clip (mp3/wav/m4a)", type=["mp3", "wav", "m4a", "ogg"]
    )

    if uploaded is not None:
        tmp_path = None
        ranked = []
        try:
            with st.spinner("Loading and fingerprinting audio..."):
                tmp_path = save_uploaded_to_tempfile(uploaded)
                audio, sr = load_audio(tmp_path, sr=11025, max_seconds=20)
                fp = fingerprint_audio(audio, sr)
                ranked, offset_counts, best_offsets = match_query(fp["hashes"], db, top_k=5)
        except Exception as e:
            import traceback
            st.error(f"Failed to process this file: {e}")
            with st.expander("Full error details (debug)"):
                st.code(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if ranked and ranked[0][1] > 0:
            best_song, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0
            margin = best_score / max(second_score, 1)

            st.markdown(
                f"""<div class="match-readout">
                    <div class="label">Identified song</div>
                    <div class="value">🎯 {best_song}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            col1, col2, col3 = st.columns(3)
            col1.metric("Top match score", best_score)
            col2.metric("Runner-up score", second_score)
            col3.metric("Decisiveness margin", f"{margin:.1f}×")

            st.markdown("##### Top candidates")
            st.table(pd.DataFrame(ranked, columns=["Song", "Score"]))

            st.markdown("##### Intermediate steps")
            tab1, tab2, tab3 = st.tabs(["Spectrogram", "Constellation", "Offset histogram"])
            with tab1:
                fig1 = plot_spectrogram(fp)
                st.pyplot(fig1)
                plt.close(fig1)
            with tab2:
                fig2 = plot_constellation(fp)
                st.pyplot(fig2)
                plt.close(fig2)
            with tab3:
                top_songs = [s for s, _ in ranked[:3]]
                fig3 = plot_offset_histogram(offset_counts, best_offsets, top_songs)
                st.pyplot(fig3)
                plt.close(fig3)
        elif ranked:
            st.warning("No confident match found for this clip.")

else:
    st.subheader("Batch identification")
    st.write(
        "Upload multiple query clips. Each will be identified and written to "
        "`results.csv` with columns **filename, prediction** (matched song's "
        "filename without extension)."
    )
    uploaded_files = st.file_uploader(
        "Upload query clips (mp3/wav/m4a)",
        type=["mp3", "wav", "m4a", "ogg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(f"Run batch identification on {len(uploaded_files)} clips"):
            rows = []
            progress = st.progress(0.0)
            status = st.empty()

            for i, uploaded in enumerate(uploaded_files):
                status.write(f"Processing: {uploaded.name}")
                tmp_path = save_uploaded_to_tempfile(uploaded)
                try:
                    audio, sr = load_audio(tmp_path, sr=11025, max_seconds=20)
                    fp = fingerprint_audio(audio, sr)
                    ranked, _, _ = match_query(fp["hashes"], db, top_k=1)
                    prediction = ranked[0][0] if ranked and ranked[0][1] > 0 else "NO_MATCH"
                except Exception:
                    prediction = "ERROR"
                finally:
                    os.unlink(tmp_path)

                rows.append({"filename": uploaded.name, "prediction": prediction})
                progress.progress((i + 1) / len(uploaded_files))

            status.write("Done.")
            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )
