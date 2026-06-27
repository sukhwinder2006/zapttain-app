"""
fingerprint.py — Q3A core: Shazam-style audio fingerprinting.

Pipeline:
  load_audio        decode any audio file -> mono float waveform at target sr
  fingerprint_audio  waveform -> spectrogram -> constellation of peaks -> hashes
  build_database     fingerprint a folder of reference songs -> hash -> [(song, offset), ...]
  match_query        compare a query's hashes against the database via offset histograms

A "hash" pairs two nearby spectral peaks (f1, f2) with the time gap (dt)
between them. Two clips of the *same* recording will produce many hashes
that are identical, and — crucially — all anchored at a consistent time
offset (query_time - song_time). A wrong song only produces a handful of
coincidental hash collisions, scattered across random offsets. So the
correct match stands out as a sharp spike in the offset histogram, while
wrong matches stay flat and noisy.
"""

import os
from collections import defaultdict, Counter

import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter


# ---------------------------------------------------------------------------
# 1. Audio loading
# ---------------------------------------------------------------------------

def load_audio(path, sr=11025, max_seconds=20):
    """
    Decode an audio file to mono, resampled to `sr` Hz, truncated/padded
    to at most `max_seconds`. Uses librosa (soundfile/audioread backends)
    so it can read mp3/m4a/ogg/wav without depending on ffmpeg or the
    deprecated `audioop` module (which pydub needs but Python 3.13+ removed).

    Returns
    -------
    audio : np.ndarray, float32, range [-1, 1]
    sr    : int, the sample rate actually used
    """
    import librosa

    duration = max_seconds if max_seconds is not None else None
    audio, sr = librosa.load(path, sr=sr, mono=True, duration=duration)
    return audio.astype(np.float32), sr


# ---------------------------------------------------------------------------
# 2. Spectrogram -> constellation of peaks -> hashes
# ---------------------------------------------------------------------------

def _spectrogram(audio, sr, nperseg=1024, noverlap=512):
    f, t, Sxx = sps.spectrogram(
        audio, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, mode="magnitude"
    )
    # avoid log(0); convert to dB
    Sxx_db = 20 * np.log10(Sxx + 1e-6)
    return f, t, Sxx_db


def _find_peaks(Sxx_db, amp_min=10, neighborhood=(20, 20)):
    """
    Local maxima of the spectrogram that stand out from their surroundings.
    `neighborhood` = (freq_bins, time_bins) window size for the local max filter.
    `amp_min` is a floor in dB above the per-clip noise level to reject quiet peaks.
    """
    struct = np.ones(neighborhood)
    local_max = maximum_filter(Sxx_db, footprint=struct) == Sxx_db

    threshold = Sxx_db.mean() + amp_min
    detected = local_max & (Sxx_db > threshold)

    freq_idx, time_idx = np.where(detected)
    return freq_idx, time_idx


def _make_hashes(freq_idx, time_idx, fan_out=8, min_dt=1, max_dt=200):
    """
    Pair each peak with up to `fan_out` later peaks within [min_dt, max_dt]
    time bins, forming a hash key (f1, f2, dt) -> anchor time (t1).
    """
    order = np.argsort(time_idx)
    freq_idx = freq_idx[order]
    time_idx = time_idx[order]

    hashes = []  # list of ((f1, f2, dt), t1)
    n = len(time_idx)
    for i in range(n):
        f1, t1 = freq_idx[i], time_idx[i]
        count = 0
        for j in range(i + 1, n):
            f2, t2 = freq_idx[j], time_idx[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hashes.append(((int(f1), int(f2), int(dt)), int(t1)))
            count += 1
            if count >= fan_out:
                break
    return hashes


def fingerprint_audio(audio, sr, nperseg=1024, noverlap=512, amp_min=10, fan_out=8):
    """
    Full pipeline: waveform -> spectrogram -> peaks -> hashes.
    Returns a dict with everything the app needs for plotting + matching.
    """
    f, t, Sxx_db = _spectrogram(audio, sr, nperseg=nperseg, noverlap=noverlap)
    freq_idx, time_idx = _find_peaks(Sxx_db, amp_min=amp_min)
    hashes = _make_hashes(freq_idx, time_idx, fan_out=fan_out)

    return {
        "f": f,
        "t": t,
        "Sxx_db": Sxx_db,
        "freq_idx": freq_idx,
        "time_idx": time_idx,
        "hashes": hashes,
    }


# ---------------------------------------------------------------------------
# 3. Database construction
# ---------------------------------------------------------------------------

def build_database(song_dir, sr=11025, max_seconds=None, extensions=(".mp3", ".wav", ".m4a", ".ogg")):
    """
    Fingerprint every song in `song_dir` and build:
        db[hash_key] = [(song_filename, t1), (song_filename, t1), ...]
    plus the list of indexed song filenames.

    max_seconds=None indexes the full track (recommended for the reference
    database — only queries should be clipped).
    """
    db = defaultdict(list)
    song_list = sorted(
        fn for fn in os.listdir(song_dir) if os.path.splitext(fn)[1].lower() in extensions
    )

    for fn in song_list:
        path = os.path.join(song_dir, fn)
        audio, _ = load_audio(path, sr=sr, max_seconds=max_seconds)
        fp = fingerprint_audio(audio, sr)
        for h, t1 in fp["hashes"]:
            db[h].append((fn, t1))

    return {"db": dict(db), "song_list": song_list}


# ---------------------------------------------------------------------------
# 4. Matching a query against the database
# ---------------------------------------------------------------------------

def match_query(query_hashes, db, top_k=5):
    """
    For each song, build a histogram of (song_t1 - query_t1) offsets across
    all matching hashes. A true match produces one dominant spike; a wrong
    song produces a flat, scattered histogram. The song's score is the
    height of its tallest spike.

    Returns
    -------
    ranked        : list of (song_filename_no_ext, score), best first, length top_k
    offset_counts : dict song_filename -> Counter(offset -> count)
    best_offsets  : dict song_filename -> offset with the highest count
    """
    offset_counts = defaultdict(Counter)

    for h, q_t1 in query_hashes:
        matches = db.get(h)
        if not matches:
            continue
        for song_fn, song_t1 in matches:
            offset = song_t1 - q_t1
            offset_counts[song_fn][offset] += 1

    scores = {}
    best_offsets = {}
    for song_fn, counter in offset_counts.items():
        best_offset, best_count = counter.most_common(1)[0]
        scores[song_fn] = best_count
        best_offsets[song_fn] = best_offset

    ranked_all = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked = [(os.path.splitext(fn)[0], score) for fn, score in ranked_all[:top_k]]

    # re-key offset_counts/best_offsets to match the (no-extension) names
    # used in `ranked`, since that's what the app looks up by.
    offset_counts_named = {os.path.splitext(fn)[0]: c for fn, c in offset_counts.items()}
    best_offsets_named = {os.path.splitext(fn)[0]: o for fn, o in best_offsets.items()}

    return ranked, offset_counts_named, best_offsets_named
