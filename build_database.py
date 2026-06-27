"""
build_database.py — run this ONCE locally to index your reference song
library into song_database.pkl, which app.py loads at startup.

Usage:
    python build_database.py path/to/songs_folder

This writes song_database.pkl into the current directory. Commit that
.pkl file to the repo so the deployed app has the database ready to go
(per the assignment: "make sure the indexed song database ships with
the deployed app so it works immediately").
"""

import sys
import pickle

from fingerprint import build_database


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_database.py path/to/songs_folder")
        sys.exit(1)

    song_dir = sys.argv[1]
    print(f"Indexing songs in: {song_dir}")

    data = build_database(song_dir, sr=11025, max_seconds=None)

    print(f"Indexed {len(data['song_list'])} songs:")
    for s in data["song_list"]:
        print(f"  - {s}")

    out_path = "song_database.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nWrote {out_path} ({len(data['db'])} unique hashes).")
    print("Commit this file to your repo before deploying.")


if __name__ == "__main__":
    main()
