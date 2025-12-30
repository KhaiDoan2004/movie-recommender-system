"""
Download missing poster IMAGES only (no posters.json URL file)
- Uses existing posters_mapping.json (movieId -> local_path) to resume
- For missing ones: call TMDB detail API to get poster_path (not saved), then download image
- Saves posters as posters/{movieId}.jpg (stable + easy resume)
"""

import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# ===== Config =====
TMDB_API_KEY = "aa562e984bf91c35a272732b9786f4bf"  # nên đưa vào env sau, nhưng để bạn chạy trước đã

# Lấy root dự án (3 levels up từ recommender/utils/)
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = str(BASE_DIR / "data_cleaned")
POSTERS_DIR = str(BASE_DIR / "posters")
OUTPUT_FILE = os.path.join(DATA_DIR, "posters_mapping.json")
LINKS_FILE = os.path.join(DATA_DIR, "links_cleaned.csv")

os.makedirs(POSTERS_DIR, exist_ok=True)

lock = threading.Lock()


def to_abs_path(p: str) -> str:
    """Convert mapping path to absolute path (fix lỗi do working directory)."""
    if not p:
        return p
    p = p.replace("/", os.sep).replace("\\", os.sep)
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(BASE_DIR, p))


def to_rel_path(p: str) -> str:
    """Store path as relative to project root for portability."""
    p_abs = to_abs_path(p)
    try:
        return os.path.relpath(p_abs, start=BASE_DIR)
    except Exception:
        return p_abs


def file_ok(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def load_existing_mapping() -> dict:
    if not os.path.exists(OUTPUT_FILE):
        return {}
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize to absolute for checking
        for k, v in list(data.items()):
            data[k] = to_abs_path(v)
        print(f"✅ Loaded {len(data)} posters from mapping file")
        return data
    except Exception as e:
        print(f"⚠️ Could not load mapping file: {e}")
        return {}


def atomic_save_json(path: str, data: dict):
    tmp = path + ".tmp"
    # store relative paths for portability
    data_rel = {str(k): to_rel_path(v) for k, v in data.items()}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data_rel, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def get_poster_url_from_tmdb(tmdb_id: int, delay: float = 0.1) -> str | None:
    """Call TMDB detail endpoint to get poster_path, return full image URL (not saved)."""
    if delay > 0:
        time.sleep(delay)
    
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return None

        if r.status_code == 429:
            # rate limit - wait longer
            time.sleep(2.0)
            return None

        return None
    except requests.exceptions.RequestException:
        return None


def download_image(poster_url: str, save_path_abs: str) -> bool:
    """Download image url to local file."""
    try:
        r = requests.get(poster_url, timeout=15, stream=True)
        if r.status_code == 200:
            with open(save_path_abs, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return file_ok(save_path_abs)

        # transient errors
        if r.status_code in (429, 500, 502, 503, 504):
            return False

        return False
    except requests.exceptions.RequestException:
        return False


def process_movie(row, mapping, stats, request_delay=0.1):
    """
    For one movie:
    - if local poster exists -> skip
    - else call TMDB to get poster url -> download -> update mapping
    """
    movie_id = int(row["movieId"])
    tmdb_id = row.get("tmdbId")

    # Expected save path: posters/{movieId}.jpg (ổn định, dễ resume)
    save_path_abs = os.path.join(BASE_DIR, POSTERS_DIR, f"{movie_id}.jpg")

    # If mapping says exists and file ok -> skip
    if str(movie_id) in mapping:
        mapped_abs = to_abs_path(mapping[str(movie_id)])
        if file_ok(mapped_abs):
            with lock:
                stats["skipped"] += 1
            return

    # If file already exists even without mapping -> update mapping & skip
    if file_ok(save_path_abs):
        with lock:
            mapping[str(movie_id)] = save_path_abs
            stats["skipped"] += 1
        return

    # Validate tmdbId
    if pd.isna(tmdb_id) or tmdb_id == 0:
        with lock:
            stats["no_tmdb"] += 1
        return

    try:
        tmdb_id = int(tmdb_id)
    except Exception:
        with lock:
            stats["no_tmdb"] += 1
        return

    # 1) get poster url (not saved) - với delay để tránh rate limit
    poster_url = get_poster_url_from_tmdb(tmdb_id, delay=request_delay)
    if not poster_url:
        with lock:
            stats["no_poster"] += 1
        return

    # 2) download image with small retry
    ok = False
    for attempt in range(3):
        if download_image(poster_url, save_path_abs):
            ok = True
            break
        time.sleep(0.5 * (attempt + 1))

    with lock:
        if ok:
            mapping[str(movie_id)] = save_path_abs
            stats["downloaded"] += 1
        else:
            stats["failed"] += 1


def download_missing_posters():
    print("=" * 80)
    print("DOWNLOAD POSTERS (IMAGE ONLY) - RESUME (FIXED)")
    print("=" * 80)

    print("\n📦 Loading existing posters mapping...")
    mapping = load_existing_mapping()
    print(f"   ✅ Found {len(mapping)} posters in mapping")

    print("\n📦 Loading links data...")
    links = pd.read_csv(LINKS_FILE)
    print(f"   ✅ Loaded {len(links)} links")

    # Build missing list robustly
    missing_rows = []
    for _, row in links.iterrows():
        movie_id = int(row["movieId"])

        mapped = mapping.get(str(movie_id))
        if mapped and file_ok(to_abs_path(mapped)):
            continue

        # if file already exists as posters/{movieId}.jpg -> update mapping later
        expected_abs = os.path.join(BASE_DIR, POSTERS_DIR, f"{movie_id}.jpg")
        if file_ok(expected_abs):
            mapping[str(movie_id)] = expected_abs
            continue

        missing_rows.append(row)

    print(f"\n📊 Statistics:")
    print(f"   - Total movies: {len(links)}")
    print(f"   - Posters valid in mapping (after check): {sum(1 for k,v in mapping.items() if file_ok(to_abs_path(v)))}")
    print(f"   - Missing posters to download: {len(missing_rows)}")

    if not missing_rows:
        print("\n✅ All posters already downloaded!")
        atomic_save_json(OUTPUT_FILE, mapping)
        return

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "no_tmdb": 0, "no_poster": 0}

    # Tối ưu cho số lượng ít (~100 ảnh): giảm workers, save thường xuyên hơn
    num_missing = len(missing_rows)
    if num_missing <= 100:
        # Tối ưu cho ít ảnh: ít workers, save thường xuyên, delay nhỏ
        max_workers = 2
        save_every = 10
        request_delay = 0.3
    elif num_missing <= 500:
        # Trung bình
        max_workers = 4
        save_every = 50
        request_delay = 0.2
    else:
        # Nhiều ảnh: giữ nguyên
        max_workers = 6
        save_every = 200
        request_delay = 0.1

    print(f"\n🚀 Downloading {num_missing} missing posters...")
    print(f"   (Using {max_workers} workers, saving every {save_every} posters)\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_movie, row, mapping, stats, request_delay) for row in missing_rows]

        done = 0
        with tqdm(total=len(futures), desc="Downloading posters", unit="poster") as pbar:
            for fut in as_completed(futures):
                fut.result()
                done += 1
                pbar.update(1)

                # Update progress more frequently for small batches
                update_freq = 10 if num_missing <= 100 else 50
                if done % update_freq == 0:
                    pbar.set_postfix({
                        "✅": stats["downloaded"],
                        "⏭️": stats["skipped"],
                        "❌": stats["failed"],
                        "🚫": stats["no_poster"] + stats["no_tmdb"],
                    })

                # Save more frequently for small batches
                if done % save_every == 0:
                    atomic_save_json(OUTPUT_FILE, mapping)

    print("\n💾 Saving final mapping...")
    atomic_save_json(OUTPUT_FILE, mapping)
    print(f"   ✅ Saved to {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"✅ Downloaded: {stats['downloaded']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print(f"🚫 No TMDB ID: {stats['no_tmdb']}")
    print(f"🚫 No poster available: {stats['no_poster']}")
    print(f"\n📁 Total posters in mapping: {len(mapping)}")
    print(f"📁 Posters directory: {POSTERS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    download_missing_posters()
