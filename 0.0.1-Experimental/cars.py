
import csv
import io
import json
import os
import time
import hashlib
import threading

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =====================================
# CONFIGURATION
# =====================================

ROOT = Path("dataset")
IMAGES = ROOT / "images"

ROOT.mkdir(exist_ok=True)
IMAGES.mkdir(exist_ok=True)

TARGET = 1000
WORKERS = 6
PAGE_SIZE = 20
MAX_PAGES = 100

TIMEOUT = (15, 40)
MAX_SIZE = 20 * 1024 * 1024
MIN_SIZE = 128

API = "https://api.openverse.org/v1/images/"

QUERIES = [
    "car",
    "automobile",
    "sports car",
    "classic car",
    "vintage automobile",
    "electric vehicle",
    "racing car",
    "pickup truck",
    "luxury automobile",
    "car photography",
    "sedan",
    "SUV"
]

LICENSES = [
    "cc0", "pdm", "by",
    "by-sa", "by-nc", "by-nc-sa"
]

PROGRESS = ROOT / "progress.json"
CAPTIONS = ROOT / "captions.csv"
ATTRIBUTIONS = ROOT / "attributions.csv"

FIELDS = [
    "filename", "caption", "title",
    "creator", "creator_url", "license",
    "license_version", "license_url",
    "attribution", "source_url",
    "image_url", "image_id",
    "image_hash", "search_query"
]


# =====================================
# THREAD STATE
# =====================================

local = threading.local()
lock = threading.RLock()
save_lock = threading.Lock()

records = []
seen_ids = set()
seen_urls = set()
seen_hashes = set()
completed_pages = set()

image_number = 0


# =====================================
# HTTP SESSION
# =====================================

def get_session():

    if not hasattr(local, "session"):

        s = requests.Session()

        s.headers.update({
            "User-Agent": "CarDatasetCollector/4.0"
        })

        retry = Retry(
            total=4,
            connect=4,
            read=3,
            status=4,
            backoff_factor=1,
            status_forcelist=[
                429, 500, 502, 503, 504
            ],
            allowed_methods=["GET"],
            respect_retry_after_header=True
        )

        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=4,
            pool_maxsize=4
        )

        s.mount("https://", adapter)
        s.mount("http://", adapter)

        local.session = s

    return local.session


# =====================================
# CSV SAVING
# =====================================

def save_csv():

    temp = CAPTIONS.with_suffix(".tmp")

    with open(temp, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "caption"]
        )

        writer.writeheader()

        for r in records:

            writer.writerow({
                "filename": r["filename"],
                "caption": r["caption"]
            })

    os.replace(temp, CAPTIONS)

    temp = ATTRIBUTIONS.with_suffix(".tmp")

    with open(temp, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(records)

    os.replace(temp, ATTRIBUTIONS)


# =====================================
# PROGRESS SAVING
# =====================================

def save_progress():

    with save_lock:

        with lock:

            data = {
                "records": list(records),
                "completed_pages": list(completed_pages),
                "downloaded": len(records),
                "target": TARGET
            }

        temp = PROGRESS.with_suffix(".tmp")

        with open(temp, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(temp, PROGRESS)

        save_csv()


# =====================================
# RESUME
# =====================================

def load_progress():

    global image_number

    if PROGRESS.exists():

        try:

            with open(PROGRESS, encoding="utf-8") as f:
                data = json.load(f)

            records.extend(data.get("records", []))

            completed_pages.update(
                data.get("completed_pages", [])
            )

        except (OSError, ValueError) as error:

            print("Progress loading error:", error)

    elif ATTRIBUTIONS.exists():

        try:

            with open(
                ATTRIBUTIONS,
                newline="",
                encoding="utf-8"
            ) as f:

                records.extend(csv.DictReader(f))

        except (OSError, csv.Error) as error:

            print("CSV loading error:", error)

    valid = []

    for r in records:

        filename = r.get("filename", "")
        path = IMAGES / filename

        if not filename or not path.is_file():
            continue

        valid.append(r)

        if r.get("image_id"):
            seen_ids.add(r["image_id"])

        if r.get("image_url"):
            seen_urls.add(r["image_url"])

        if r.get("image_hash"):
            seen_hashes.add(r["image_hash"])

    records.clear()
    records.extend(valid)

    # Recover existing files even if metadata was
    # not saved before a previous interruption.

    numbers = []

    for path in IMAGES.glob("car_*.jpg"):

        try:

            number = int(path.stem.split("_")[1])
            numbers.append(number)

            if not any(
                r["filename"] == path.name
                for r in records
            ):

                digest = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

                if digest not in seen_hashes:

                    seen_hashes.add(digest)

                    records.append({
                        "filename": path.name,
                        "caption": "Car photograph",
                        "title": "",
                        "creator": "",
                        "creator_url": "",
                        "license": "",
                        "license_version": "",
                        "license_url": "",
                        "attribution": "",
                        "source_url": "",
                        "image_url": "",
                        "image_id": "",
                        "image_hash": digest,
                        "search_query": ""
                    })

        except (ValueError, OSError, IndexError):
            continue

    image_number = max(numbers, default=0)

    print("Existing images:", len(records))


# =====================================
# SEARCH API
# =====================================

def search(query, page):

    response = get_session().get(
        API,
        params={
            "q": query,
            "license": ",".join(LICENSES),
            "mature": "false",
            "page": page,
            "page_size": PAGE_SIZE
        },
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return response.json().get("results", [])


# =====================================
# DOWNLOAD IMAGE BYTES
# =====================================

def download(url):

    with get_session().get(
        url,
        timeout=TIMEOUT,
        stream=True
    ) as response:

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "image" not in content_type:
            raise ValueError(
                f"Not an image: {content_type}"
            )

        content = bytearray()

        for chunk in response.iter_content(16384):

            if not chunk:
                continue

            content.extend(chunk)

            if len(content) > MAX_SIZE:
                raise ValueError("Image too large")

    raw = bytes(content)

    # Validate actual image contents.

    stream = io.BytesIO(raw)

    image = Image.open(stream)
    image.verify()

    stream.seek(0)

    image = Image.open(stream)
    image.load()

    if image.width < MIN_SIZE or image.height < MIN_SIZE:
        raise ValueError("Image resolution too small")

    # Convert to standard JPEG.

    image = image.convert("RGB")

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=92,
        optimize=True
    )

    return output.getvalue(), hashlib.sha256(raw).hexdigest()


# =====================================
# PROCESS IMAGE
# =====================================

def process(item, query):

    global image_number

    image_id = item.get("id")
    url = item.get("url")

    if not image_id or not url:
        return False

    with lock:

        if image_id in seen_ids or url in seen_urls:
            return False

    try:

        image_bytes, digest = download(url)

        with lock:

            if digest in seen_hashes:
                return False

            if len(records) >= TARGET:
                return False

            next_number = image_number + 1
            filename = f"car_{next_number:04d}.jpg"

            path = IMAGES / filename

            # Do not overwrite existing files.

            with open(path, "xb") as f:
                f.write(image_bytes)
                f.flush()
                os.fsync(f.fileno())

            image_number = next_number

            tags = item.get("tags") or []

            tag_names = [
                t.get("name", "")
                for t in tags[:8]
                if isinstance(t, dict) and t.get("name")
            ]

            title = item.get("title") or "Car photograph"

            record = {
                "filename": filename,
                "caption": (
                    f"Car photograph. {title}. "
                    f"Search category: {query}. "
                    f"Tags: {', '.join(tag_names)}"
                ),
                "title": title,
                "creator": item.get("creator") or "",
                "creator_url": item.get("creator_url") or "",
                "license": item.get("license") or "",
                "license_version": item.get("license_version") or "",
                "license_url": item.get("license_url") or "",
                "attribution": item.get("attribution") or "",
                "source_url": item.get("foreign_landing_url") or "",
                "image_url": url,
                "image_id": image_id,
                "image_hash": digest,
                "search_query": query
            }

            records.append(record)

            seen_ids.add(image_id)
            seen_urls.add(url)
            seen_hashes.add(digest)

        save_progress()

        print(
            f"SUCCESS {len(records)}/{TARGET}: {filename}"
        )

        return True

    except (
        requests.RequestException,
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ) as error:

        print("Download failed:", str(error)[:150])
        return False


# =====================================
# MAIN
# =====================================

def main():

    load_progress()

    print("\nStarting downloader")
    print("Target:", TARGET)
    print("Already downloaded:", len(records))

    for query in QUERIES:

        if len(records) >= TARGET:
            break

        print("\nSEARCH:", query)

        for page in range(1, MAX_PAGES + 1):

            if len(records) >= TARGET:
                break

            key = f"{query}|{page}"

            if key in completed_pages:
                continue

            try:

                results = search(query, page)

            except requests.RequestException as error:

                print("API ERROR:", error)

                # Leave page uncompleted so it can
                # be retried on the next run.
                break

            except ValueError as error:

                print("Invalid API response:", error)
                break

            if not results:
                completed_pages.add(key)
                save_progress()
                break

            print(f"Page {page}: {len(results)} results")

            jobs = []

            with lock:

                for item in results:

                    image_id = item.get("id")
                    url = item.get("url")

                    if not image_id or not url:
                        continue

                    if image_id in seen_ids or url in seen_urls:
                        continue

                    jobs.append(item)

            with ThreadPoolExecutor(
                max_workers=WORKERS
            ) as executor:

                futures = [
                    executor.submit(process, item, query)
                    for item in jobs
                ]

                for future in as_completed(futures):

                    try:
                        future.result()

                    except Exception as error:
                        print("Worker error:", error)

            completed_pages.add(key)

            save_progress()

            print(
                f"TOTAL: {len(records)}/{TARGET}"
            )

            time.sleep(0.3)

    save_progress()

    print("\n================================")
    print("DOWNLOAD FINISHED")
    print("================================")

    print("Images downloaded:", len(records))
    print("Target:", TARGET)
    print("Image folder:", IMAGES.resolve())
    print("Captions:", CAPTIONS.resolve())
    print("Attributions:", ATTRIBUTIONS.resolve())

    if len(records) < TARGET:
        print("Run again to continue.")
    else:
        print("TARGET REACHED!")


# =====================================
# EXECUTION
# =====================================

try:

    main()

except KeyboardInterrupt:

    print("\nInterrupted. Saving progress...")
    save_progress()

    print("Progress saved.")

finally:

    if hasattr(local, "session"):
        local.session.close()
