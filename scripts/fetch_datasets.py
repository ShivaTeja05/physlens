"""Download the public footage and sensor data the benchmarks use, into datasets/ (not in git).

  python3 scripts/fetch_datasets.py --list               every set, its files, sources and sizes
  python3 scripts/fetch_datasets.py --yes 8555 10072     download those sets (skips files already there)
  python3 scripts/fetch_datasets.py --yes all

Sources and terms: NHTSA Vehicle Crash Test Database (public information; credit NHTSA; NHTSA asks that its videos
not be edited; don't re-host them), NASA (public domain), Wikimedia Commons (CC BY; credit the author, see
datasets/commons/SOURCE.md). Standard library only.
"""
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
NV = "https://nrd-static.nhtsa.dot.gov/videos/vehdb"
NU = "https://nrd-static.nhtsa.dot.gov/compress/uds/vehdb"
NT = "https://nrd-static.nhtsa.dot.gov/tsv/vehdb"
API = "https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/metadata"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath"

SETS = {
    "9070": ("NHTSA 9070: Lexus NX into a rigid barrier, 56.55 km/h", [
        (f"{NV}/v00000/v09000/transcoded/v09070C012.mp4", "nhtsa/9070/cam12.mp4"),      # Right Overall, 1,000 fps
        (f"{NV}/v00000/v09000/transcoded/v09070C015.mp4", "nhtsa/9070/cam15.mp4"),      # Real-Time Left Overall
        (f"{NV}/v00000/v09000/transcoded/v09070C016.mp4", "nhtsa/9070/cam16.mp4"),
        (f"{NT}/v00000/v09000/v09070/v09070tsv.093", "nhtsa/9070/sensors/floorpan_093.tsv"),
        (f"{NT}/v00000/v09000/v09070/v09070tsv.094", "nhtsa/9070/sensors/floorpan_094.tsv"),
        (f"{API}/9070", "nhtsa/9070/metadata.json")]),
    "9001": ("NHTSA 9001: barrier cart into a parked Camry, 62.19 km/h", [
        (f"{NV}/v00000/v09000/transcoded/v09001C002.mp4", "nhtsa/9001/cam02.mp4"),
        (f"{NV}/v00000/v09000/transcoded/v09001C010.mp4", "nhtsa/9001/cam10.mp4"),
        (f"{API}/9001", "nhtsa/9001/metadata.json")]),
    "15756": ("NHTSA 15756: Nissan Kicks sideways into a pole, 31.09 km/h", [
        ("https://nrd-static.nhtsa.dot.gov/vehdb/15756/videos/Camera%20No.%2010%20-%20%20Overhead%20Wide-View%20of%20Impact.avi",
         "nhtsa/15756/cam10_overhead_wide.avi"),
        ("https://nrd-static.nhtsa.dot.gov/vehdb/15756/videos/Camera%20No.%201%20-%20Real%20Time%20Film%20Coverage.mp4",
         "nhtsa/15756/cam01_realtime.mp4")]),
    "8555": ("NHTSA 8555: Ford F-150 into the rear of a standing Jeep Liberty, 69.1 km/h", [
        (f"{NV}/v00000/v08500/transcoded/v08555C008.mp4", "nhtsa/8555/cam08_overhead.mp4"),
        (f"{NU}/v00000/v08500/v08555UDS.zip", "nhtsa/8555/v08555UDS.zip")]),
    "7371": ("NHTSA 7371: Yaris into a standing Yaris, 113 km/h, 15°", [
        (f"{NV}/v00000/v07300/transcoded/v07371C009.mp4", "nhtsa/7371/cam09_overhead_wide.mp4"),
        (f"{NU}/v00000/v07300/v07371UDS.zip", "nhtsa/7371/v07371UDS.zip")]),
    "10072": ("NHTSA 10072: moving barrier into a standing Nissan Altima, 90.33 km/h, 15°", [
        (f"{NV}/v10000/v10000/transcoded/v10072C006.mp4", "nhtsa/10072/cam06_overhead_vehicle.mp4"),
        (f"{NV}/v10000/v10000/transcoded/v10072C007.mp4", "nhtsa/10072/cam07_overhead_omdb.mp4"),
        (f"{NU}/v10000/v10000/v10072UDS.zip", "nhtsa/10072/v10072UDS.zip")]),
    "truck": ("Wikimedia Commons (CC BY 3.0, yovo68): box truck at the 11foot8 bridge", [
        (f"{COMMONS}/Boxtruck_carnage_on_Friday_the_13th_at_the_11foot8_bridge.webm", "commons/truck-11foot8/clip.webm")]),
    "rally": ("Wikimedia Commons (CC BY 2.0, Iain Cameron): rally car past a handheld camera", [
        (f"{COMMONS}/Jari-Matti_Latvala_in_Toyota_Yaris_10_at_Route_de_la_Garde_on_Thursday.webm", "commons/rally-latvala/clip.webm")]),
    "apollo15": ("NASA (public domain): Apollo 15 hammer and feather drop", [
        ("https://nssdc.gsfc.nasa.gov/planetary/image/AS15_Ham_feath_drop3.mp4", "nasa/apollo15/hammer_feather.mp4")]),
}
UA = {"User-Agent": "PhysLens-dataset-fetch/1.0 (research; github.com/ShivaTeja05/physlens)"}


def size(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=30) as r:
            n = r.headers.get("Content-Length")
            return f"{int(n) / 1e6:.1f} MB" if n else "?"
    except Exception as e:                                               # noqa: BLE001
        return f"unavailable ({e.__class__.__name__})"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print("  have", dest.relative_to(ROOT))
        return
    print("  get ", dest.relative_to(ROOT), "...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.rename(dest)
    if dest.suffix == ".zip":                                            # accelerometer curves -> <test>/uds/
        with zipfile.ZipFile(dest) as z:
            z.extractall(dest.parent / "uds")
    if dest.suffix == ".webm":                                           # the scripts read an H.264 copy
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            exe = shutil.which("ffmpeg")
        if exe:
            subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(dest), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-an", str(dest.with_suffix(".mp4"))], check=True)
        else:
            print("  (no ffmpeg found: convert", dest.name, "to clip.mp4 yourself)")


def main(argv):
    if "--list" in argv or not argv:
        for key, (title, files) in SETS.items():
            print(f"{key}: {title}")
            for url, rel in files:
                print(f"    {rel:48s} {size(url):>12s}   {url}")
        print("\nDownload with: python3 scripts/fetch_datasets.py --yes <set> [<set> ...] | all")
        return
    if "--yes" not in argv:
        sys.exit("add --yes to download (see --list for what and how big)")
    keys = [a for a in argv if not a.startswith("--")]
    keys = list(SETS) if keys == ["all"] else keys
    for key in keys:
        if key not in SETS:
            sys.exit(f"unknown set {key}; known: {', '.join(SETS)}")
        print(f"{key}: {SETS[key][0]}")
        for url, rel in SETS[key][1]:
            fetch(url, DATA / rel)


if __name__ == "__main__":
    main(sys.argv[1:])
