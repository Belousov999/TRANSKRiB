import os
import time
import logging
import subprocess
import threading
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime

import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============================================================
# ПУТИ
# ============================================================
INPUT_DIR   = r"D:\TRANSKRiB\INPUT"
OUTPUT_DIR  = r"D:\TRANSKRiB\TTGP\INPUTG"
LOG_DIR     = r"D:\TRANSKRiB\CONVERTOR\ConvLog"
FFMPEG_PATH = r"D:\TRANSKRiB\CONVERTOR\ffmpeg\bin\ffmpeg.exe"

# ============================================================
# НАСТРОЙКИ
# ============================================================
AFFINITY = [2, 3, 4, 5]      # ограничение CPU для процесса python
FFMPEG_THREADS = "2"         # сколько потоков даём ffmpeg (если хочешь строго 1 — поставь "1")

AUDIO_EXTS = {".mp3", ".flac", ".ogg", ".aac", ".wav", ".wma", ".m4a", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"}

POLL_READY_DELAY = 0.5       # задержка между проверками "файл готов"
READY_STABLE_CHECKS = 3      # сколько раз подряд размер должен совпасть

# ============================================================
# ЛОГИ
# ============================================================
def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, datetime.now().strftime("convert_%Y%m%d.log"))

    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(sh)

    logging.info("=== CONVERT WATCHDOG START ===")
    logging.info(f"INPUT  : {INPUT_DIR}")
    logging.info(f"OUTPUT : {OUTPUT_DIR}")
    logging.info(f"LOG    : {LOG_DIR}")
    logging.info(f"FFMPEG : {FFMPEG_PATH}")
    logging.info(f"AFFINITY: {AFFINITY} | ffmpeg threads={FFMPEG_THREADS}")

def set_affinity() -> None:
    try:
        p = psutil.Process()
        p.cpu_affinity(AFFINITY)
        logging.info(f"Affinity set to: {p.cpu_affinity()}")
    except Exception as e:
        logging.warning(f"Cannot set affinity {AFFINITY}: {e}")

# ============================================================
# ВСПОМОГАТЕЛЬНОЕ
# ============================================================
def is_target_file(p: Path) -> bool:
    if not p.is_file():
        return False
    ext = p.suffix.lower()
    return ext in AUDIO_EXTS or ext in VIDEO_EXTS

def unique_out_path(out_dir: Path, stem: str) -> Path:
    out = out_dir / f"{stem}.wav"
    if not out.exists():
        return out
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{stem}_{stamp}.wav"

def wait_file_ready(p: Path) -> bool:
    """
    Ждём пока файл перестанет расти и станет доступен для чтения.
    Это нужно, чтобы не схватить файл, который ещё копируется.
    """
    last = -1
    stable = 0

    for _ in range(60):  # максимум ~30 сек при delay=0.5
        if not p.exists():
            return False

        try:
            size = p.stat().st_size
        except Exception:
            time.sleep(POLL_READY_DELAY)
            continue

        if size == last and size > 0:
            stable += 1
        else:
            stable = 0
            last = size

        # пробуем открыть (если файл залочен — будет исключение)
        if stable >= READY_STABLE_CHECKS:
            try:
                with open(p, "rb") as f:
                    f.read(1)
                return True
            except Exception:
                pass

        time.sleep(POLL_READY_DELAY)

    return False

# ============================================================
# КОНВЕРТАЦИЯ
# ============================================================
def convert_one(src: Path) -> None:
    if not src.exists() or not is_target_file(src):
        return

    if not wait_file_ready(src):
        logging.info(f"SKIP (not ready/vanished): {src.name}")
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = unique_out_path(out_dir, src.stem)

    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-threads", FFMPEG_THREADS,
        "-y",
        str(dst),
    ]

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elapsed = time.perf_counter() - t0

        # удаляем оригинал только при успехе
        try:
            src.unlink()
        except Exception as e:
            logging.warning(f"Converted ok, but cannot delete original: {src} | {e}")

        logging.info(f"OK: {src.name} -> {dst.name} | {elapsed:.2f}s")

    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - t0
        try:
            err = e.stderr.decode("utf-8", errors="ignore").strip()
        except Exception:
            err = str(e)
        logging.error(f"FAIL: {src.name} | {elapsed:.2f}s | {err}")

# ============================================================
# WATCHDOG HANDLER + ОЧЕРЕДЬ (ОДИН WORKER)
# ============================================================
class EnqueueHandler(FileSystemEventHandler):
    def __init__(self, q: Queue, pending: set, lock: threading.Lock):
        super().__init__()
        self.q = q
        self.pending = pending
        self.lock = lock

    def _enqueue(self, path: str):
        p = Path(path)
        if not is_target_file(p):
            return
        # дедуп: не кладём в очередь одно и то же несколько раз
        with self.lock:
            if str(p) in self.pending:
                return
            self.pending.add(str(p))
        self.q.put(p)

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            # важнее destination
            self._enqueue(event.dest_path)

def worker_loop(q: Queue, pending: set, lock: threading.Lock, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            p: Path = q.get(timeout=0.5)
        except Empty:
            continue

        try:
            convert_one(p)
        finally:
            with lock:
                pending.discard(str(p))
            q.task_done()

# ============================================================
# MAIN
# ============================================================
def main():
    setup_logging()
    set_affinity()

    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        logging.error(f"INPUT folder not found: {INPUT_DIR}")
        return

    if not Path(FFMPEG_PATH).exists():
        logging.error(f"ffmpeg.exe not found: {FFMPEG_PATH}")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    q = Queue()
    pending = set()
    lock = threading.Lock()
    stop_event = threading.Event()

    # сначала — обработаем уже лежащие файлы
    existing = [p for p in in_dir.iterdir() if is_target_file(p)]
    existing.sort(key=lambda x: x.stat().st_mtime)
    for p in existing:
        with lock:
            pending.add(str(p))
        q.put(p)
    if existing:
        logging.info(f"Seeded existing files: {len(existing)}")

    # worker (один — строго однопоточно)
    worker = threading.Thread(target=worker_loop, name="worker-1",
                              args=(q, pending, lock, stop_event), daemon=True)
    worker.start()

    # watchdog
    handler = EnqueueHandler(q, pending, lock)
    observer = Observer()
    observer.schedule(handler, INPUT_DIR, recursive=False)
    observer.start()

    logging.info("Watching for new files... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("=== STOP (Ctrl+C) ===")
    finally:
        stop_event.set()
        observer.stop()
        observer.join(timeout=5)

if __name__ == "__main__":
    main()
