import os
import sys
import time
import shutil
import queue
import threading
import logging
import re
import argparse
from datetime import datetime

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ============================================================
# ПУТИ (НОВЫЕ)
# ============================================================
ROOT = r"D:\TRANSKRiB"

INPUT_FOLDER       = rf"{ROOT}\TTGP\INPUTG"
RESULT_FOLDER      = rf"{ROOT}\!RESULT"
WAV_ARCHIVE_FOLDER = rf"{ROOT}\WavARH"

LOG_FOLDER         = rf"{ROOT}\TTGP\TTGPlog"
ARCHIVE_LOG_FOLDER = rf"{ROOT}\TTGP\LogARH"

REJECT_FOLDER      = rf"{ROOT}\TTGP\REJECTG"

MODELS = {
    "largev3":     r"D:\TRANSKRiB\LargeV3",
    "turbo":       r"D:\TRANSKRiB\ARH\turbo",
    "turboenru32": r"D:\TRANSKRiB\ARH\TurboEnRu3.2",
}
DEFAULT_MODEL_KEY = "turbo"

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
MIN_DURATION_SEC = 2.0
SEGMENT_DURATION_SEC = 30
WRAP_COL = 110
MARK_EVERY_SEC = 120

FILE_STABLE_TIMEOUT_SEC = 30 * 60
FILE_STABLE_POLL_SEC = 2.0
FILE_STABLE_REQUIRED_SAME = 3

SAMPLE_RATE = 16000

# Whisper log-mel
N_SAMPLES = 30 * SAMPLE_RATE
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
FMIN = 0.0
FMAX = 8000.0
N_FRAMES = 3000

WHISPER_LANGUAGE = "russian"
WHISPER_TASK = "transcribe"

# ============================================================
# ГЛОБАЛЬНЫЕ
# ============================================================
stop_flag = threading.Event()
work_q: "queue.Queue[str]" = queue.Queue()
in_queue_or_processing = set()
set_lock = threading.Lock()

model = None
tokenizer = None
NUM_MELS = 80

_mel_filters_cache = {}
_window_cache = {}

_prefix_re = re.compile(r"^(?P<prefix>\d{8}_\d{4})_(?P<rest>.+)$")

# ============================================================
# ЛОГИ
# ============================================================
def setup_logging():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_file = os.path.join(LOG_FOLDER, datetime.now().strftime("TTGP_GPU_%Y%m%d.txt"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logging.info("=== TTGP GPU worker started ===")
    logging.info(f"INPUT_FOLDER  : {INPUT_FOLDER}")
    logging.info(f"RESULT_FOLDER : {RESULT_FOLDER}")
    logging.info(f"WAV_ARCHIVE   : {WAV_ARCHIVE_FOLDER}")
    logging.info(f"REJECT_FOLDER : {REJECT_FOLDER}")
    logging.info(f"LOG_FOLDER    : {LOG_FOLDER}")
    logging.info(f"LOG_ARCHIVE   : {ARCHIVE_LOG_FOLDER}")

def archive_old_logs():
    os.makedirs(ARCHIVE_LOG_FOLDER, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    if not os.path.isdir(LOG_FOLDER):
        return

    for lf in os.listdir(LOG_FOLDER):
        if lf.endswith(".txt") and today_str not in lf:
            src = os.path.join(LOG_FOLDER, lf)
            dst = os.path.join(ARCHIVE_LOG_FOLDER, lf)
            try:
                shutil.move(src, dst)
            except Exception as e:
                logging.error(f"Не смог архивировать лог {src}: {e}")

def ensure_folders():
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(WAV_ARCHIVE_FOLDER, exist_ok=True)
    os.makedirs(REJECT_FOLDER, exist_ok=True)
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(ARCHIVE_LOG_FOLDER, exist_ok=True)

# ============================================================
# УТИЛИТЫ
# ============================================================
def is_wav(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".wav"

def format_mm_ss_xx(seconds: float) -> str:
    if seconds is None:
        return "00:00.00"
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"

def minutes_word(n: int) -> str:
    n = abs(int(n))
    n10 = n % 10
    n100 = n % 100
    if 11 <= n100 <= 14:
        return "минут"
    if n10 == 1:
        return "минута"
    if 2 <= n10 <= 4:
        return "минуты"
    return "минут"

def parse_prefix_dt_from_basename(base_name: str):
    m = _prefix_re.match(base_name)
    if not m:
        return None, None
    prefix = m.group("prefix")
    try:
        dt = datetime.strptime(prefix, "%Y%m%d_%H%M")
        return dt, prefix
    except Exception:
        return None, prefix

def mtime_dt(path: str) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path))

def derive_prefix_from_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M")

def wait_file_stable(path: str) -> bool:
    start = time.time()
    last_size = -1
    same_count = 0

    while time.time() - start < FILE_STABLE_TIMEOUT_SEC:
        if stop_flag.is_set():
            return False
        if not os.path.exists(path):
            return False

        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1

        if size > 0 and size == last_size:
            same_count += 1
        else:
            same_count = 0
            last_size = size

        if same_count >= FILE_STABLE_REQUIRED_SAME:
            return True

        time.sleep(FILE_STABLE_POLL_SEC)

    logging.warning(f"Файл не стабилизировался по размеру за таймаут: {path}")
    return False

def _safe_move(dst_dir: str, src_path: str) -> str:
    base = os.path.basename(src_path)
    dst = os.path.join(dst_dir, base)

    if os.path.exists(dst):
        name, ext = os.path.splitext(base)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(dst_dir, f"{name}_{stamp}{ext}")

    shutil.move(src_path, dst)
    return dst

def safe_move_to_archive(src_path: str) -> str:
    return _safe_move(WAV_ARCHIVE_FOLDER, src_path)

def safe_move_to_reject(src_path: str) -> str:
    return _safe_move(REJECT_FOLDER, src_path)

def wrap_text(text: str, width: int = WRAP_COL) -> str:
    import textwrap
    text = " ".join(text.split())
    if not text:
        return ""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False
    )

def wav_info(path: str):
    info = sf.info(path)
    duration = info.frames / float(info.samplerate) if info.samplerate else 0.0
    return info.samplerate, info.channels, duration

def ensure_prefix_or_rename_by_mtime(path: str):
    old_name = os.path.basename(path)
    base = os.path.splitext(old_name)[0]

    rec_dt, prefix = parse_prefix_dt_from_basename(base)
    if rec_dt is not None:
        return path, rec_dt, prefix, "name", old_name

    rec_dt = mtime_dt(path)
    prefix = derive_prefix_from_dt(rec_dt)

    dir_ = os.path.dirname(path)
    ext = os.path.splitext(old_name)[1]
    new_name = f"{prefix}_{base}{ext}"
    new_path = os.path.join(dir_, new_name)

    if os.path.exists(new_path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{prefix}_{base}_{stamp}{ext}"
        new_path = os.path.join(dir_, new_name)

    try:
        os.rename(path, new_path)
        logging.info(f"RENAME(mtime): {old_name} -> {os.path.basename(new_path)} | mtime={rec_dt.strftime('%Y-%m-%d %H:%M')}")
        return new_path, rec_dt, prefix, "mtime", old_name
    except Exception as e:
        logging.error(f"Не смог переименовать по mtime: {path} | {e}")
        return path, rec_dt, prefix, "mtime", old_name

# ============================================================
# WHISPER LOG-MEL
# ============================================================
def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _get_window(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key in _window_cache:
        return _window_cache[key]
    w = torch.hann_window(WIN_LENGTH, device=device, dtype=torch.float32)
    _window_cache[key] = w
    return w

def _build_mel_filters(device: torch.device, n_mels: int) -> torch.Tensor:
    key = (str(device), int(n_mels))
    if key in _mel_filters_cache:
        return _mel_filters_cache[key]

    n_freqs = N_FFT // 2 + 1
    m_min = _hz_to_mel(torch.tensor(FMIN, device=device))
    m_max = _hz_to_mel(torch.tensor(FMAX, device=device))

    m_pts = torch.linspace(m_min, m_max, n_mels + 2, device=device)
    hz_pts = _mel_to_hz(m_pts)

    bins = torch.floor((N_FFT + 1) * hz_pts / SAMPLE_RATE).to(torch.int64)
    bins = torch.clamp(bins, 0, n_freqs - 1)

    fb = torch.zeros((n_mels, n_freqs), device=device, dtype=torch.float32)

    for m in range(n_mels):
        left = int(bins[m].item())
        center = int(bins[m + 1].item())
        right = int(bins[m + 2].item())

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        if right > n_freqs:
            right = n_freqs

        if center > left:
            fb[m, left:center] = (torch.arange(left, center, device=device, dtype=torch.float32) - left) / (center - left)
        if right > center:
            fb[m, center:right] = (right - torch.arange(center, right, device=device, dtype=torch.float32)) / (right - center)

    _mel_filters_cache[key] = fb
    return fb

def whisper_log_mel_30s(audio_1d: torch.Tensor, n_mels: int) -> torch.Tensor:
    device = audio_1d.device

    if audio_1d.numel() < N_SAMPLES:
        pad = N_SAMPLES - audio_1d.numel()
        audio_1d = torch.nn.functional.pad(audio_1d, (0, pad))
    else:
        audio_1d = audio_1d[:N_SAMPLES]

    window = _get_window(device)
    mel_filters = _build_mel_filters(device, n_mels)

    stft = torch.stft(
        audio_1d,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True
    )

    magnitudes = stft.abs().pow(2.0)
    mel_spec = mel_filters @ magnitudes

    if mel_spec.shape[1] >= N_FRAMES:
        mel_spec = mel_spec[:, :N_FRAMES]
    else:
        mel_spec = torch.nn.functional.pad(mel_spec, (0, N_FRAMES - mel_spec.shape[1]))

    log_mel = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    return log_mel.unsqueeze(0)

def build_forced_decoder_ids(tok):
    if WHISPER_LANGUAGE is None:
        return None
    if hasattr(tok, "get_decoder_prompt_ids"):
        try:
            return tok.get_decoder_prompt_ids(language=WHISPER_LANGUAGE, task=WHISPER_TASK)
        except Exception:
            return None
    return None

# ============================================================
# МОДЕЛЬ
# ============================================================
def load_model(model_key: str):
    if model_key not in MODELS:
        raise RuntimeError(f"Неизвестный ключ модели: {model_key}. Доступно: {list(MODELS.keys())}")

    model_path = MODELS[model_key]

    if not torch.cuda.is_available():
        raise RuntimeError("GPU недоступен. Проверь CUDA/драйвер.")

    logging.info(f"MODEL_KEY={model_key}")
    logging.info(f"MODEL_PATH={model_path}")
    logging.info(f"torch={torch.__version__} | cuda={torch.version.cuda} | cuda_available={torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    mdl = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to("cuda")
    mdl.eval()

    global NUM_MELS
    NUM_MELS = int(getattr(mdl.config, "num_mel_bins", 80))
    logging.info(f"model.config.num_mel_bins={getattr(mdl.config, 'num_mel_bins', None)} | NUM_MELS(used)={NUM_MELS}")

    try:
        mdl.config.pad_token_id = tok.eos_token_id
    except Exception:
        pass

    return mdl, tok

# ============================================================
# ОБРАБОТКА WAV
# ============================================================
def transcribe_wav(path: str):
    if not is_wav(path):
        return
    if not os.path.exists(path):
        logging.info(f"Файл исчез до обработки: {path}")
        return
    if not wait_file_stable(path):
        logging.warning(f"Файл не готов/не стабилен: {path}")
        return

    path, rec_dt, rec_prefix, rec_source, old_name = ensure_prefix_or_rename_by_mtime(path)
    if not os.path.exists(path):
        logging.info(f"Файл исчез после переименования: {path}")
        return

    t0 = time.perf_counter()

    try:
        sr, ch, dur = wav_info(path)
    except Exception as e:
        logging.error(f"Не смог прочитать WAV info: {path} | {e}")
        try:
            moved = safe_move_to_reject(path)
            logging.info(f"REJECT -> {moved}")
        except Exception:
            pass
        return

    logging.info(
        f"Старт: {os.path.basename(path)} | rec={rec_dt.strftime('%Y-%m-%d %H:%M')}({rec_source}) | "
        f"dur={format_mm_ss_xx(dur)} | sr={sr} ch={ch} | num_mels={NUM_MELS}"
    )

    if dur < MIN_DURATION_SEC:
        logging.info(f"Пропуск: меньше {MIN_DURATION_SEC:.1f} сек ({dur:.2f}s): {path}")
        try:
            moved = safe_move_to_reject(path)
            logging.info(f"REJECT(short) -> {moved}")
        except Exception:
            pass
        return

    if sr != SAMPLE_RATE or ch != 1:
        logging.error(f"ОШИБКА ФОРМАТА: ожидаю WAV 16kHz mono, получено sr={sr}, ch={ch}.")
        try:
            moved = safe_move_to_reject(path)
            logging.info(f"REJECT(format) -> {moved}")
        except Exception:
            pass
        return

    base_name = os.path.splitext(os.path.basename(path))[0]
    out_txt = os.path.join(RESULT_FOLDER, f"{base_name}.txt")

    forced_ids = build_forced_decoder_ids(tokenizer)

    try:
        with sf.SoundFile(path, mode="r") as f, open(out_txt, "w", encoding="utf-8") as out:
            out.write(f"{rec_dt.strftime('%Y-%m-%d %H:%M')}\n")
            out.write(f"Файл: {os.path.basename(path)}\n")
            if rec_source == "mtime":
                out.write(f"Исходное имя: {old_name}\n")
            out.write(f"Модель: num_mel_bins={NUM_MELS}\n\n")

            seg_frames = SEGMENT_DURATION_SEC * SAMPLE_RATE
            seg_idx = 0

            while True:
                if stop_flag.is_set():
                    logging.info("Остановка по stop_flag.")
                    return

                audio_np = f.read(seg_frames, dtype="float32", always_2d=False)
                if audio_np is None or len(audio_np) == 0:
                    break

                seg_start_sec = seg_idx * SEGMENT_DURATION_SEC

                if seg_start_sec > 0 and (seg_start_sec % MARK_EVERY_SEC == 0):
                    mins = seg_start_sec // 60
                    marker = f"{mins} {minutes_word(mins)}"
                    out.write("\n" + marker + "\n\n")
                    logging.info(f"MARK: {marker} (t={format_mm_ss_xx(seg_start_sec)})")

                audio = torch.from_numpy(audio_np).to("cuda", non_blocking=True)
                input_features = whisper_log_mel_30s(audio, NUM_MELS).to(dtype=torch.float16)

                attention_mask = torch.ones(
                    (input_features.size(0), input_features.size(-1)),
                    dtype=torch.long,
                    device="cuda"
                )

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        forced_decoder_ids=forced_ids,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=256
                    )

                text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if text:
                    out.write(wrap_text(text, WRAP_COL))
                    out.write("\n\n")

                logging.info(f"SEG#{seg_idx+1} t={format_mm_ss_xx(seg_start_sec)} chars={len(text)}")
                seg_idx += 1

        archived = safe_move_to_archive(path)
        elapsed = time.perf_counter() - t0
        logging.info(f"ГОТОВО: {os.path.basename(archived)} | dur={format_mm_ss_xx(dur)} | proc={elapsed:.2f}s | archived={archived}")
        logging.info(f"TXT: {out_txt}")

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logging.exception(f"Ошибка транскрибации: {path} | время={elapsed:.2f}s | {e}")
        try:
            if os.path.exists(out_txt):
                os.remove(out_txt)
        except Exception:
            pass
        try:
            if os.path.exists(path):
                moved = safe_move_to_reject(path)
                logging.info(f"REJECT(after error) -> {moved}")
        except Exception:
            pass

# ============================================================
# ОЧЕРЕДЬ / WORKER
# ============================================================
def enqueue_file(path: str):
    if not is_wav(path):
        return
    with set_lock:
        if path in in_queue_or_processing:
            return
        in_queue_or_processing.add(path)
    work_q.put(path)

def worker_loop():
    while not stop_flag.is_set():
        try:
            path = work_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            transcribe_wav(path)
        finally:
            with set_lock:
                in_queue_or_processing.discard(path)
            work_q.task_done()

# ============================================================
# WATCHDOG
# ============================================================
class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if is_wav(event.src_path):
            logging.info(f"Новый WAV: {event.src_path}")
            enqueue_file(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if is_wav(event.dest_path):
            logging.info(f"WAV moved/renamed -> {event.dest_path}")
            enqueue_file(event.dest_path)

def process_existing_files():
    if not os.path.isdir(INPUT_FOLDER):
        return

    files = []
    for name in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, name)
        if os.path.isfile(path) and is_wav(path):
            try:
                files.append((os.path.getmtime(path), path))
            except Exception:
                files.append((0, path))

    files.sort(key=lambda x: (x[0], x[1]))
    for _, path in files:
        enqueue_file(path)

# ============================================================
# WATCHDOG
# ============================================================
class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if is_wav(event.src_path):
            logging.info(f"Новый WAV: {event.src_path}")
            enqueue_file(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if not is_wav(event.dest_path):
            return

        inp = os.path.abspath(INPUT_FOLDER).lower()
        src = os.path.abspath(event.src_path).lower()
        dst = os.path.abspath(event.dest_path).lower()

        def is_inside(path: str, folder: str) -> bool:
            try:
                return os.path.commonpath([path, folder]) == folder
            except Exception:
                # fallback, если commonpath не сработал
                return path.startswith(folder)

        src_in = is_inside(src, inp)
        dst_in = is_inside(dst, inp)

        # Берём только "въезд" WAV в INPUTG извне.
        # Внутренние rename (src_in && dst_in) и выезд в архив (src_in && !dst_in) игнорируем.
        if dst_in and not src_in:
            logging.info(f"WAV moved into INPUTG -> {event.dest_path}")
            enqueue_file(event.dest_path)


def process_existing_files():
    """Подхватываем WAV, которые уже лежат в INPUTG на момент старта."""
    if not os.path.isdir(INPUT_FOLDER):
        return

    files = []
    for name in os.listdir(INPUT_FOLDER):
        path = os.path.join(INPUT_FOLDER, name)
        if os.path.isfile(path) and is_wav(path):
            try:
                files.append((os.path.getmtime(path), path))
            except Exception:
                files.append((0, path))

    # кто раньше появился/изменён — тот раньше в очередь
    files.sort(key=lambda x: (x[0], x[1]))

    for _, path in files:
        enqueue_file(path)


# ============================================================
# MAIN
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        choices=list(MODELS.keys()),
        help="Ключ модели: turbo | turboenru32 | largev3"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    setup_logging()
    archive_old_logs()
    ensure_folders()

    global model, tokenizer
    model, tokenizer = load_model(args.model)

    # один воркер + очередь
    threading.Thread(target=worker_loop, daemon=True).start()

    # подхват уже лежащих файлов
    process_existing_files()

    # watchdog
    handler = FileHandler()
    observer = Observer()
    observer.schedule(handler, INPUT_FOLDER, recursive=False)
    observer.start()

    try:
        while not stop_flag.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_flag.set()
    finally:
        observer.stop()
        observer.join()
        logging.info("TTGP GPU воркер остановлен.")


if __name__ == "__main__":
    main()
