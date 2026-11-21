"""從手機影片抽幀，供預標註與 CVAT 標註使用。

每隔 --interval 秒取一張；在該時間窗內挑「最清晰」的幀（Laplacian variance 最大，
減少走路晃動造成的模糊），再和上一張已存的圖做縮圖差異比對，太像就跳過。

輸出：
  <out>/<prefix><影片名>_t<秒數>s.jpg     例：own_crossA_t0012.5s.jpg
  <out>/manifest.csv                        每張圖的來源影片、幀號、秒數、清晰度、差異值
  （manifest 會累加；同一影片重跑會先移除該影片舊的列與圖檔，避免重複）

用法：
  python tools/extract_frames.py videos/street4.mp4 --out frames/own
  python tools/extract_frames.py videos/ --out frames/own --interval 1.5
  python tools/extract_frames.py videos/a.mp4 --max-seconds 20 --out scratch   # 小量測試

注意：
  * 時間取自影片時間戳（CAP_PROP_POS_MSEC），可處理手機常見的可變幀率影片。
  * 依「影片／地點」切 train/val/test，不要依單張隨機切（見 HANDOFF 第 3 節）；
    manifest.csv 的 source 欄可直接拿來分組。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm', '.3gp'}
MANIFEST_FIELDS = ['file', 'source', 'frame_index', 'time_s', 'width', 'height',
                   'sharpness', 'diff_prev', 'candidates_in_window']


def safe_stem(path: Path) -> str:
    """影片名轉成安全檔名（保留中英數，其他字元換成 _）。"""
    stem = re.sub(r'[^\w\-]+', '_', path.stem, flags=re.UNICODE).strip('_')
    return stem or 'video'


def thumb(frame: np.ndarray, size=(64, 36)) -> np.ndarray:
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)


def sharpness(frame: np.ndarray, width=640) -> float:
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """0~255；用平均亮度正規化，避免自動曝光變化被當成新畫面。"""
    a = a - a.mean()
    b = b - b.mean()
    return float(np.abs(a - b).mean())


def write_jpg(path: Path, frame: np.ndarray, quality: int) -> None:
    # cv2.imwrite 在 Windows 遇到中文路徑會失敗，改用 imencode + write_bytes。
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f'JPEG encode failed: {path}')
    path.write_bytes(buf.tobytes())


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_manifest(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def extract(video: Path, out: Path, prefix: str, interval: float, diff_thresh: float,
            samples_per_window: int, min_sharpness: float, quality: int,
            max_seconds: float | None, dry_run: bool = False) -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'無法開啟影片：{video}（中文路徑失敗時請先改成英文檔名）')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # 每個時間窗內評估幾幀（其餘只 grab 不解碼成影像，較快）。
    eval_every = max(1, round(fps * interval / max(1, samples_per_window)))
    stem = safe_stem(video)
    rows: list[dict] = []
    stats = dict(source=video.name, fps=fps, frames=0, windows=0, saved=0,
                 skipped_similar=0, skipped_blurry=0, duration_s=0.0)

    last_saved_thumb = None
    window_start = 0.0
    best = None  # (sharpness, frame, index, time_s, n_candidates)
    n_candidates = 0
    index = -1

    def flush_window():
        nonlocal last_saved_thumb, best, n_candidates
        if best is None:
            return
        stats['windows'] += 1
        sharp, frame, idx, t, ncand = best
        best, n_candidates = None, 0
        if sharp < min_sharpness:
            stats['skipped_blurry'] += 1
            return
        th = thumb(frame)
        diff = mean_abs_diff(th, last_saved_thumb) if last_saved_thumb is not None else float('inf')
        if diff < diff_thresh:
            stats['skipped_similar'] += 1
            return
        name = f'{prefix}{stem}_t{t:06.1f}s.jpg'
        if not dry_run:
            write_jpg(out / name, frame, quality)
        last_saved_thumb = th
        stats['saved'] += 1
        h, w = frame.shape[:2]
        rows.append(dict(file=name, source=video.name, frame_index=idx, time_s=f'{t:.3f}',
                         width=w, height=h, sharpness=f'{sharp:.1f}',
                         diff_prev='' if diff == float('inf') else f'{diff:.2f}',
                         candidates_in_window=ncand))

    try:
        while True:
            if not cap.grab():
                break
            index += 1
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos_ms / 1000 if pos_ms and pos_ms > 0 else index / fps
            if max_seconds is not None and t > max_seconds:
                break
            stats['frames'] += 1
            stats['duration_s'] = t
            while t >= window_start + interval:
                flush_window()
                window_start += interval
            if index % eval_every:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            n_candidates += 1
            s = sharpness(frame)
            if best is None or s > best[0]:
                best = (s, frame, index, t, n_candidates)
            else:
                best = best[:4] + (n_candidates,)
        flush_window()
    finally:
        cap.release()
    if total and stats['frames'] < total * 0.98 and max_seconds is None:
        print(f'  [警告] {video.name}: 只讀到 {stats["frames"]}/{total} 幀，影片可能損毀', file=sys.stderr)
    return rows, stats


def collect(inputs: list[str]) -> list[Path]:
    videos = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            videos += sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXT)
        elif p.is_file() and p.suffix.lower() in VIDEO_EXT:
            videos.append(p)
        else:
            raise SystemExit(f'找不到影片或不支援的格式：{item}')
    if not videos:
        raise SystemExit('沒有找到影片')
    return videos


def main(argv=None):
    if not sys.stdout.isatty():  # 輸出導向檔案／管線時固定 UTF-8，避免中文亂碼
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    ap =argparse.ArgumentParser(description='影片抽幀（含去重、挑清晰幀）')
    ap.add_argument('inputs', nargs='+', help='影片檔或資料夾（不遞迴）')
    ap.add_argument('--out', required=True, help='輸出資料夾')
    ap.add_argument('--prefix', default='own_', help='檔名前綴，預設 own_（自錄）')
    ap.add_argument('--interval', type=float, default=1.0, help='每幾秒取一張（建議 1~2），預設 1.0')
    ap.add_argument('--diff-thresh', type=float, default=6.0,
                    help='與上一張已存圖的縮圖差異低於此值就跳過（0~255），預設 6.0'
                         '（實測：站著不動約 0~5、走路 1 秒約 10~40）；調高=更少圖')
    ap.add_argument('--samples-per-window', type=int, default=5,
                    help='每個時間窗評估幾幀來挑最清晰的，預設 5')
    ap.add_argument('--min-sharpness', type=float, default=15.0,
                    help='Laplacian variance 低於此值視為糊掉而丟棄，預設 15（很保守）')
    ap.add_argument('--quality', type=int, default=95, help='JPEG 品質，預設 95')
    ap.add_argument('--max-seconds', type=float, help='每支影片只處理前 N 秒（小量測試用）')
    ap.add_argument('--dry-run', action='store_true', help='只統計不寫檔')
    args = ap.parse_args(argv)
    if not 0.2 <= args.interval <= 30:
        ap.error('--interval 需介於 0.2 與 30 秒')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.csv'
    manifest = load_manifest(manifest_path)
    grand = dict(saved=0, windows=0)
    for video in collect(args.inputs):
        # 重跑同一影片：先清掉舊的列與檔案，避免參數不同時留下重複圖。
        old = [r for r in manifest if r['source'] == video.name]
        if old and not args.dry_run:
            for r in old:
                (out / r['file']).unlink(missing_ok=True)
            manifest = [r for r in manifest if r['source'] != video.name]
        rows, st = extract(video, out, args.prefix, args.interval, args.diff_thresh,
                           args.samples_per_window, args.min_sharpness, args.quality,
                           args.max_seconds, args.dry_run)
        manifest += rows
        grand['saved'] += st['saved']
        grand['windows'] += st['windows']
        print(f'{video.name}: {st["duration_s"]:.1f}s @ {st["fps"]:.2f}fps, 時間窗 {st["windows"]}, '
              f'存 {st["saved"]}, 太像跳過 {st["skipped_similar"]}, 太糊跳過 {st["skipped_blurry"]}')
        if not args.dry_run:
            save_manifest(manifest_path, manifest)
    print(f'完成：共存 {grand["saved"]} 張 / {grand["windows"]} 個時間窗 → {out}'
          + ('（dry-run，未寫檔）' if args.dry_run else ''))


if __name__ == '__main__':
    main()
