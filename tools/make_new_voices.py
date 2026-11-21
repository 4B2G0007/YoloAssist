"""產生新類別（斑馬線、上樓梯、下樓梯）的提示語音 WAV。

檔名與句型完全照 assist/voice_catalog.py 對「新增類別」的規則：
  normal_{left,center,right}_<類別>.wav  「左前方有斑馬線。」「前方有…」「右前方有…」
  urgent_center_<類別>.wav               「注意，前方有斑馬線。」
（日後把這三類加進 assist/config.py 的 CLASSES／NEW_CLASSES 後，voice_catalog 會要求的就是這些檔。）

語音引擎：Windows 內建 SAPI（Microsoft Hanhan, zh-TW），離線、免安裝。
輸出 48 kHz、16-bit、單聲道，與現有 Clipchamp 語音格式相同，並裁掉前後靜音。
注意：聲音和既有 Clipchamp 錄音不同；要一致的話，可用 manifest 的文字在 Clipchamp 以同一個聲音重錄，
再用相同檔名覆蓋（tools/check_voices.py 只檢查格式）。

用法：
  python tools/make_new_voices.py            # 不覆蓋已存在的檔
  python tools/make_new_voices.py --force    # 覆蓋
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NEW = [('crosswalk', '斑馬線'), ('stairs_up', '上樓梯'), ('stairs_down', '下樓梯')]
DIRECTIONS = {'left': '左前方', 'center': '前方', 'right': '右前方'}
VOICE = 'Microsoft Hanhan Desktop'
RATE_HZ = 48000


def catalog():
    out = {}
    for slug, zh in NEW:
        for d, word in DIRECTIONS.items():
            out[f'normal_{d}_{slug}.wav'] = f'{word}有{zh}。'
        out[f'urgent_center_{slug}.wav'] = f'注意，前方有{zh}。'
    return out


PS = r'''
param([string]$Text, [string]$Out, [string]$Voice, [int]$Rate, [int]$Hz)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice($Voice)
$s.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo($Hz, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($Out, $fmt)
$s.Speak($Text)
$s.Dispose()
'''


def synth(text, out: Path, rate: int):
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / 'tts.ps1'
        script.write_text(PS, encoding='utf-8-sig')  # BOM：讓 PowerShell 5.1 正確讀 UTF-8
        raw = Path(td) / 'raw.wav'
        r = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                            '-File', str(script), '-Text', text, '-Out', str(raw), '-Voice', VOICE,
                            '-Rate', str(rate), '-Hz', str(RATE_HZ)], capture_output=True, text=True)
        if r.returncode or not raw.is_file():
            raise RuntimeError(f'TTS 失敗：{text}\n{r.stderr[-500:]}')
        with wave.open(str(raw)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE_HZ, 1, 2)
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    # 裁掉前後靜音（保留 60 ms 緩衝），播放延遲更短。
    loud = np.flatnonzero(np.abs(pcm.astype(np.int32)) > 500)
    if len(loud):
        pad = int(0.06 * RATE_HZ)
        pcm = pcm[max(0, loud[0]-pad):min(len(pcm), loud[-1]+pad)]
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE_HZ)
        w.writeframes(pcm.tobytes())
    return len(pcm) / RATE_HZ


def main(argv=None):
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--audio-dir', default=str(ROOT / 'audio'))
    ap.add_argument('--rate', type=int, default=1, help='SAPI 語速 -10~10，預設 1（略快）')
    ap.add_argument('--force', action='store_true', help='覆蓋已存在的 WAV')
    args = ap.parse_args(argv)
    folder = Path(args.audio_dir)
    manifest_path = folder / 'manifest_v1.2_new_classes.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    for order, (name, text) in enumerate(catalog().items(), 1):
        path = folder / name
        if path.is_file() and not args.force:
            print(f'略過（已存在）：{name}')
            continue
        seconds = synth(text, path, args.rate)
        manifest[name] = dict(text=text, seconds=seconds, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              source=f'Windows SAPI {VOICE} (zh-TW), rate={args.rate}', order=order)
        print(f'{name}  {seconds:.2f}s  {text}')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'manifest：{manifest_path}')


if __name__ == '__main__':
    main()
