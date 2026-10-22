# YOLOAssist V1.1.1：Raspberry Pi 5＋Camera Module 3 移交版

本版根據組員提供的 `stream.py` 相機設定接入現有 YOLO 程式。請先停止
`stream.py`、`rpicam-hello` 或其他使用相機的程式，再啟動本版。

**這是相機／平台相容更新，不是新的碰撞預測系統。** 原本的辨識類別、ROI、
距離比例門檻、提示冷卻與語音優先序均維持 V1.1。不宣告道路安全，不能取代
白手杖、導盲犬或陪同者；測試時不要製造真實交通碰撞情境。

## 1. 先用組員現有環境檢查，不必重新安裝全部套件

將整個 `Grad-RPi5-Camera3` 解壓縮到樹莓派家目錄。以下範例假設資料夾在
`~/Grad-RPi5-Camera3`，如果你改名，請調整 `cd` 路徑。

使用組員先前能跑 `stream.py` 的 Python／虛擬環境：

```bash
cd ~/Grad-RPi5-Camera3
python tools/check_pi.py
python tools/check_voices.py
```

第一個工具會檢查目前 Python 的 NumPy、OpenCV、PyTorch、Ultralytics、
Picamera2、libcamera、模型檔及 Linux 音訊播放器；不會安裝套件，也不會開相機。
第二個預期為 `expected: 74`、`valid: 74`，且 `missing`／`invalid` 都是空陣列。
若模組都能匯入，不必為了本更新重新安裝。

### 只有環境不齊時才處理

- 已經能跑相機，不必重灌系統，也不必更動相機接線或重新啟用舊相機介面。
- 不要把 Windows 的 `yolo-env` 搬過來，Pi 使用 Linux ARM64 CPU 套件。
- Picamera2／libcamera 優先使用 Raspberry Pi OS 的 apt 套件，不用 pip 重裝相機堆疊。
- 若需要新的虛擬環境，請用 Raspberry Pi OS 的系統 Python 建立：

```bash
python3 -m venv --system-site-packages yolo-env
source yolo-env/bin/activate
```

`--system-site-packages` 讓環境看得到原本的相機系統套件。只有缺少 Picamera2 時才執行：

```bash
sudo apt update
sudo apt install python3-picamera2
```

只有缺少 YOLO 時，才在虛擬環境內安裝；本版以 `ultralytics==8.4.175` 驗證：

```bash
python -m pip install ultralytics==8.4.175
```

請勿使用桌機 CUDA 的安裝指令，不需要 NVIDIA 驅動。安裝相依套件可能改變
NumPy／OpenCV 版本，因此安裝後必須重跑 `check_pi.py --capture`，確認相機堆疊
仍能匯入及取像；若出現 NumPy binary incompatibility，不要持續升級／降級試錯，
請保留錯誤訊息與 `python -m pip list` 供排查。Trixie／Bookworm 的系統套件版本不同，
此壓縮檔不包含也不保證任何一組預先建置的 ARM Python 環境。

## 2. 單獨確認相機輸入

先停止仍在執行的 `stream.py`（在它的終端機按 Ctrl+C），再執行：

```bash
python tools/check_pi.py --capture
```

工具會開啟相機、完成約一秒暖機、擷取一張並關閉相機，不會下載模型或執行
YOLO 推論。預期 `capture.ok: true`、`shape: [480, 640, 3]`、`color_order: BGR`。
Module 3 應有 `autofocus_applied: true`；這只表示已送出連續自動對焦設定，
不保證每一張都已合焦。缺少 Linux 播放器會顯示警告，但不影響取像檢查。

## 3. 啟動辨識與語音

建議第一次明確指定來源：

```bash
python main.py --source picamera2 --audio
```

程式偵測到真正的 Raspberry Pi 後，也可以直接：

```bash
python main.py --audio
```

- 使用 `config.pi.json`、CPU、640×480、`XBGR8888` 相機輸入。
- 依 Picamera2 的實際記憶體排列 `[R,G,B,X]` 轉成三通道 BGR，才交給 YOLO。
- Module 3 啟用連續自動對焦；其他不支援對焦的模組只警告，不強制設定。
- 相機請求 30 FPS；**不代表 YOLO 推論能達到 30 FPS**。
- 相機背景執行緒只保留最新畫面，不累積舊畫面等待推論。
- 預設不開 GUI 視窗、不開網頁，可在 SSH 終端機執行。
- `--audio` 才會播放，單純 `python main.py` 不會有語音。
- 按 Ctrl+C 停止；預設保留完整測試紀錄。

若樹莓派本機接螢幕、在桌面終端機執行，可以加 `--show`：

```bash
python main.py --source picamera2 --audio --show
```

視窗聚焦時，Q／Esc 停止，S 主動描述最多三個目標，M 切換 walk／classic。
SSH／沒有圖形桌面時不要加 `--show`，也不提供 S／M 全域鍵盤控制。

若暫時不需要紀錄、只想看運作狀態：

```bash
python main.py --source picamera2 --audio --no-record
```

`--no-record` 不會產生影片或測試檔。若要交回結果供優化，請勿加此參數。

## 4. 音效檢查

語音從樹莓派的預設音效裝置播放，不會傳回 SSH 筆電，也不會透過網頁播放。
先確認選好的 USB／HDMI／其他音效裝置可以用 Linux 播放器播放：

```bash
aplay audio/quality_recovered.wav
```

缺少 `aplay` 時才安裝 `alsa-utils`。若有裝置卻沒聲音，先檢查音量和預設輸出。
原本的 walk 模式不會看到物件就一直說話，同場景近物預設約 12 秒才重複，
因此「沒有提示」不一定是音訊故障；看 events.csv 的 decision／playback／reason。
本次沒有改成「越靠近越急」，也未修訂語音冷卻／排程邏輯。

## 5. 影片與桌機仍可用

你附的 `street.mp4` 是 AV1，部分 Linux OpenCV 無法解碼。原檔完整保留；
另附從它轉出的 `street_h264.mp4` 作相容測試，不會自動覆寫來源影片。

Pi 上先跑 100 幀：

```bash
python main.py --source street_h264.mp4 --device cpu --max-frames 100
```

`status: limited` 是因為你指定了 `--max-frames`，並非失敗。跑完整影片時拿掉它。
影片純分析模式預設不播放語音；不要將離線分析速度當成即時警示能力。

桌機維持原來用法，會選 `config.v1.1.json`，有 CUDA 可用時 `auto` 仍選 GPU：

```bash
python main.py --source street.mp4 --show
```

影片要有即時語音必須 `--realtime --audio`。Pi 追不上影片時間時仍會抑制過期提示，
本次沒有取消這個保護；需要先確認實際延遲，不保證 .pt／CPU 即時效能。

## 6. 可調的相機設定

優先修改 `config.pi.json`，不必編輯 Python：

| 設定 | 預設 | 說明 |
| --- | --- | --- |
| camera_size | [640,480] | 沿用組員測試解析度 |
| camera_format | XBGR8888 | 沿用組員實際程式，不是它註解所寫的 RGB888 |
| camera_index | 0 | Picamera2 相機編號，不是 USB 攝影機編號 |
| camera_fps | 30 | 請求擷取率，不是推論率 |
| camera_autofocus | continuous | Module 3 連續對焦；default 表示不指定 |
| camera_warmup_s | 1.0 | 初始曝光／對焦暖機時間 |
| camera_timeout | 5.0 | 等待相機畫面的逾時秒數 |
| device | cpu | Pi 不使用 NVIDIA CUDA |
| imgsz | 640 | 保留原本模型輸入大小，未暗中降低解析度 |

部分設定也能直接從指令覆蓋，例如第二顆相機：

```bash
python main.py --source picamera2 --camera-index 1 --audio
```

支援 RGB888、BGR888、XBGR8888、XRGB8888 的正確色彩轉換。不必盲目試換紅藍通道。
`config.pi.json` 是獨立設定，未列出的判斷參數使用 `assist/config.py` 的 V1.1 預設；
不會讀入你日後另外修改的 `config.v1.1.json`。若要自訂 Pi 門檻，請明確寫進 Pi 設定。

## 7. 紀錄與已知限制

每次執行在 `results/` 產生獨立資料夾，含標記影片、逐幀 JSONL、事件 CSV、summary、
run_config 與 review_notes。相機 summary 另含請求的相機格式／對焦設定狀態。

- JSONL 的時間戳是分析依據。相機標記影片仍是舊版名目 10 FPS，不能當成實際牆鐘速度。
- MP4 沒有提示音軌；判斷與播放狀態請看 CSV。
- 部分超過兩秒的相機推論仍會被判定過期而抑制警示，CPU 太慢時應先檢查 latency。
- 本次沒有 NCNN、Hailo、測距、TTC 或新風險分級。
- 本次沒有 Flask 網頁／8080 服務；`reference/stream.py` 只是原檔留存，不會被主程式匯入。
- 不要同時啟動原串流程式與主程式；若未來要看瀏覽器畫面，應共用同一相機擷取來源。
- 軟體驗證在 Linux x86-64 CPU 進行，並非實體 ARM Pi 效能或相機／喇叭驗證。

交回測試結果時，請提供 `*_summary.json`、`*_events.csv`、`run_config.json`；
需要排查逐幀行為時再附 `*_detections.jsonl` 及相關影片片段。

## 8. 組員下載及更新

ZIP 內已附模型、74 段語音與兩份測試影片，不必另外下載模型。
如要放 GitHub，上傳這個資料夾內的原始檔，不要只上傳 ZIP。
`.gitignore` 排除虛擬環境、暫存、驗證輸出與影片，但保留 `.pt`／`.wav`。
因此 GitHub clone 不會自動取得被排除的影片，請另外分享影片。

```bash
git clone https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

下載只取得程式及有提交的素材，不會安裝 Python 環境。Private 儲存庫仍須授權與登入。

## 9. 軟體測試

```bash
python -m unittest discover -s tests -v
```

包含原 V1.1 回歸測試與新增的色彩轉換、平台預設、相機參數、對焦控制、錯誤提示、
釋放資源及模擬相機到紀錄輸出的測試。模擬相機結果不代表真實相機已通過驗收。

## 參考與授權

- Raspberry Pi Picamera2 官方手冊（格式、對焦、安裝）：
  https://pip.raspberrypi.com/documents/RP-008156-DS-picamera2-manual.pdf
- Picamera2 官方程式碼（格式排列）：
  https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py
- YOLOAssist 與 Ultralytics 相關授權見 `LICENSE`（AGPL-3.0）。保留模型及原始碼的授權資訊。
- 語音及測試影片由專案使用者提供；公開上傳前請自行確認素材權利與路人隱私。
