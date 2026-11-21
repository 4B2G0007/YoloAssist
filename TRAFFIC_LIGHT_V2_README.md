# YoloAssist — Traffic Light V2.1 顯示與反應更新

使用者截圖的 YOLO 0.31 小於 V2 的門檻 0.40，因此停在 below_yolo_confidence，沒有進入燈色分析。灰框表示未確認的 Debug 候選。

V2.1 預設門檻改為 0.30，同色確認改為最近 5 次實際推論中的 2 次。候選第一次出現就畫亮青粗框，確認後再換成紅、黃、綠框；保留 HSV、形狀與明暗對比過濾。

這能減少門檻與確認造成的等待，**不代表已測得 GPU 推論加速，也不能保證模型漏檢的號誌都會找到。** 沒有收到完整 street4.mp4；本次為靜態與模擬測試。

## 安裝

1. 關閉 main.py，備份原專案（可繼續使用現有 yolo-env）。
2. 將 YoloAssist-TrafficLight-V2.1-Update.zip 解壓到 main.py、assist/、設定檔所在的專案根目錄。
3. 合併資料夾並覆蓋同名檔，保留 assist/ 中其他檔案。此包包含完整 V2 整合所需檔案，已裝 V2 或原 GitHub 基準版本皆可套用。
4. 啟用原本可跑 YOLO 的環境。模型、影片、音檔與環境不在更新包內。

## Windows 測試指令

適合目前已能使用 CUDA 的 RTX 4060 Ti 桌機；逐張推論，先省去錄影與日誌寫入。指令不啟用聲音：

~~~bat
python main.py --source videos/street4.mp4 --traffic-light --show --realtime --device 0 --stride 1 --tl-conf 0.30 --tl-min-frames 2 --no-record
~~~

需要看過濾原因時加 --tl-debug。需要保存錄影、JSONL、CSV 時，移除 --no-record 並加 --output results/tl-v21。

遠方小號誌仍常漏框時，可比較較大輸入尺寸：

~~~bat
python main.py --source videos/street4.mp4 --traffic-light --show --realtime --device 0 --stride 1 --imgsz 960 --tl-conf 0.30 --tl-min-frames 2 --no-record
~~~

--imgsz 960 提高共用 YOLO 輸入解析度，可能幫助小目標，但增加推論成本；不是加速開關。預設仍沿用設定檔的 640。原始燈光像素不足時，增大推論尺寸也無法補出可靠燈色。

GPU 不可用時，將 --device 0 改成 --device cpu，或省略此參數。Pi 沿用 --source picamera2 --device cpu，不直接套用桌機的 960 指令。

## 框與標籤

| 顯示 | 意義 |
| --- | --- |
| 亮青 LIGHT: CHECKING | 候選達門檻、當次有可用燈色，但還在等待確認 |
| 亮青 LIGHT: UNKNOWN | 候選達門檻，但仍無法可靠判斷燈色 |
| 亮紅／亮黃／亮綠 | 當次顏色證據與歷史確認條件同時成立 |
| 洋紅 LIGHT: LOW CONF | 未達門檻，僅 Debug 顯示，不投同色票 |
| YOLO 0.31 | 框的偵測信心值，不是燈色正確機率 |
| HELD 0.10s | stride 跳過影格沿用舊結果，不算新票 |

框使用 YOLO 座標，並有深色外緣；標籤盡量放在燈箱外，避開其他號誌和原物件。上緣框不再被不透明狀態列整個蓋住。畫面空間不足時仍可能局部重疊。

docs/traffic-light-v2/PREVIEW_V21_SYNTHETIC.png 是實際繪製程式輸出的合成示意，並非街景實測。

## 參數

| 參數 | V2.1 預設 | 用途 |
| --- | --- | --- |
| --traffic-light | 關閉 | 開啟號誌模組，省略時保持原物件流程 |
| --tl-conf | 0.30 | YOLO 號誌門檻，原為 0.40 |
| --tl-min-contrast | 22 | 明暗對比門檻，保持不變 |
| --tl-min-frames | 2 | 最近 5 次實際推論至少幾次同色，原為 3 |
| --tl-debug | 關閉 | 額外顯示診斷與低信心候選 |
| --imgsz | 設定檔值，通常 640 | 新增命令列覆寫，所有物件仍共用一次推論 |

舊指令若明寫 --tl-conf 0.40 --tl-min-frames 3，仍會沿用舊門檻；請改用上方新指令。較低門檻會增加候選框。若誤框太多，可調回 0.40；若燈色跳動，可調成 3 次確認。

stride=1 且持續同色時，影格 0 顯示候選、影格 1 最早確認。stride=3 時，影格 0 顯示候選、影格 3 最早確認。UNKNOWN 與漏檢仍排擠舊票；切換影片、時間倒退或影像過時會重置。低推論 FPS、長 stride、模型只在近處給框等問題仍可能造成延遲。

## 相容性與紀錄

同一模型、同一次 YOLO、同一視窗；CPU/CUDA、Picamera2、批次、原物件/方向/危險提示、錄影、JSONL、CSV 與原音訊選項保留。號誌不產生語音或通行決策。綠燈結果不等於可以安全過馬路。

保留 V2 的 traffic_lights、traffic_lights_inferred 與 run_config 設定，每筆號誌另有 required_observations、minimum_detector_confidence。尚未確認的 color 仍為 UNKNOWN，亮青框不會記成確認的燈色。

## 修改檔案

相對 V2，本次修改 main.py、assist/traffic_light.py、assist/render.py、tests/test_traffic_light.py，以及安裝/測試文件。另附未再修改的 assist/detector.py、assist/runner.py，便於一次套用完整整合。

包內有安裝說明、測試報告、合成預覽與 SHA-256 清單。**無新增 Python 套件需求。** 沿用 ultralytics、torch、opencv-python、numpy、Pillow。Windows 顯示需要有 GUI 的 OpenCV；不需重裝已能使用 CUDA 的 torch。

~~~bat
python -m compileall -q main.py assist tests/test_traffic_light.py
python -m unittest discover -s tests -p test_traffic_light.py -v
~~~

本次未測真實街景準確率與反應秒數、Windows 實際視窗、CUDA 或 Pi 硬體，詳見 docs/traffic-light-v2/TEST_REPORT.md。
