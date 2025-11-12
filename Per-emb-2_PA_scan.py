"""
掃描不同 position angle (PA)：
對每個 PA：
  - 用環境變數 PA_OVERRIDE_DEG, PA_TAG 呼叫 Per-emb-2.py
  - Per-emb-2.py 自己跑 grid search + 畫圖
  - 圖片會存在 Per-emb-2_plots/ 中，檔名含對應 PA 標籤
"""

import os
import subprocess
import numpy as np

SCRIPT = "Per_emb_2.py"
OUTPUT_DIR = "Per-emb-2_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 想掃描的 PA 清單（你可以改）
pa_list = np.arange(-90, 91, 10)

for pa in pa_list:
    pa = int(pa)
    pa_tag = f"PA_{pa:+03d}"

    print(f"\n[SCAN] Run {SCRIPT} with PA={pa:+03d} deg")

    env = os.environ.copy()
    env["PA_OVERRIDE_DEG"] = str(pa)
    env["PA_TAG"] = pa_tag

    # 呼叫一次 Per-emb-2.py
    subprocess.run(
        ["python", SCRIPT],
        env=env,
        check=True,
    )

print("\n[SCAN DONE] All PAs finished. Check Per-emb-2_plots/ for outputs.")