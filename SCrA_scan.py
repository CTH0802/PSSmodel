import os
import subprocess
import numpy as np

SCRIPT = "SCrA.py"
OUTPUT_DIR = "SCrA_results/PA_scan"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pa_list = np.arange(0, 360, 10)

for pa in pa_list:
    pa = int(pa)
    pa_tag = f"PA_{pa:+03d}"

    print(f"\n[SCAN] Run {SCRIPT} with PA={pa:+03d} deg")

    env = os.environ.copy()
    env["PA_OVERRIDE_DEG"] = str(pa)
    env["PA_TAG"] = pa_tag
    env["PLOT_DIR"] = OUTPUT_DIR

    # 呼叫一次 Per-emb-2.py
    subprocess.run(
        ["python", SCRIPT],
        env=env,
        check=True,
    )

print("\n[SCAN DONE] All PAs finished. Check SCrA_results/PA_scan/ for outputs.")