# The Cutaway · IndicF5 (AI4Bharat, MIT) voices on a GitHub Actions CPU runner.
# Usage: python f5_cpu.py <lines.json> <shard> <nshards> <takes> <outdir>
# lines.json: [{"id": "X_R01", "ref": "MAR_F_WIKI_00001", "text": "<Devanagari>", "speed": 1.0}]
import json, os, sys, subprocess, urllib.request, time
src, shard, n, takes, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(out, exist_ok=True)
REFS = {  # official AI4Bharat IndicF5 demo prompts (github.com/AI4Bharat/IndicF5/prompts) + transcripts
  "MAR_F_WIKI_00001": "दिगंतराव्दारे अंतराळ कक्षेतला कचरा चिन्हित करण्यासाठी प्रयत्न केले जात आहे.",
  "MAR_M_WIKI_00001": "या प्रथाला एकोणीसशे पंचातर ईसवी पासून भारतीय दंड संहिताची धारा चारशे अठ्ठावीस आणि चारशे एकोणतीसच्या अन्तर्गत निषेध केला.",
  "PAN_F_HAPPY_00002": "ਇੱਕ ਗ੍ਰਾਹਕ ਨੇ ਸਾਡੀ ਬੇਮਿਸਾਲ ਸੇਵਾ ਬਾਰੇ ਦਿਲੋਂਗਵਾਹੀ ਦਿੱਤੀ ਜਿਸ ਨਾਲ ਸਾਨੂੰ ਅਨੰਦ ਮਹਿਸੂਸ ਹੋਇਆ।",
}
L = json.load(open(src))
jobs = [(l, t) for l in L for t in range(1, takes + 1)]
jobs = [j for i, j in enumerate(jobs) if i % n == shard]
if not jobs: sys.exit(0)
for r in {l.get("ref", "MAR_F_WIKI_00001") for l, _ in jobs}:
    if r not in REFS and os.path.exists(f"refs/{r}.wav"): continue
    urllib.request.urlretrieve(f"https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/{r}.wav", f"{r}.wav")
import numpy as np, soundfile as sf, torch
from transformers import AutoModel
m = AutoModel.from_pretrained("Aditya02/IndicF5", trust_remote_code=True)
t0 = time.time()
for l, t in jobs:
    r = l.get("ref", "MAR_F_WIKI_00001"); rt = l.get("ref_text") or REFS.get(r) or open(f"refs/{r}.txt").read().strip()
    rp = f"{r}.wav" if os.path.exists(f"{r}.wav") else f"refs/{r}.wav"
    torch.manual_seed(1000 + t * 7 + len(l["text"]))
    a = m(l["text"], ref_audio_path=rp, ref_text=rt)
    if a.dtype == np.int16: a = a.astype(np.float32) / 32768.0
    w = f"{out}/{l['id']}_t{t}.wav"; sf.write(w, np.asarray(a, dtype=np.float32), 24000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-b:a", "160k", w[:-4] + ".mp3"], check=True); os.remove(w)
    print("done", l["id"], t, round(time.time() - t0), flush=True)
