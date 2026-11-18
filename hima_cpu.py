# The Cutaway · Hima TTS nano (Hindi, 6 built-in voices) on a CPU GitHub Actions runner.
# Usage: python hima_cpu.py jobs/<job>/lines.json <shard> <nshards> <takes> <outdir>
# lines.json: [{"id": "HR_R01", "voice": "Siddharth|Ishaan|Nikhil|Meera|Sneha|Nandini", "pitch": 0, "text": "..."}]
import json, os, subprocess, sys, time
path, shard, nsh, takes, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), os.path.abspath(sys.argv[5])
path = os.path.abspath(path); os.makedirs(out, exist_ok=True)
allines = json.load(open(path)); buckets = [[] for _ in range(nsh)]; load = [0] * nsh
for l in sorted(allines, key=lambda x: -len(x["text"])):  # balance shards by text length
    i = load.index(min(load)); buckets[i].append(l); load[i] += len(l["text"]) + 40
lines = buckets[shard]
if not lines: print("nothing for this shard"); sys.exit(0)
t0 = time.time()
from huggingface_hub import snapshot_download
H = os.path.abspath(snapshot_download("Prahaladha/hima-tts-nano-hi", local_dir="hima"))
sys.path.insert(0, H); os.chdir(H)
import soundfile as sf, torch
torch.set_num_threads(os.cpu_count() or 4)
from hima_tts import TTS
tts = TTS(".")
VOICES = {"Siddharth", "Ishaan", "Nikhil", "Meera", "Sneha", "Nandini"}
print("loaded in", round(time.time() - t0), "s", flush=True)
SEEDS = [7, 21, 42][:takes]
for l in lines:
    v = l["voice"]; assert v in VOICES, f"unknown voice {v}"
    for k, seed in enumerate(SEEDS):
        t1 = time.time(); w = f"{out}/{l['id']}_t{k+1}.wav"; m = f"{out}/{l['id']}_t{k+1}.mp3"
        sf.write(w, tts.say(l["text"], voice=v, seed=seed), 24000)
        p = l.get("pitch", 0) or 0; r = 2 ** (p / 12)
        af = ["-af", f"asetrate={24000*r:.0f},aresample=24000,atempo={1/r:.4f}"] if p else []
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, *af, "-b:a", "96k", m], check=True); os.remove(w)
        print("done", l["id"], v, f"t{k+1}", round(time.time() - t1), "s", flush=True)
print("SHARD DONE", round(time.time() - t0), "s")
