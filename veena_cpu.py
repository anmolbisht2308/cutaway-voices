# The Cutaway · Veena TTS on a CPU-only GitHub Actions runner. Usage: python veena_cpu.py jobs/<job>/lines.json <shard> <nshards> <takes> <outdir>
# lines.json: [{"id": "HR_R01", "voice": "agastya|vinaya|kavya|maitri", "pitch": 0, "text": "..."}]
import json, os, subprocess, sys, time, torch, soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
path, shard, nsh, takes, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(out, exist_ok=True); torch.set_num_threads(os.cpu_count() or 4)
allines = json.load(open(path)); buckets = [[] for _ in range(nsh)]; load = [0] * nsh
for l in sorted(allines, key=lambda x: -len(x["text"])):  # balance shards by text length (longest first)
    i = load.index(min(load)); buckets[i].append(l); load[i] += len(l["text"]) + 40
lines = buckets[shard]
if not lines: print("nothing for this shard"); sys.exit(0)
t0 = time.time()
RID = "maya-research/Veena"
vm = AutoModelForCausalLM.from_pretrained(RID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval()
vt = AutoTokenizer.from_pretrained(RID, trust_remote_code=True)
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
print("loaded in", round(time.time() - t0), "s", flush=True)
SOS, EOS, SOH, EOH, SOA, EOA, OFF = 128257, 128258, 128259, 128260, 128261, 128262, 128266
def say(text, spk, seed, temp=0.45):
    torch.manual_seed(seed)
    ids = vt.encode(f"<spk_{spk}> {text}", add_special_tokens=False); inp = [SOH, *ids, EOH, SOA, SOS]
    x = torch.tensor([inp]); mx = min(int(len(text) * 1.3) * 7 + 21, 2000)
    with torch.inference_mode():
        o = vm.generate(x, max_new_tokens=mx, do_sample=True, temperature=temp, top_p=0.9, repetition_penalty=1.05, pad_token_id=vt.pad_token_id, eos_token_id=[EOS, EOA])
    t = [i for i in o[0][len(inp):].tolist() if OFF <= i < OFF + 7 * 4096]; t = t[: len(t) // 7 * 7]
    L = [[], [], []]; off = [OFF + i * 4096 for i in range(7)]
    for i in range(0, len(t), 7):
        L[0].append(t[i] - off[0]); L[1] += [t[i + 1] - off[1], t[i + 4] - off[4]]; L[2] += [t[i + 2] - off[2], t[i + 3] - off[3], t[i + 5] - off[5], t[i + 6] - off[6]]
    codes = [torch.tensor(c, dtype=torch.int32).unsqueeze(0).clamp(0, 4095) for c in L]
    with torch.inference_mode(): return snac.decode(codes).squeeze().clamp(-1, 1).numpy()
SEEDS = [7, 21, 42][:takes]
for l in lines:
    for k, seed in enumerate(SEEDS):
        t1 = time.time(); w = f"{out}/{l['id']}_t{k+1}.wav"; m = f"{out}/{l['id']}_t{k+1}.mp3"
        sf.write(w, say(l["text"], l["voice"], seed), 24000)
        p = l.get("pitch", 0) or 0; r = 2 ** (p / 12)
        af = ["-af", f"asetrate={24000*r:.0f},aresample=24000,atempo={1/r:.4f}"] if p else []
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, *af, "-b:a", "96k", m], check=True); os.remove(w)
        print("done", l["id"], f"t{k+1}", round(time.time() - t1), "s", flush=True)
print("SHARD DONE", round(time.time() - t0), "s")
