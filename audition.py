# The Cutaway · voice audition on a GitHub Actions CPU runner. Usage: python audition.py <engine:voice> <outdir>
import os, sys, subprocess, json, time
item, out = sys.argv[1], os.path.abspath(sys.argv[2]); os.makedirs(out, exist_ok=True); ROOT = os.getcwd()
eng, voice = item.split(":", 1)
P = {"words": "नमस्ते दोस्तों! आज की कहानी में ज्ञान, क्षमा, श्रद्धा, ऋषि और दृष्टि जैसे शब्द हैं। मेरा phone अचानक hang हो गया, तो मैंने WhatsApp पर message भेजा, और Google Maps से रास्ता ढूँढा। उन्नीस सौ सैंतालीस में, भारत आज़ाद हुआ।",
     "story": "रात के ठीक दो बजे… चलती ट्रेन में, एक बुज़ुर्ग के बैग से, चालीस हज़ार रुपये गायब! तीन लोग जाग रहे थे। और उनमें से एक, झूठ बोल रहा है।"}
pip = lambda *a: subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=True)
def save(a, sr, name):
    import soundfile as sf
    w = f"{out}/{name}.wav"; sf.write(w, a, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-b:a", "128k", f"{out}/{name}.mp3"], check=True); os.remove(w)
t0 = time.time()
if eng == "parler":
    pip("git+https://github.com/huggingface/parler-tts.git", "soundfile")
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    m = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").eval()
    tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts"); dtok = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
    desc = f"{voice} speaks in a warm, expressive storytelling voice at a moderate pace with a slightly low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up, and very clear audio with no background noise."
    for tag, txt in P.items():
        torch.manual_seed(21); d = dtok(desc, return_tensors="pt"); p = tok(txt, return_tensors="pt")
        with torch.inference_mode(): a = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask, do_sample=True, temperature=0.8)
        save(a.numpy().squeeze(), m.config.sampling_rate, f"parler_{voice}_{tag}"); print("done", tag, round(time.time() - t0), flush=True)
elif eng == "veena":
    pip("transformers>=4.44,<4.57", "accelerate", "snac", "soundfile")
    sys.argv = ["x", "/dev/null", "0", "1", "1", out]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    vm = AutoModelForCausalLM.from_pretrained("maya-research/Veena", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval()
    vt = AutoTokenizer.from_pretrained("maya-research/Veena", trust_remote_code=True); snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    SOS, EOS, SOH, EOH, SOA, EOA, OFF = 128257, 128258, 128259, 128260, 128261, 128262, 128266
    for tag, txt in P.items():
        torch.manual_seed(21); ids = vt.encode(f"<spk_{voice}> {txt}", add_special_tokens=False); inp = [SOH, *ids, EOH, SOA, SOS]
        with torch.inference_mode(): o = vm.generate(torch.tensor([inp]), max_new_tokens=min(int(len(txt) * 1.3) * 7 + 21, 2400), do_sample=True, temperature=0.45, top_p=0.9, repetition_penalty=1.05, pad_token_id=vt.pad_token_id, eos_token_id=[EOS, EOA])
        t = [i for i in o[0][len(inp):].tolist() if OFF <= i < OFF + 7 * 4096]; t = t[: len(t) // 7 * 7]; L = [[], [], []]; off = [OFF + i * 4096 for i in range(7)]
        for i in range(0, len(t), 7):
            L[0].append(t[i] - off[0]); L[1] += [t[i + 1] - off[1], t[i + 4] - off[4]]; L[2] += [t[i + 2] - off[2], t[i + 3] - off[3], t[i + 5] - off[5], t[i + 6] - off[6]]
        with torch.inference_mode(): a = snac.decode([torch.tensor(c, dtype=torch.int32).unsqueeze(0).clamp(0, 4095) for c in L]).squeeze().clamp(-1, 1).numpy()
        save(a, 24000, f"veena_{voice}_{tag}"); print("done", tag, round(time.time() - t0), flush=True)
elif eng == "hima":
    pip("huggingface_hub", "soundfile")
    from huggingface_hub import snapshot_download
    d = snapshot_download("Prahaladha/hima-tts-nano-hi", local_dir="hima")
    pip("-r", "hima/requirements.txt"); sys.path.insert(0, "hima"); os.chdir("hima")
    from hima_tts import TTS
    tts = TTS(".")
    for v in ["Siddharth", "Ishaan", "Nikhil", "Meera", "Sneha", "Nandini"]:
        for tag, txt in P.items():
            a = tts.say(txt, voice=v, seed=21); save(a, 24000, f"hima_{v}_{tag}")
        print("done", v, round(time.time() - t0), flush=True)
elif eng == "cbx":
    pip("huggingface_hub")
    from huggingface_hub import snapshot_download
    snapshot_download("ResembleAI/Chatterbox-Multilingual-TTS-hi", repo_type="space", local_dir="cbx")
    reqs = [l.strip() for l in open("cbx/requirements.txt") if l.strip() and not l.startswith(("gradio", "spaces", "#"))]
    pip(*reqs); pip("soundfile"); sys.path.insert(0, "cbx")
    import torch, urllib.request
    from chatterbox.src.chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained("cpu")
    ref = {"hi_f1": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac"}.get(voice)
    if ref: urllib.request.urlretrieve(ref, "ref.flac"); ref = "ref.flac"
    else: ref = f"{ROOT}/refs/{voice}.mp3"
    for tag, txt in P.items():
        torch.manual_seed(21); w = m.generate(txt, audio_prompt_path=ref, language_id="hi", exaggeration=0.5, temperature=0.8, cfg_weight=0.5)
        save(w.squeeze(0).numpy(), m.sr, f"chatterbox_{voice}_{tag}"); print("done", tag, round(time.time() - t0), flush=True)
print("ITEM DONE", item, round(time.time() - t0))
