"""
web.py -- a local dashboard for Forge.

    python -m forge.web        # open http://localhost:8000

Flow:
  1. Pretrain a base model on Style A (streamed loss).
  2. Attach LoRA adapters and fine-tune on Style B (streamed loss). The
     dashboard shows how few parameters are training.
  3. Compare: generate from the base vs the LoRA-adapted model side by side,
     and confirm the base weights never changed.

Standard-library HTTP server, no framework, responsive. Trains tiny models so
everything happens live on a CPU in seconds.

HONEST NOTE: the rendered appearance isn't verified from a headless sandbox; the
backend (training, adaptation, frozen-base check, generation) is tested. The
visual is confirmed by opening it in a browser.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .backbone.model import GPT
from .backbone.data import CharDataset, generate
from .backbone.optimizer import Adam
from .adapt import apply_lora, lora_params_and_grads, count_params
from .quant import quantize_int8, dequantize_int8, quantization_error, bytes_int8, bytes_of
from .adapters import perplexity


STYLE_A = ("the cat sat on the mat and the dog ran in the sun "
           "a cat and a dog sat in the sun on the mat ") * 30
STYLE_B = ("to be or not to be that is the question "
           "whether it is nobler in the mind to be ") * 30

_STATE = {"model": None, "dataset": None, "frozen_ref": None, "phase": None}
_LOCK = threading.Lock()


def _shared_dataset():
    return CharDataset(STYLE_A + STYLE_B, block_size=32)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forge</title>
<style>
  :root{
    --bg:#ffffff; --sub:#f5f5f7; --inset:#fafafa;
    --ink:#1d1d1f; --ink2:#48484a; --muted:#86868b;
    --line:#d2d2d7; --line2:#e8e8ed;
    --a:#cf5a1e; --a-press:#b34a15; --a-tint:rgba(207,90,30,.10);
    --b:#2563eb; --b-press:#1d4fd0; --b-tint:rgba(37,99,235,.10);
    --good:#1a7f4b; --bad:#c0392b;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Arial,sans-serif;
    --maxw:1000px;
  }
  *{box-sizing:border-box}
  html{-webkit-text-size-adjust:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
       font-size:17px;line-height:1.5;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}

  /* sticky, blurred nav */
  .nav{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;
       padding:13px clamp(18px,5vw,40px);
       background:rgba(255,255,255,.72);backdrop-filter:saturate(180%) blur(20px);
       -webkit-backdrop-filter:saturate(180%) blur(20px);border-bottom:1px solid var(--line2)}
  .wordmark{display:flex;align-items:center;gap:9px;font-weight:600;font-size:19px;letter-spacing:-.012em}
  .wordmark .mark{color:var(--a);display:inline-flex;align-items:center}
  .tagline{color:var(--muted);font-size:14px;margin-left:auto}

  main{max-width:var(--maxw);margin:0 auto;padding:0 clamp(18px,5vw,40px) 120px}

  /* hero: open with the thesis */
  .hero{padding:clamp(52px,9vw,100px) 0 clamp(24px,4vw,40px);max-width:770px}
  .eyebrow{font-size:14px;font-weight:600;color:var(--a);margin:0 0 18px}
  .hero h1{font-size:clamp(2.4rem,6vw,3.9rem);line-height:1.05;letter-spacing:-.025em;font-weight:600;margin:0 0 22px}
  .lede{font-size:clamp(1.1rem,2.1vw,1.3rem);line-height:1.45;color:var(--ink2);margin:0;max-width:62ch;font-weight:400}

  /* panels */
  .panel{background:var(--sub);border-radius:20px;padding:clamp(20px,3vw,30px);margin-bottom:20px}
  .phead{display:flex;align-items:center;gap:12px;margin:0 0 20px;flex-wrap:wrap}
  .step{flex:none;width:26px;height:26px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
        background:var(--line2);color:var(--ink2);font-size:13px;font-weight:600;font-family:var(--mono)}
  .step.a{background:var(--a-tint);color:var(--a)}
  .step.b{background:var(--b-tint);color:var(--b)}
  h2{font-size:19px;letter-spacing:-.014em;color:var(--ink);margin:0;font-weight:600}
  .chip{font-size:12px;font-family:var(--mono);padding:3px 11px;border-radius:980px;font-weight:500}
  .chip.a{background:var(--a-tint);color:var(--a)}
  .chip.b{background:var(--b-tint);color:var(--b)}

  .row{display:flex;gap:clamp(16px,3vw,26px);flex-wrap:wrap;align-items:flex-end}
  .fld{display:flex;flex-direction:column;gap:8px;min-width:0}
  .fld label{font-size:12px;color:var(--muted);font-weight:600}
  select,input{font-family:var(--sans);font-size:15px;border-radius:12px;border:1px solid var(--line);
    background:var(--bg);color:var(--ink);padding:10px 14px;transition:border-color .15s ease,box-shadow .15s ease}
  select{padding-right:36px;cursor:pointer;appearance:none;-webkit-appearance:none;
         background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8'><path d='M1 1.5l5 5 5-5' stroke='%2386868b' stroke-width='1.6' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>");
         background-repeat:no-repeat;background-position:right 13px center}
  select:hover,input:hover{border-color:var(--muted)}
  input:focus,select:focus{outline:none;border-color:var(--ink);box-shadow:0 0 0 3px rgba(0,0,0,.06)}

  button{font-family:var(--sans);font-size:16px;border:none;font-weight:500;cursor:pointer;
    padding:12px 26px;border-radius:980px;transition:background .15s ease,transform .06s ease}
  .pa{background:var(--a);color:#fff}
  .pa:hover{background:var(--a-press)}
  .pb{background:var(--b);color:#fff}
  .pb:hover{background:var(--b-press)}
  button:active{transform:scale(.98)}
  button:disabled{opacity:.4;cursor:default}

  .status{color:var(--muted);font-size:14px;margin-top:16px;font-family:var(--mono);min-height:20px}
  .stat{font-family:var(--mono);font-size:14px;color:var(--ink2)}

  /* loss charts: bars rise as loss falls; color set per phase in JS */
  .loss{height:124px;display:flex;align-items:flex-end;gap:2px;margin-top:16px;
        background:var(--inset);border:1px solid var(--line2);border-radius:14px;padding:0 14px 8px}
  .loss .bar{flex:1;border-radius:3px 3px 0 0;min-height:1px}

  /* base vs adapted comparison */
  .cmp{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}
  .cmp .card{background:var(--bg);border:1px solid var(--line2);border-radius:14px;padding:16px}
  .cmp h3{margin:0 0 10px;font-size:12px;letter-spacing:.02em;font-weight:600}
  .cmp .out{font-family:var(--mono);font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word;color:var(--ink)}
  .muted{color:var(--muted)}

  .spinner{display:inline-block;width:13px;height:13px;border:2px solid var(--line);border-top-color:var(--a);
    border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .ok{color:var(--good)} .no{color:var(--bad)}
  .i{vertical-align:-2px;margin-right:5px}

  footer{color:var(--muted);font-size:13px;margin-top:clamp(40px,7vw,64px);padding-top:26px;
    border-top:1px solid var(--line2);line-height:1.65;max-width:74ch}

  /* ---- Responsive: tablet and below ---- */
  @media (max-width:720px){
    .tagline{display:none}
    .fld{flex-basis:100%}
    .fld input,.fld select{width:100%}
    .row button{width:100%}
    .cmp{grid-template-columns:1fr}          /* base vs adapted stack vertically */
  }
  /* ---- Responsive: small phones ---- */
  @media (max-width:440px){
    .cmp .out{font-size:12px}
  }
  /* respect reduced-motion preference */
  @media (prefers-reduced-motion:reduce){ .spinner{animation:none} button{transition:none} }
</style>
</head>
<body>
<nav class="nav">
  <span class="wordmark"><span class="mark"><svg width="19" height="15" viewBox="0 0 20 16" fill="currentColor" aria-hidden="true"><path d="M2 3.6H13.2C13.2 5 12.2 5.9 10.9 5.9H9.2L9.9 7.6C11.9 8.1 13.3 9.5 13.6 11.3H4.4C4.7 9.4 6.1 8 8.1 7.5L8.8 5.9H5.1C3.4 5.9 2 5 2 3.6Z"/><rect x="4" y="11.2" width="10" height="2.6" rx="1"/></svg></span> forge</span>
  <span class="tagline">LoRA fine-tuning, from scratch</span>
</nav>
<main>
  <section class="hero">
    <p class="eyebrow">Adapt a frozen model</p>
    <h1>Fine-tune without touching the weights.</h1>
    <p class="lede">Forge attaches low-rank adapters to a frozen transformer and trains only those, a few percent of the model. Pretrain a base on one style, adapt it to another, then generate from both and confirm the base never moved.</p>
  </section>

  <div class="panel">
    <div class="phead"><span class="step a">1</span><h2>Pretrain the base</h2><span class="chip a">Style A: cat / dog / sun</span></div>
    <div class="row">
      <div class="fld"><label>Architecture</label>
        <select id="arch"><option value="gpt2">GPT-2 style</option><option value="llama">Llama style</option></select></div>
      <div class="fld"><label>Steps</label>
        <select id="preSteps"><option>150</option><option selected>200</option><option>300</option></select></div>
      <button class="pa" id="preBtn" onclick="pretrain()">Pretrain base</button>
    </div>
    <div class="status" id="preStatus">Trains a small base model on Style A. A few seconds on a CPU.</div>
    <div class="loss" id="preLoss"></div>
  </div>

  <div class="panel" id="ftPanel" style="opacity:.5;pointer-events:none">
    <div class="phead"><span class="step b">2</span><h2>Attach LoRA and fine-tune</h2><span class="chip b">Style B: Shakespeare</span></div>
    <div class="row">
      <div class="fld"><label>Adapter</label>
        <select id="method"><option value="lora" selected>LoRA</option><option value="dora">DoRA (2024)</option></select></div>
      <div class="fld"><label>LoRA rank</label>
        <select id="rank"><option>2</option><option selected>4</option><option>8</option></select></div>
      <div class="fld"><label>Steps</label>
        <select id="ftSteps"><option>150</option><option selected>200</option><option>300</option></select></div>
      <button class="pb" id="ftBtn" onclick="finetune()">Fine-tune adapters</button>
    </div>
    <div class="status" id="ftStatus">Freezes the base, trains only the low-rank adapters on Style B.</div>
    <div id="paramStat" class="stat" style="margin-top:10px"></div>
    <div class="loss" id="ftLoss"></div>
  </div>

  <div class="panel" id="cmpPanel" style="opacity:.5;pointer-events:none">
    <div class="phead"><span class="step">3</span><h2>Compare base and adapted</h2></div>
    <div class="row">
      <div class="fld" style="flex:1"><label>Prompt</label><input type="text" id="prompt" value="the "></div>
      <div class="fld"><label>Temperature</label><input type="number" id="temp" value="0.8" step="0.1" min="0.1" max="2" style="width:110px"></div>
      <button class="pa" id="cmpBtn" onclick="compare()">Generate both</button>
    </div>
    <div class="cmp">
      <div class="card"><h3 style="color:var(--a)">Base (Style A)</h3><div class="out" id="outBase"><span class="muted">Not generated yet.</span></div></div>
      <div class="card"><h3 style="color:var(--b)">LoRA-adapted (Style B)</h3><div class="out" id="outTuned"><span class="muted">Not generated yet.</span></div></div>
    </div>
    <div class="stat" id="frozenCheck" style="margin-top:14px"></div>
  </div>

  <footer>
    The base transformer is Glassbox (built from scratch, gradient-checked). LoRA
    adds two thin matrices per attention head; only those train, the base is
    frozen. Tiny models for illustration: the mechanism, not scale.
  </footer>
</main>

<script>
let based = false, tuned = false;

const ICON = {
  check:'<svg class="i" width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 7.5 6 11l5.5-7.5"/></svg>',
  x:'<svg class="i" width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3.5 3.5l7 7m0-7l-7 7"/></svg>',
};

function bars(el, losses, color){
  const max = Math.max(...losses), min = Math.min(...losses), span=(max-min)||1;
  el.innerHTML=''; const stride=Math.max(1,Math.floor(losses.length/120));
  for(let i=0;i<losses.length;i+=stride){
    const v=losses[i], h=6+(1-(v-min)/span)*100;
    const b=document.createElement('div'); b.className='bar'; b.style.height=h+'px';
    b.style.background=color; b.title='step '+(i+1)+': '+v.toFixed(3); el.appendChild(b);
  }
}

function pretrain(){
  const btn=document.getElementById('preBtn'); btn.disabled=true;
  const st=document.getElementById('preStatus'); st.innerHTML='<span class="spinner"></span> starting...';
  const losses=[]; const p=new URLSearchParams({arch:document.getElementById('arch').value, steps:document.getElementById('preSteps').value});
  const es=new EventSource('/pretrain?'+p.toString());
  es.addEventListener('step',e=>{const d=JSON.parse(e.data); losses.push(d.loss);
    st.innerHTML='<span class="spinner"></span> step '+d.step+'/'+d.total+' · loss '+d.loss.toFixed(3);
    bars(document.getElementById('preLoss'),losses,'var(--a)');});
  es.addEventListener('done',e=>{es.close(); const d=JSON.parse(e.data);
    st.textContent='Base trained on Style A, loss '+d.first.toFixed(2)+' → '+d.last.toFixed(3);
    based=true; enable('ftPanel'); btn.disabled=false;});
  es.onerror=()=>{es.close(); st.textContent='error'; btn.disabled=false;};
}

function finetune(){
  if(!based) return;
  const btn=document.getElementById('ftBtn'); btn.disabled=true;
  const method=document.getElementById('method').value;
  const st=document.getElementById('ftStatus'); st.innerHTML='<span class="spinner"></span> attaching '+method.toUpperCase()+'...';
  const losses=[]; const p=new URLSearchParams({rank:document.getElementById('rank').value, steps:document.getElementById('ftSteps').value, method:method});
  const es=new EventSource('/finetune?'+p.toString());
  es.addEventListener('params',e=>{const d=JSON.parse(e.data);
    document.getElementById('paramStat').innerHTML='Training <b>'+d.trainable.toLocaleString()+
      '</b> of <b>'+d.total.toLocaleString()+'</b> params <span class="ok">('+d.pct+'% of the model)</span>';});
  es.addEventListener('step',e=>{const d=JSON.parse(e.data); losses.push(d.loss);
    st.innerHTML='<span class="spinner"></span> step '+d.step+'/'+d.total+' · loss '+d.loss.toFixed(3);
    bars(document.getElementById('ftLoss'),losses,'var(--b)');});
  es.addEventListener('done',e=>{es.close(); const d=JSON.parse(e.data);
    st.textContent='Adapters fine-tuned on Style B, loss '+d.first.toFixed(2)+' → '+d.last.toFixed(3);
    tuned=true; enable('cmpPanel'); btn.disabled=false;});
  es.onerror=()=>{es.close(); st.textContent='error'; btn.disabled=false;};
}

async function compare(){
  if(!tuned) return;
  const btn=document.getElementById('cmpBtn'); btn.disabled=true;
  document.getElementById('outBase').innerHTML='<span class="spinner"></span>';
  document.getElementById('outTuned').innerHTML='<span class="spinner"></span>';
  const body={prompt:document.getElementById('prompt').value, temperature:parseFloat(document.getElementById('temp').value)};
  try{
    const r=await fetch('/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    document.getElementById('outBase').textContent=d.base;
    document.getElementById('outTuned').textContent=d.tuned;
    const fc=document.getElementById('frozenCheck');
    const frozenMsg = d.base_frozen
      ? '<span class="ok">'+ICON.check+'base weights unchanged.</span> Only the adapters moved. That is LoRA.'
      : '<span class="no">'+ICON.x+'base changed (unexpected).</span>';
    const pplMsg = (d.ppl_tuned!=null)
      ? '<div style="margin-top:8px">perplexity on Style B: base <b>'+d.ppl_base+
        '</b> &#8594; adapted <b class="ok">'+d.ppl_tuned+'</b> (lower is better)</div>'
      : '';
    fc.innerHTML = frozenMsg + pplMsg;
  }catch(e){ document.getElementById('outBase').textContent='error: '+e; }
  btn.disabled=false;
}

function enable(id){const el=document.getElementById(id); el.style.opacity='1'; el.style.pointerEvents='auto';}
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8") if n else "{}"
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}

    def _event(self, name, data):
        try:
            self.wfile.write(f"event: {name}\n".encode())
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        p = urlparse(self.path)
        if p.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif p.path == "/pretrain":
            self._pretrain(parse_qs(p.query))
        elif p.path == "/finetune":
            self._finetune(parse_qs(p.query))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path == "/compare":
            self._compare()
        else:
            self._send(404, "not found", "text/plain")

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def _pretrain(self, q):
        arch = q.get("arch", ["gpt2"])[0]
        arch = arch if arch in ("gpt2", "llama") else "gpt2"
        steps = max(10, min(int(q.get("steps", ["200"])[0]), 400))
        self._sse_open()
        ds = _shared_dataset()
        model = GPT(vocab_size=ds.vocab_size, d_model=64, n_heads=4, n_layers=2,
                    block_size=32, seed=0, arch=arch)
        data = ds.encode(STYLE_A)
        rng = np.random.default_rng(0); opt = Adam(lr=3e-3)
        losses = []
        for step in range(1, steps + 1):
            x, y = _batch(data, ds.block_size, rng)
            _, loss = model.forward(x, y); model.backward()
            opt.step(model.params_and_grads()); losses.append(float(loss))
            if step == 1 or step % 3 == 0 or step == steps:
                self._event("step", {"step": step, "total": steps, "loss": float(loss)})
        with _LOCK:
            _STATE.update(model=model, dataset=ds, phase="base",
                          frozen_ref=model.blocks[0].attn.heads[0].Wq.copy())
        self._event("done", {"first": losses[0], "last": losses[-1]})

    def _finetune(self, q):
        rank = max(1, min(int(q.get("rank", ["4"])[0]), 16))
        steps = max(10, min(int(q.get("steps", ["200"])[0]), 400))
        method = q.get("method", ["lora"])[0]
        method = method if method in ("lora", "dora") else "lora"
        with _LOCK:
            model, ds = _STATE["model"], _STATE["dataset"]
        if model is None:
            self._sse_open(); self._event("done", {"first": 0, "last": 0, "error": "no base"}); return
        self._sse_open()
        apply_lora(model, rank=rank, alpha=2.0 * rank, method=method)
        trainable, total = count_params(model)
        self._event("params", {"trainable": trainable, "total": total,
                                "pct": round(100 * trainable / total, 1)})
        data = ds.encode(STYLE_B)
        rng = np.random.default_rng(1); opt = Adam(lr=5e-3)
        losses = []
        for step in range(1, steps + 1):
            x, y = _batch(data, ds.block_size, rng)
            _, loss = model.forward(x, y); model.backward()
            opt.step(lora_params_and_grads(model)); losses.append(float(loss))
            if step == 1 or step % 3 == 0 or step == steps:
                self._event("step", {"step": step, "total": steps, "loss": float(loss)})
        with _LOCK:
            _STATE["phase"] = "tuned"
        self._event("done", {"first": losses[0], "last": losses[-1]})

    def _compare(self):
        body = self._json_body()
        with _LOCK:
            model, ds, frozen_ref = _STATE["model"], _STATE["dataset"], _STATE["frozen_ref"]
        if model is None or _STATE["phase"] != "tuned":
            self._send(200, json.dumps({"error": "pretrain and fine-tune first"}), "application/json")
            return
        prompt = "".join(c for c in str(body.get("prompt", "the ")) if c in ds.stoi) or ds.itos[0]
        temp = max(0.1, min(float(body.get("temperature", 0.8)), 2.0))

        # tuned generation (adapters active)
        tuned_txt = generate(model, ds, prompt, 44, seed=1, temperature=temp)
        # base generation: zero the adapters temporarily so only frozen base speaks
        saved = _zero_adapters(model)
        base_txt = generate(model, ds, prompt, 44, seed=1, temperature=temp)
        _restore_adapters(model, saved)

        base_frozen = np.array_equal(frozen_ref, model.blocks[0].attn.heads[0].Wq)

        # honest metric: perplexity on Style B (the fine-tuning target) for the
        # adapted model vs the base (adapters zeroed). Lower = better; the
        # adapted model should score lower, quantifying that it really adapted.
        ppl_tuned = perplexity(model, ds, STYLE_B[:600])
        saved = _zero_adapters(model)
        ppl_base = perplexity(model, ds, STYLE_B[:600])
        _restore_adapters(model, saved)

        self._send(200, json.dumps({
            "base": base_txt, "tuned": tuned_txt,
            "base_frozen": bool(base_frozen),
            "ppl_base": round(ppl_base, 2), "ppl_tuned": round(ppl_tuned, 2),
        }), "application/json")

    def log_message(self, *a):
        pass


def _batch(data, blk, rng, bs=16):
    n = len(data) - blk - 1
    ix = rng.integers(0, n, size=bs)
    x = np.stack([data[i:i + blk] for i in ix])
    y = np.stack([data[i + 1:i + 1 + blk] for i in ix])
    return x, y


def _zero_adapters(model):
    """Temporarily neutralize the adapters so the model computes the pure frozen
    base output (the 'base' side of the comparison). For LoRA that means zeroing
    the B matrices (A@B -> 0). For DoRA we must ALSO reset each magnitude vector
    to the base weight's column norms, otherwise the trained magnitude would
    still perturb the output -- zeroing B alone isn't enough for DoRA."""
    from .dora import _col_norm
    from .adapt import DoRAHead
    saved = []
    for blk in model.blocks:
        for h in blk.attn.heads:
            rec = {"h": h, "Bq": h.Bq.copy(), "Bv": h.Bv.copy()}
            h.Bq = np.zeros_like(h.Bq)
            h.Bv = np.zeros_like(h.Bv)
            if isinstance(h, DoRAHead):              # magnitude vectors are DoRA-only
                rec["mq"] = h.mq.copy(); rec["mv"] = h.mv.copy()
                h.mq = _col_norm(h.Wq).astype(h.mq.dtype)
                h.mv = _col_norm(h.Wv).astype(h.mv.dtype)
            saved.append(rec)
    return saved


def _restore_adapters(model, saved):
    for rec in saved:
        h = rec["h"]
        h.Bq = rec["Bq"]
        h.Bv = rec["Bv"]
        if "mq" in rec:                              # DoRA head
            h.mq = rec["mq"]
            h.mv = rec["mv"]


def serve(port: int = 8000):
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"Forge dashboard at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Forge web dashboard")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    serve(args.port)
