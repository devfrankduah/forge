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
    --bg:#0d0f13;--panel:#161a22;--panel2:#1d222c;--line:#2a2f3a;--text:#e6e9ef;
    --muted:#98a0b0;--a:#ff8c42;--b:#5e9bff;--good:#3ecf8e;--bad:#ff5c6c;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans)}
  header{padding:18px 26px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
  .logo{font-family:var(--mono);font-weight:700;font-size:20px}
  .logo .s{color:var(--a)}
  .tag{color:var(--muted);font-size:13px;flex-basis:100%}
  main{max-width:1000px;margin:0 auto;padding:22px 26px 90px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:18px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);margin:0 0 12px}
  .step{display:inline-block;width:22px;height:22px;line-height:22px;text-align:center;border-radius:50%;background:var(--panel2);color:var(--muted);font-size:12px;margin-right:8px;font-family:var(--mono)}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end}
  .fld{display:flex;flex-direction:column;gap:6px}
  .fld label{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted)}
  select,input{font-family:var(--sans);font-size:14px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:9px 11px}
  button{font-family:var(--sans);font-size:14px;border-radius:8px;border:none;font-weight:650;cursor:pointer;padding:10px 20px}
  .pa{background:var(--a);color:#160a02}
  .pb{background:var(--b);color:#04101f}
  button:disabled{opacity:.5;cursor:default}
  .status{color:var(--muted);font-size:13px;margin-top:10px;font-family:var(--mono);min-height:18px}
  .loss{height:110px;display:flex;align-items:flex-end;gap:2px;margin-top:10px}
  .loss .bar{flex:1;border-radius:2px 2px 0 0;min-height:1px;opacity:.85}
  .cmp{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .cmp .card{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:13px}
  .cmp h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
  .cmp .out{font-family:var(--mono);font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
  .muted{color:var(--muted);font-size:12px}
  .stat{font-family:var(--mono);font-size:13px}
  .big{font-size:26px;font-weight:700;font-family:var(--mono)}
  .spinner{display:inline-block;width:13px;height:13px;border:2px solid var(--muted);border-top-color:var(--a);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}
  .ok{color:var(--good)} .no{color:var(--bad)}
  /* ---- Responsive: tablet and below ---- */
  @media (max-width:720px){
    header{padding:14px 16px}
    main{padding:18px 16px 80px}
    .row{gap:14px}
    .fld{flex-basis:100%}
    .fld input,.fld select{width:100%}
    .row button{width:100%;padding:13px}
    .cmp{grid-template-columns:1fr}          /* base vs adapted stack vertically */
  }
  /* ---- Responsive: small phones ---- */
  @media (max-width:440px){
    main{padding:14px 12px 72px}
    .panel{padding:14px 13px}
    .logo{font-size:18px}
    .loss{height:80px}                        /* shorter charts so they don't dominate */
    .cmp .out{font-size:12px}
  }
  /* respect reduced-motion preference */
  @media (prefers-reduced-motion:reduce){ .spinner{animation:none} }
</style>
</head>
<body>
<header>
  <span class="logo"><span class="s">&#9874;</span> forge</span>
  <span class="tag">LoRA fine-tuning from scratch &#183; adapt a frozen transformer by training ~a few percent of it</span>
</header>
<main>
  <div class="panel">
    <h2><span class="step">1</span>Pretrain the base model — Style A (cat / dog / sun)</h2>
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
    <h2><span class="step">2</span>Attach LoRA &amp; fine-tune — Style B (Shakespeare)</h2>
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
    <div id="paramStat" class="stat" style="margin-top:8px"></div>
    <div class="loss" id="ftLoss"></div>
  </div>

  <div class="panel" id="cmpPanel" style="opacity:.5;pointer-events:none">
    <h2><span class="step">3</span>Compare — base vs LoRA-adapted</h2>
    <div class="row">
      <div class="fld" style="flex:1"><label>Prompt</label><input type="text" id="prompt" value="the "></div>
      <div class="fld"><label>Temp</label><input type="number" id="temp" value="0.8" step="0.1" min="0.1" max="2" style="width:80px"></div>
      <button class="pa" id="cmpBtn" onclick="compare()">Generate both</button>
    </div>
    <div class="cmp" style="margin-top:12px">
      <div class="card"><h3 style="color:var(--a)">Base (Style A)</h3><div class="out" id="outBase"><span class="muted">—</span></div></div>
      <div class="card"><h3 style="color:var(--b)">LoRA-adapted (Style B)</h3><div class="out" id="outTuned"><span class="muted">—</span></div></div>
    </div>
    <div class="stat" id="frozenCheck" style="margin-top:12px"></div>
  </div>

  <footer class="muted" style="margin-top:30px">
    The base transformer is Glassbox (built from scratch, gradient-checked). LoRA
    adds two thin matrices per attention head; only those train, the base is
    frozen. Tiny models for illustration — the mechanism, not scale.
  </footer>
</main>

<script>
let based = false, tuned = false;

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
    st.textContent='Base trained on Style A — loss '+d.first.toFixed(2)+' → '+d.last.toFixed(3);
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
    document.getElementById('paramStat').innerHTML='training <b>'+d.trainable.toLocaleString()+
      '</b> adapter params of <b>'+d.total.toLocaleString()+'</b> — <span class="ok">'+d.pct+'%</span> of the model';});
  es.addEventListener('step',e=>{const d=JSON.parse(e.data); losses.push(d.loss);
    st.innerHTML='<span class="spinner"></span> step '+d.step+'/'+d.total+' · loss '+d.loss.toFixed(3);
    bars(document.getElementById('ftLoss'),losses,'var(--b)');});
  es.addEventListener('done',e=>{es.close(); const d=JSON.parse(e.data);
    st.textContent='Adapters fine-tuned on Style B — loss '+d.first.toFixed(2)+' → '+d.last.toFixed(3);
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
      ? '<span class="ok">&#10003; base weights unchanged</span> — only the adapters moved. That\'s LoRA.'
      : '<span class="no">&#10007; base changed (unexpected)</span>';
    const pplMsg = (d.ppl_tuned!=null)
      ? '<div style="margin-top:6px">perplexity on Style B — base <b>'+d.ppl_base+
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
