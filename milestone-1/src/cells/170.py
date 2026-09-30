import requests, time, hashlib

class OllamaLLM:
    """Thin Ollama client with a prompt cache. The cache is what lets all three systems
    share ONE first draft per question (same prompt -> same program), and a fixed seed
    makes that draft reproducible. Metering (section 14) replays the cost of a cached
    answer, so every system is still charged for the draft it uses."""
    def __init__(self, model, temperature=0.1, host=OLLAMA_HOST, use_cache=True,
                 num_predict=NUM_PREDICT, num_ctx=NUM_CTX, keep_alive=KEEP_ALIVE, seed=SEED):
        self.model = model
        self.temperature = temperature
        self.host = host
        self.use_cache = use_cache
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.seed = seed
        self.cache = {}
        self.actual_calls = 0   # real network calls (cache misses)

    def options(self):
        return {"temperature": self.temperature, "num_predict": self.num_predict,
                "num_ctx": self.num_ctx, "seed": self.seed}

    def chat(self, system, user):
        key = hashlib.sha256(
            ("|".join([self.model, str(self.temperature), system, user])).encode()
        ).hexdigest()
        if self.use_cache and key in self.cache:
            return self.cache[key], 0.0, True
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        t0 = time.time()
        r = requests.post(self.host + "/api/chat", timeout=900, json={
            "model": self.model, "stream": False, "messages": msgs,
            "keep_alive": self.keep_alive, "options": self.options()})
        dt = time.time() - t0
        r.raise_for_status()
        content = r.json()["message"]["content"]
        self.actual_calls += 1
        if self.use_cache:
            self.cache[key] = content
        return content, dt, False

def extract_code(text):
    import re
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()

def extract_asserts(text, entry):
    import re
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    body = "\n".join(blocks) if blocks else text
    out = []
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("assert") and (entry is None or entry in s):
            try:
                compile(s, "<t>", "exec")      # drop half-written asserts
            except SyntaxError:
                continue
            out.append(s)
    return out
