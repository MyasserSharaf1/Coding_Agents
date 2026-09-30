import os, json, time, hashlib, requests

KIMI_PRESETS = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "moonshotai/kimi-k2.6:free"},
    "moonshot":   {"base_url": "https://api.moonshot.ai/v1", "model": "kimi-k2.6"},
    "nvidia":     {"base_url": "https://integrate.api.nvidia.com/v1", "model": "moonshotai/kimi-k2.6"},
}


class RemoteQuotaError(RuntimeError):
    """The API keeps refusing (daily cap or sustained rate limit). The run stops cleanly
    and resumes later instead of skipping every remaining question."""


def get_secret(name):
    v = os.environ.get(name)
    if v:
        return v
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


class RemoteLLM:
    """OpenAI-compatible chat client with the same chat() interface as OllamaLLM.

    Remote calls are metered as calls, remote tokens and remote seconds. They use no local
    GPU, so they add nothing to GPU-seconds or joules - every table and chart says so.
    Answers are cached on disk, so a resumed session or a second system with the same
    prompt never pays for the same diagnosis twice."""

    def __init__(self, provider=KIMI_PROVIDER, model=KIMI_MODEL, api_key=None,
                 temperature=KIMI_TEMPERATURE, max_tokens=KIMI_MAX_TOKENS,
                 thinking=KIMI_THINKING, min_interval=KIMI_MIN_INTERVAL, cache_path=None):
        preset = KIMI_PRESETS[provider]
        self.provider, self.base_url = provider, preset["base_url"]
        self.model = model or preset["model"]
        self.api_key = api_key or get_secret(KIMI_API_KEY_NAME)
        self.temperature, self.max_tokens, self.thinking = temperature, max_tokens, thinking
        self.min_interval, self._last = min_interval, 0.0
        self.use_cache, self.actual_calls = True, 0
        self.cache_path = cache_path or os.path.join(M1_OUTPUT_DIR, "kimi_cache.jsonl")
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line); self.cache[r["key"]] = r
                    except Exception:
                        pass
        if not self.api_key:
            raise RuntimeError("No API key: add a Kaggle secret (or env var) named %r." % KIMI_API_KEY_NAME)

    def _payload(self, msgs):
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if self.provider == "moonshot":
            body["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        elif self.provider == "openrouter":
            body["reasoning"] = {"enabled": bool(self.thinking)}
        return body

    def _post(self, msgs):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        headers = {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}
        last = None
        for attempt in range(6):
            self._last = time.time()
            try:
                r = requests.post(self.base_url + "/chat/completions", headers=headers,
                                  json=self._payload(msgs), timeout=180)
            except requests.exceptions.RequestException as e:
                last = str(e); time.sleep(min(60, 5 * 2 ** attempt)); continue
            if r.status_code == 429 or r.status_code >= 500:
                last = "%d %s" % (r.status_code, r.text[:300])
                if r.status_code == 429 and ("per day" in r.text.lower() or "daily" in r.text.lower()):
                    raise RemoteQuotaError("Daily request cap reached: " + last)
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if r.status_code in (401, 402, 403):
                raise RemoteQuotaError("API refused the key or has no credit: %d %s"
                                       % (r.status_code, r.text[:300]))
            r.raise_for_status()
            return r.json()
        raise RemoteQuotaError("Still failing after retries: %s" % last)

    def chat(self, system, user):
        key = hashlib.sha256("|".join([self.model, str(self.temperature),
                                       system, user]).encode()).hexdigest()
        role = globals().get("_CURRENT_ROLE", ["supervisor"])[-1]
        if key in self.cache:
            rec = dict(self.cache[key]["cost"]); rec.update({"role": role, "cached": True})
            if "CALL_LOG" in globals():
                CALL_LOG.append(rec)
            return self.cache[key]["content"], 0.0, True
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        t0 = time.time()
        data = self._post(msgs)
        dt = time.time() - t0
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning") or "").strip()
        usage = data.get("usage") or {}
        self.actual_calls += 1
        cost = {"role": role, "cached": False, "remote": True, "remote_model": self.model,
                "gpu_seconds": 0.0, "wall_seconds": 0.0, "joules": 0.0,
                "mean_power_w": float("nan"), "prompt_tokens": 0, "completion_tokens": 0,
                "remote_prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "remote_completion_tokens": int(usage.get("completion_tokens") or 0),
                "remote_seconds": dt, "load_seconds": 0.0, "load_flag": False}
        self.cache[key] = {"key": key, "content": content, "cost": cost}
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(self.cache[key]) + "\n")
        if "CALL_LOG" in globals():
            CALL_LOG.append(dict(cost))
        return content, dt, False


KIMI_LLM = None
if REVIEWER_BACKEND == "kimi" and "reviewer" in SUPERVISOR_BY_SYSTEM.values():
    KIMI_LLM = RemoteLLM()
    t0 = time.time()
    _probe = KIMI_LLM._post([{"role": "user", "content": "Reply with the single word: OK"}])
    print("Kimi reachable via %s (%s) in %.1fs -> %r"
          % (KIMI_LLM.provider, KIMI_LLM.model, time.time() - t0,
             (_probe["choices"][0]["message"].get("content") or "")[:40]))
    print("Diagnoses already cached from earlier sessions:", len(KIMI_LLM.cache))
else:
    print("Reviewer runs locally (%s) - no API used." % REVIEWER_NAME)
