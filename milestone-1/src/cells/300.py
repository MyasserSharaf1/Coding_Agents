import hashlib, requests, contextlib

CALL_LOG = []                 # one record per model call, in order
_CURRENT_ROLE = ["generator"]

@contextlib.contextmanager
def _role(name):
    _CURRENT_ROLE.append(name)
    try:
        yield
    finally:
        _CURRENT_ROLE.pop()


def _metered_chat(self, system, user):
    """Drop-in replacement for OllamaLLM.chat that records cost per call."""
    key = hashlib.sha256(
        ("|".join([self.model, str(self.temperature), system, user])).encode()
    ).hexdigest()
    role = _CURRENT_ROLE[-1]

    if not hasattr(self, "cost_cache"):
        self.cost_cache = {}

    # Cache hit: replay the ORIGINAL measured cost so a shared generation is still
    # charged to every system that consumes it. The record is flagged cached=True so
    # per-call analyses (section 22) can drop the duplicate.
    if self.use_cache and key in self.cache:
        prior = self.cost_cache.get(key)
        if prior is not None:
            rec = dict(prior)
            rec.update({"role": role, "cached": True})
            CALL_LOG.append(rec)
        return self.cache[key], 0.0, True

    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]

    tok = M1_METER.start()
    try:
        r = requests.post(self.host + "/api/chat", timeout=900, json={
            "model": self.model, "stream": False, "messages": msgs,
            "keep_alive": self.keep_alive, "options": self.options()})
        r.raise_for_status()
        data = r.json()
    finally:
        m = M1_METER.stop(tok)

    content = data["message"]["content"]
    self.actual_calls += 1

    total_ns = data.get("total_duration")
    load_ns = data.get("load_duration", 0) or 0
    # Exclude one-off model load: it is amortised across the run, not a per-task cost.
    compute_s = (max(total_ns / 1e9 - load_ns / 1e9, 0.0) if total_ns
                 else m["wall_seconds"])
    joules = m["joules"]
    # A reload inside the window would put load energy into this call's joules but not
    # into its GPU-seconds. Scale it out and flag the call.
    load_s = load_ns / 1e9
    if load_s > 0.5 and m["wall_seconds"] > 0 and joules == joules:
        joules = joules * max(m["wall_seconds"] - load_s, 0.0) / m["wall_seconds"]

    cost = {
        "role": role,
        "cached": False,
        "gpu_seconds": compute_s * M1_CONFIG["N_GPUS"],
        "wall_seconds": m["wall_seconds"],
        "joules": joules,
        # power over the SAME window the energy was measured in (all cards)
        "mean_power_w": (m["joules"] / m["wall_seconds"]) if m["wall_seconds"] > 0 else float("nan"),
        "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
        "completion_tokens": data.get("eval_count", 0) or 0,
        "load_seconds": load_s,
        "load_flag": load_s > 0.5,
    }
    if self.use_cache:
        self.cache[key] = content
        self.cost_cache[key] = cost
    CALL_LOG.append(dict(cost))
    return content, m["wall_seconds"], False


OllamaLLM.chat = _metered_chat
print("OllamaLLM.chat is now metered.")

# --- role tagging on the agent entry points ---------------------------------
def _tag_role(cls, method, role_name):
    orig = getattr(cls, method)
    if getattr(orig, "_m1_tagged", False):
        return
    def wrapper(self, *a, **kw):
        with _role(role_name):
            return orig(self, *a, **kw)
    wrapper._m1_tagged = True
    setattr(cls, method, wrapper)

_tag_role(CodeGenerationAgent, "generate", "generator")
_tag_role(CodeGenerationAgent, "repair",   "repair")
_tag_role(TestGenerationAgent, "generate", "test_writer")
_tag_role(SupervisorAgent,     "review",   "supervisor")
print("Role tagging installed: generator / repair / test_writer / supervisor")


# --- per-(question, system) cost --------------------------------------------
ROLE_KEYS = ["generator", "test_writer", "supervisor", "repair"]

def _cost_slice(i0):
    rows = CALL_LOG[i0:]
    out = {"gpu_seconds": 0.0, "wall_seconds": 0.0, "joules": 0.0,
           "prompt_tokens": 0, "completion_tokens": 0,
           "calls_total": 0, "calls_cached": 0, "joules_missing": 0,
           "remote_calls": 0, "remote_tokens": 0, "remote_seconds": 0.0}
    for k in ROLE_KEYS:
        out["calls_" + k] = 0
    for r in rows:
        out["calls_total"] += 1
        out["calls_" + r["role"]] = out.get("calls_" + r["role"], 0) + 1
        if r.get("cached"):
            out["calls_cached"] += 1
        if r.get("remote"):                 # API call: no local GPU time or energy
            out["remote_calls"] += 1
            out["remote_tokens"] += r.get("remote_prompt_tokens", 0) + r.get("remote_completion_tokens", 0)
            out["remote_seconds"] += r.get("remote_seconds", 0.0)
        out["gpu_seconds"] += r.get("gpu_seconds", 0.0)
        out["wall_seconds"] += r.get("wall_seconds", 0.0)
        out["prompt_tokens"] += r.get("prompt_tokens", 0)
        out["completion_tokens"] += r.get("completion_tokens", 0)
        j = r.get("joules", float("nan"))
        if j != j:
            out["joules_missing"] += 1
        else:
            out["joules"] += j
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    # IDLE_WATTS is summed across ALL cards and the joules were measured over the
    # calls' wall windows, so the idle share is idle W x wall seconds.
    idle = M1_CONFIG["IDLE_WATTS"]
    out["joules_above_idle"] = (max(out["joules"] - idle * out["wall_seconds"], 0.0)
                                if (idle == idle and not out["joules_missing"])
                                else float("nan"))
    if out["joules_missing"]:
        out["joules"] = float("nan")
    out["mean_power_w"] = (out["joules"] / out["wall_seconds"]
                           if out["wall_seconds"] > 0 else float("nan"))
    return out


def _tag_calls(i0, task_id, system):
    """Stamp each call made during one solve() with its question and system."""
    for r in CALL_LOG[i0:]:
        r["task_id"] = task_id
        r["system"] = system

print("Per-question cost attribution ready.")
