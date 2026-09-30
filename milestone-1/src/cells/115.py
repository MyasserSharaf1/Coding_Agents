# --- The local reviewer model (skipped when REVIEWER_BACKEND = "kimi") ---------------
import requests
if REVIEWER_BACKEND == "local" and "reviewer" in SUPERVISOR_BY_SYSTEM.values():
    if not ensure_model(REVIEWER_MODEL):
        raise RuntimeError("Reviewer model %r unavailable - pick another REVIEWER_MODEL." % REVIEWER_MODEL)
    r_resp, r_dt = _quick_probe(REVIEWER_MODEL)
    print("Reviewer %s responded in %.1fs -> %r" % (REVIEWER_MODEL, r_dt, r_resp[:40]))
    # Both models must stay resident together, or every review would reload a model and the
    # load time would land inside the measured calls.
    loaded = requests.get(OLLAMA_HOST + "/api/ps", timeout=10).json().get("models", [])
    names = [m.get("name") for m in loaded]
    print("Resident in VRAM now:", ", ".join("%s (%.1f GB)" % (m.get("name"), m.get("size_vram", 0) / 1e9)
                                              for m in loaded))
    if not (any(MODEL_NAME in n for n in names) and any(REVIEWER_MODEL in n for n in names)):
        _quick_probe(MODEL_NAME)
        names = [m.get("name") for m in requests.get(OLLAMA_HOST + "/api/ps", timeout=10).json().get("models", [])]
    if any(MODEL_NAME in n for n in names) and any(REVIEWER_MODEL in n for n in names):
        print("OK: both models resident - reviews will not trigger reloads.")
    else:
        print("WARNING: only one model fits in VRAM, so each review would reload a model.\n"
              "  Use a smaller REVIEWER_MODEL (e.g. 'qwen2.5-coder:3b') or restart the kernel so\n"
              "  OLLAMA_MAX_LOADED_MODELS=2 is applied when the server starts.")
else:
    print("No local reviewer needed.")
