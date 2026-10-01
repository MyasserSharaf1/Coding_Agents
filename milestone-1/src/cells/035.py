# --- Resume across Kaggle sessions -------------------------------------------------------
# A new Kaggle session starts with an EMPTY /kaggle/working, so results from an unfinished run
# are only kept if you attach them: right sidebar -> Add Input -> Your Work -> Notebooks ->
# this notebook (the version that stopped). This cell finds <benchmark>/rows.jsonl anywhere under
# /kaggle/input and copies the most complete set into this run's output folder, so the run cell
# skips the questions that are already done. Old runs whose settings differ are ignored.
import glob, shutil, json

_MUST_MATCH = ["systems", "feedback_tests", "reviewer_model", "max_retries", "seed", "temperature",
               "early_stop_no_progress", "test_agent_max_tests", "num_predict", "num_predict_lcb",
               "num_ctx", "lcb_timeout_s", "lcb_min_date", "model_name", "mbpp_plus_extended_tests"]


def _n_lines(path):
    try:
        with open(path) as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def restore_from_inputs(root="/kaggle/input"):
    if not (RESUME and RESUME_FROM_INPUT and os.path.isdir(root)):
        return
    for b, v in M1_BENCHMARKS:
        tag = "livecodebench" if b == "livecodebench" else "%s_%s" % (b, v)
        local_dir = os.path.join(M1_OUTPUT_DIR, tag)
        best, best_n = None, _n_lines(os.path.join(local_dir, "rows.jsonl"))
        for rf in glob.glob(os.path.join(root, "**", tag, "rows.jsonl"), recursive=True):
            d = os.path.dirname(rf)
            try:
                old = json.load(open(os.path.join(d, "m1_config.json")))
            except Exception:
                print("  skip %s: no m1_config.json, cannot check its settings" % d)
                continue
            diff = [k for k in _MUST_MATCH if json.dumps(old.get(k), default=str)
                    != json.dumps(EXPERIMENT_CONFIG.get(k), default=str)]
            if diff:
                print("  skip %s: different settings (%s)" % (d, ", ".join(diff)))
                continue
            n = _n_lines(rf)
            if n > best_n:
                best, best_n = d, n
        if best:
            os.makedirs(local_dir, exist_ok=True)
            for fn in ["rows.jsonl", "calls.jsonl", "traces.jsonl"]:
                if os.path.exists(os.path.join(best, fn)):
                    shutil.copy2(os.path.join(best, fn), os.path.join(local_dir, fn))
            print("  %s: restored %d result rows from %s" % (tag, best_n, best))
        else:
            print("  %s: nothing to restore (%d rows already here)" % (tag, best_n))


restore_from_inputs(globals().get("KAGGLE_INPUT_DIR", "/kaggle/input"))
