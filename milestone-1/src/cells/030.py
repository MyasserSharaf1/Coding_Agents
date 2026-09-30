import random, os, time

T_SESSION_START = time.time()        # the session budget below is measured from here

# ---- Model -----------------------------------------------------------------
MODEL_NAME       = "qwen3-coder:30b"          # ~18 GB at Q4: needs GPU T4 x2
ALT_MODELS       = ["qwen2.5-coder:7b", "qwen2.5-coder:3b"]   # single-GPU fallbacks
SUPERVISOR_MODEL = None     # None = the code model reviews its own work (self-critique)

if SUPERVISOR_MODEL is not None and not isinstance(SUPERVISOR_MODEL, str):
    raise ValueError("SUPERVISOR_MODEL must be a model name string or None.")

# ---- Benchmarks ------------------------------------------------------------
M1_BENCHMARKS = [("mbpp", "plus"), ("humaneval", "plus"), ("livecodebench", "lite")]
# questions per benchmark; None = every question available
M1_N_PROBLEMS = {"mbpp_plus": 120,         # of 378 (seeded sample)
                 "humaneval_plus": None,   # all 164
                 "livecodebench": 100}     # of the 182 in the date window below (seeded sample)
M1_MIN_PROBLEMS = 100        # floor per benchmark (n >= 45 is the McNemar floor; 120 gives power)
ALLOW_SMALL_N   = False      # True only for a quick smoke test below the floor
SEED            = 42
MBPP_PLUS_USE_EXTENDED_TESTS = True   # grade MBPP+ with the EvalPlus suite, not 3 asserts

# LiveCodeBench (code_generation_lite). Problems released after the model's training
# cutoff are the contamination-safe ones; qwen3-coder was released in mid-2025.
LCB_REPO      = "livecodebench/code_generation_lite"
LCB_FILES     = None           # None = every test*.jsonl in the repo; or e.g. ["test5.jsonl", "test6.jsonl"]
LCB_MIN_DATE  = "2025-01-01"   # keep problems with contest_date >= this
LCB_TIMEOUT_S = 900            # safety cap for a whole suite; each stdin case is limited to 10 s,
                               # as in the official harness (a 60 s suite cap failed slow-but-correct programs)

# ---- Systems ---------------------------------------------------------------
# Every system shares the same first draft from MODEL_NAME. They differ in who writes the
# repair feedback (the code model reviewing itself, or a separate REVIEWER) and in the test agent.
#
# REVIEWER_BACKEND = "local": a second, smaller model runs on the same Kaggle GPUs. Free, no API
#                    key, no rate limits, never stops - and fully measured in GPU-s and joules.
# REVIEWER_BACKEND = "kimi" : Kimi K2.6 over an API (settings below). Stronger, but rate-limited,
#                    and its compute is NOT in GPU-s or joules.
REVIEWER_BACKEND = "local"
REVIEWER_MODEL   = "qwen2.5-coder:7b"   # ~4.7 GB at Q4; fits next to the 30B on 2x T4
REVIEWER_NAME    = REVIEWER_MODEL if REVIEWER_BACKEND == "local" else "Kimi K2.6"

SYSTEMS = ["baseline", "repair", "repair_rev", "repair_tests_rev"]
SYSTEM_LABEL = {"baseline": "baseline",
                "repair": "repair (self-review)",
                "repair_rev": "repair (%s review)" % REVIEWER_NAME,
                "repair_tests_rev": "repair (%s) + test agent" % REVIEWER_NAME}
SYSTEM_SHORT = {"baseline": "base", "repair": "rep", "repair_rev": "rep-R",
                "repair_tests_rev": "rep-R+T"}
SUPERVISOR_BY_SYSTEM = {"repair": "self", "repair_rev": "reviewer", "repair_tests_rev": "reviewer"}
TEST_AGENT_SYSTEMS   = {"repair_tests_rev": "repair_rev"}   # system -> same system without tests
COMPARISONS = [("baseline", "repair"), ("baseline", "repair_rev"),
               ("repair", "repair_rev"),                   # does a separate reviewer help?
               ("repair_rev", "repair_tests_rev")]         # does the test agent help?

# ---- Kimi reviewer (only used when REVIEWER_BACKEND = "kimi") -------------------
# Kimi K2.6 is a ~1T-parameter model; it cannot run on the T4s, so it is called over an
# OpenAI-compatible API. Put the key in Kaggle: Add-ons -> Secrets -> name "KIMI_API_KEY".
#   "openrouter" : moonshotai/kimi-k2.6:free is free but capped (20/min, 50/day; 1000/day
#                  once you have bought 10 credits). The paid id "moonshotai/kimi-k2.6" has no daily cap.
#   "moonshot"   : Moonshot's own API, model "kimi-k2.6" (paid)
#   "nvidia"     : build.nvidia.com trial credits, model "moonshotai/kimi-k2.6"
KIMI_PROVIDER     = "openrouter"
KIMI_MODEL        = None        # None = the provider's default above; e.g. "moonshotai/kimi-k2" for K2
KIMI_API_KEY_NAME = "KIMI_API_KEY"
KIMI_THINKING     = False       # short diagnoses; thinking adds tokens, latency and variance
KIMI_TEMPERATURE  = 0.6         # Moonshot's recommended value with thinking off
KIMI_MAX_TOKENS   = 600
KIMI_MIN_INTERVAL = 3.2         # seconds between API calls (free tier: 20 requests/minute)

# What the repair loop is allowed to see. Grading ALWAYS uses the hidden suite.
#   "public" : public tests only (MBPP's 3 asserts, HumanEval docstring examples, LCB public cases)
#   "hidden" : the grading suite itself - reproduces the first M1 run (optimistic upper bound)
FEEDBACK_TESTS = "public"

# ---- Repair loop -----------------------------------------------------------
TEMPERATURE               = 0.1
MAX_RETRIES               = 3       # max supervisor -> repair rounds per question
USE_SIGNATURE_HINT        = True    # show an example call so arity/order is not guessed
FEEDBACK_MODE             = "structured"   # "structured" = failing test + got/expected
EARLY_STOP_NO_PROGRESS    = True    # stop when a repair adds no passing visible tests
COMBINE_SUPERVISOR_REPAIR = False   # True = diagnosis folded into the repair call
TEST_AGENT_MAX_TESTS      = 5       # asserts kept from the test agent per question

# ---- Generation / execution ------------------------------------------------
NUM_PREDICT     = 512       # MBPP+ / HumanEval+
NUM_PREDICT_LCB = 2048      # LiveCodeBench programs are longer
NUM_CTX         = 8192      # one value for every benchmark: changing it forces a model reload
KEEP_ALIVE      = "24h"     # never unload the 30B mid-run: a reload lands in the GPU budget
TIMEOUT_SECONDS = 15        # MBPP+ / HumanEval+ suites
M1_TIMEOUT_S    = TIMEOUT_SECONDS

# ---- Checkpointing ---------------------------------------------------------
RESUME               = True   # skip questions already in outputs_milestone1/<bench>/rows.jsonl
SESSION_BUDGET_HOURS = 8.0    # stop starting new questions after this (Kaggle kills at ~9-12 h)

# ---- Output ----------------------------------------------------------------
ON_KAGGLE     = os.path.exists("/kaggle/working")
BASE_DIR      = "/kaggle/working" if ON_KAGGLE else "."
M1_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_milestone1")
OUTPUT_DIR    = M1_OUTPUT_DIR          # the Ollama log lands here too
os.makedirs(M1_OUTPUT_DIR, exist_ok=True)

random.seed(SEED)

EXPERIMENT_CONFIG = {
    "model_name": MODEL_NAME, "supervisor_model": SUPERVISOR_MODEL,
    "benchmarks": M1_BENCHMARKS, "n_problems": M1_N_PROBLEMS, "seed": SEED,
    "systems": SYSTEMS, "feedback_tests": FEEDBACK_TESTS,
    "supervisor_by_system": SUPERVISOR_BY_SYSTEM, "reviewer_backend": REVIEWER_BACKEND,
    "reviewer_model": REVIEWER_NAME, "test_agent_systems": TEST_AGENT_SYSTEMS,
    "kimi_provider": KIMI_PROVIDER if REVIEWER_BACKEND == "kimi" else None,
    "temperature": TEMPERATURE, "max_retries": MAX_RETRIES,
    "use_signature_hint": USE_SIGNATURE_HINT, "feedback_mode": FEEDBACK_MODE,
    "early_stop_no_progress": EARLY_STOP_NO_PROGRESS,
    "combine_supervisor_repair": COMBINE_SUPERVISOR_REPAIR,
    "test_agent_max_tests": TEST_AGENT_MAX_TESTS,
    "num_predict": NUM_PREDICT, "num_predict_lcb": NUM_PREDICT_LCB, "num_ctx": NUM_CTX,
    "timeout_seconds": TIMEOUT_SECONDS, "lcb_timeout_s": LCB_TIMEOUT_S,
    "lcb_min_date": LCB_MIN_DATE, "mbpp_plus_extended_tests": MBPP_PLUS_USE_EXTENDED_TESTS,
}
print("Results will be written to:", M1_OUTPUT_DIR)
for k, v in EXPERIMENT_CONFIG.items():
    print("  %-26s %s" % (k, v))
