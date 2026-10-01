# --- Claude Code (optional) ---------------------------------------------------------------
# Installs the `claude` command in this Kaggle / Colab VM so you can ask it about the run, e.g.
#   !claude -p "Read outputs_m1_evalplus/summary.md and explain the test-agent regressions"
# It does not take part in the experiment and costs no GPU time. Off by default.
#
# Needs a Claude Pro, Max, Team, Enterprise or Console account (the free plan has no Claude Code).
# There is no browser on the VM, so log in with a token:
#   1. On your own computer (with Claude Code installed) run:  claude setup-token
#   2. Kaggle: Add-ons -> Secrets -> add CLAUDE_CODE_OAUTH_TOKEN (attach it to this notebook)
#      Colab : the key icon (Secrets) -> add CLAUDE_CODE_OAUTH_TOKEN, allow notebook access
#   Or, with API billing, a secret named ANTHROPIC_API_KEY instead.
# Colab has a terminal (run `claude` there for an interactive session); on Kaggle use `!claude -p "..."`.
INSTALL_CLAUDE_CODE = False

import os, subprocess


def _secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


if INSTALL_CLAUDE_CODE:
    subprocess.run("curl -fsSL https://claude.ai/install.sh | bash", shell=True, check=True)
    os.environ["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + os.environ.get("PATH", "")
    print(subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip())
    for _name in ["CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]:
        _val = _secret(_name)
        if _val:
            os.environ[_name] = _val          # never printed
            print("Claude Code will log in with the %s secret." % _name)
            break
    else:
        print("No CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY secret found - add one (see the notes above).")
else:
    print("Claude Code not installed (set INSTALL_CLAUDE_CODE = True to add it).")
