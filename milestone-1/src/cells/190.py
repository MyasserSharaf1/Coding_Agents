class CodeGenerationAgent:
    SYS = ("You are an expert Python programmer. Write correct, self-contained "
           "Python. Return ONLY one ```python code block with the required "
           "code (plus any imports it needs). No prose.")

    def __init__(self, llm):
        self.llm = llm

    def generate(self, text, entry, hint=None, instruction=None):
        extra = ("\n%s\nMatch that call signature exactly.\n" % hint) if hint else ""
        task = instruction or ("Implement the function named `%s`.\n"
                               "Return only the function definition (and needed imports) "
                               "in one python code block." % entry)
        prompt = "Problem:\n%s\n%s\n%s" % (text, extra, task)
        raw, dt, cached = self.llm.chat(self.SYS, prompt)
        return extract_code(raw), dt

    def repair(self, text, entry, prev_code, feedback, hint=None, instruction=None):
        task = instruction or ("Return a corrected, complete implementation of `%s` in one "
                               "python code block. Fix the specific issue; keep the same "
                               "function name and signature." % entry)
        prompt = ("Problem:\n%s\n\nYour previous solution FAILED its tests.\n\n"
                  "Previous code:\n```python\n%s\n```\n\n"
                  "Reviewer feedback:\n%s\n\n%s%s"
                  % (text, prev_code, feedback, task, ("\n" + hint) if hint else ""))
        raw, dt, cached = self.llm.chat(self.SYS, prompt)
        return extract_code(raw), dt
