class TestGenerationAgent:
    """Writes extra black-box asserts from the problem statement. They join the
    VISIBLE test set of the repair_tests system; they never touch grading."""
    SYS = ("You are a meticulous test engineer. Write black-box assert-based "
           "unit tests derived from the PROBLEM statement, not from any given "
           "implementation.")

    def __init__(self, llm):
        self.llm = llm

    def generate(self, text, entry, code, call_form=None, max_tests=5):
        call_form = call_form or ("%s(...)" % entry)
        prompt = ("Problem:\n%s\n\nFunction under test: `%s`, called as `%s`.\n"
                  "Write 3-5 additional `assert` statements (one per line, each on a "
                  "single line) that check correctness based on the problem, including "
                  "edge cases. Compute every expected value by hand from the problem "
                  "statement. Return them in one python code block."
                  % (text, entry, call_form))
        raw, dt, cached = self.llm.chat(self.SYS, prompt)
        tests = extract_asserts(raw, entry)
        return tests[:max_tests], dt
