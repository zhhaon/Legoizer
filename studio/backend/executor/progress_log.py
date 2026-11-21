"""Turn carriage-return progress output into updates of the current log row."""
import re
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

class ProgressLog:
    def __init__(self, emit):
        self.emit = emit
        self.buffer = ""
        self.open_row = False

    def _publish(self):
        line = _ANSI.sub("", self.buffer).rstrip()
        self.buffer = ""
        # Preserve useful progress when tqdm clears a finished row with spaces.
        if line:
            self.emit(line, self.open_row)
            self.open_row = True

    def feed(self, text):
        for char in text:
            if char == "\r":
                self._publish()
            elif char == "\n":
                self._publish()
                self.open_row = False
            else:
                self.buffer += char

    def finish(self):
        self._publish()
