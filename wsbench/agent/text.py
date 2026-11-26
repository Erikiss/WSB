"""Text that goes to the Claude Platform or to the page must be valid UTF-8 without control characters."""
import re

_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")      # everything below space except tab, line feed, carriage return


def clean(text):
    """Replace lone UTF-16 surrogates (a browser can send them as \\ud800, a repository can have file names that are
    not UTF-8) and drop control characters, so that the SDK can encode the text and the platform accepts it."""
    return _CONTROLS.sub("", text.encode("utf-8", "replace").decode("utf-8"))


def clean_deep(value):
    """`clean` applied to every string in a JSON-like structure (keys included)."""
    if isinstance(value, str):
        return clean(value)
    if isinstance(value, dict):
        return {clean(k) if isinstance(k, str) else k: clean_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_deep(v) for v in value]
    return value
