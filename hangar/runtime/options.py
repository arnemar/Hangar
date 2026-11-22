"""Optional overrides for Ollama generation defaults."""

from typing import NamedTuple


class Option(NamedTuple):
    label: str
    ollama_key: str
    minimum: int
    maximum: int
    whole: bool = False

    def accepts(self, value: object) -> bool:
        types = (int,) if self.whole else (int, float)
        return type(value) in types and self.minimum <= value <= self.maximum


GENERATION_OPTIONS = {
    "temperature": Option("Temperature", "temperature", 0, 2),
    "top_p": Option("Top P", "top_p", 0, 1),
    "max_output_tokens": Option("Max reply tokens", "num_predict", 1, 131072, whole=True),
}


def generation_options(settings: dict) -> dict[str, int | float]:
    return {
        option.ollama_key: settings[key]
        for key, option in GENERATION_OPTIONS.items()
        if option.accepts(settings.get(key))
    }
