"""SNav LLaVA-Video training package (paper SFT via train_mem)."""

try:
    from .model import LlavaLlamaForCausalLM  # noqa: F401
except Exception:  # pragma: no cover - optional until full train deps installed
    LlavaLlamaForCausalLM = None  # type: ignore
