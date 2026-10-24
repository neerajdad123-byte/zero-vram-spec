"""Draft proposers that won held-out replay (docs/research/speed-findings.md).

Each implements the shared contract: `name`, `propose(tokens, max_k) -> list[int]`,
`observe(accepted)`. All work on token ids only, so any tokenizer works."""
from .combo import ComboProposer
from .context_ngram import ContextNgramProposer
from .line_start import LineStartProposer
from .ngram_model import NgramModelProposer
from .suffix_copy import SuffixCopyProposer

__all__ = ["ComboProposer", "ContextNgramProposer", "LineStartProposer", "NgramModelProposer", "SuffixCopyProposer"]
