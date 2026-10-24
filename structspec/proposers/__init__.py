"""Draft proposers that won held-out replay (docs/research/speed-findings.md).

Each implements the shared contract: `name`, `propose(tokens, max_k) -> list[int]`,
`observe(accepted)`. All work on token ids only, so any tokenizer works."""
from .combo import ComboProposer
from .ngram_model import NgramModelProposer
from .suffix_copy import SuffixCopyProposer

__all__ = ["ComboProposer", "NgramModelProposer", "SuffixCopyProposer"]
