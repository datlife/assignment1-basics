from collections.abc import Iterable
from typing import Generator
import regex as re
import json
from cs336_basics.constants import GPT2_REGEX_PARSER
from cs336_basics.utils import gpt2_bytes_to_unicode, update_sequence


class BPETokenizer:
    """Object-oriented implementation of a Byte-Pair Encoding Tokenizer using GPT-2 Regex"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None,
    ):
        """
        Initialize the BPE tokenizer with vocabulary mappings, BPE merge rules, and special tokens.
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = set(special_tokens) if special_tokens else set()

        self.encoder = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        # Ensure special tokens are in vocab and encoder
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded = special_token.encode("utf-8")
                if byte_encoded not in self.encoder:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = byte_encoded
                    self.encoder[byte_encoded] = new_id

        self.regex = re.compile(GPT2_REGEX_PARSER)

        if special_tokens:
            sorted_special = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(t) for t in sorted_special) + ")"
            )
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Load a trained tokenizer from vocabulary and merges files.
        """
        # Map printable unicode to bytes
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            gpt2_vocab = json.load(f)

        vocab = {
            vocab_idx: bytes([gpt2_byte_decoder[token] for token in token_str])
            for token_str, vocab_idx in gpt2_vocab.items()
        }

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned_line = line.rstrip()
                parts = cleaned_line.split(" ")
                if cleaned_line and len(parts) == 2:
                    merges.append(
                        (
                            bytes([gpt2_byte_decoder[char] for char in parts[0]]),
                            bytes([gpt2_byte_decoder[char] for char in parts[1]]),
                        )
                    )
        return cls(vocab, merges, special_tokens)

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        """
        Encode a single regex match segment (word/punctuation/space) by applying merge rules.
        """
        if not word_bytes:
            return []
        parts = tuple(bytes([b]) for b in word_bytes)
        while len(parts) > 1:
            best_pair = None
            best_rank = float("inf")
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            parts = update_sequence(parts, best_pair)

        return [self.encoder[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        """
        Encode raw text into a sequence of token IDs, handling special tokens and regex splitting.
        """
        if not text:
            return []

        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.encoder[part.encode("utf-8")])
            else:
                for match in self.regex.finditer(part):
                    word_bytes = match.group().encode("utf-8")
                    ids.extend(self._encode_word(word_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Generator[int, None, None]:
        """
        Lazily tokenize text chunks from an iterable, yielding token IDs one by one to save memory.
        """
        for text_chunk in iterable:
            if not text_chunk:
                continue
            if self.special_pattern:
                parts = self.special_pattern.split(text_chunk)
            else:
                parts = [text_chunk]

            for part in parts:
                if not part:
                    continue
                if part in self.special_tokens:
                    yield self.encoder[part.encode("utf-8")]
                else:
                    for match in self.regex.finditer(part):
                        word_bytes = match.group().encode("utf-8")
                        for token_id in self._encode_word(word_bytes):
                            yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a Unicode string.

        NOTE: we set errors=replace so that if any consequtive byte_data produce a malformed UTF-8 string. It'd still work
        """
        if not ids:
            return ""
        byte_data = b"".join(self.vocab[i] for i in ids)
        return byte_data.decode("utf-8", errors="replace")
