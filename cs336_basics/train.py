import os
import time
import collections
import regex as re
from typing import List
from itertools import repeat
import multiprocessing as mp
from typing import BinaryIO
from tqdm import tqdm
from functools import reduce
import logging
from cs336_basics.constants import GPT2_REGEX_PARSER
from cs336_basics.utils import update_sequence

logger = logging.getLogger(__name__)
# disable warning from tqdm:  DeprecationWarning: This process (pid=381428) is multi-threaded, use of fork() may lead to deadlocks in the child.
tqdm.monitor_interval = 0


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk_star(args):
    """Helper method for imap to take in list of argumnents"""
    return fn_process_chunk(*args)


def fn_process_chunk(
    filepath, special_tokens, regex_pattern, start, end
) -> collections.Counter:
    """Process chunk file[start] - file[end] into a frequency of array bytes count
    e.g {b'[h, e, l, l, o]: 1, b[w, o, r, l, d]: 2}

    Why? so we can speed up merge. In a naive impl, we construct an array.
    And words may repeat multiple times in a corpus, causing unnecessary scans.

    RETURN:
        sequences - a dictionary with key as a tuple of bytestring, value is number of occurrences in the corpus

    NOTES:

    """
    sequences = collections.Counter()
    special_token_regex = f"({'|'.join(re.escape(tok) for tok in special_tokens)})"

    with open(filepath, "rb") as file_io:
        file_io.seek(start)
        chunk = file_io.read(end - start).decode("utf-8", errors="ignore")
        split_chunks_by_speical_tokens = re.split(special_token_regex, chunk)
        for c in split_chunks_by_speical_tokens:
            # Construct sequence count
            if c in special_tokens:
                updated_token = (c.encode("utf-8"),)
                sequences[updated_token] = sequences.get(updated_token, 0) + 1
            else:
                # b'hello' ---> [b'h', b'e', b'l', b'l', b'o']
                for word in re.finditer(regex_pattern, string=c):
                    updated_token = tuple(
                        bytes([i]) for i in word.group().encode("utf-8")
                    )
                    sequences[updated_token] = sequences.get(updated_token, 0) + 1
    return sequences


def compute_pair_count(sequences):
    """Returns a dictionary of pair to its frequency appearing in sequences
    params:
        sequences: dict[tuple of bytestring, frequency]

    NOTE:
        - sequence may appear 1 or more time in a corpus
        - a pair may be overlapped in a given sequence (e.g. aaaaa)
    """
    pair_count = {}
    for bytestring_tuple, frequency in sequences:
        for pair in zip(bytestring_tuple, bytestring_tuple[1:]):
            if pair in pair_count:
                pair_count[pair] += frequency
            else:
                pair_count[pair] = frequency
    return pair_count


def compute_pair_to_sequence_idx(sequences):
    """A convenient reversed idx to find a list of sequences to be update after a merge"""
    reversed_idx = collections.defaultdict(set)  # set to dedup potential sequence
    for idx, (seq_key, _) in enumerate(sequences):
        for pair_tuple in zip(seq_key, seq_key[1:]):
            reversed_idx[pair_tuple].add(idx)
    return reversed_idx


def find_most_common_pairs(pair_count):
    """Return the most common pairs among all tokenized sequences.
    In case of a tie-breaking pair, pick the first one
    """
    # return max(pair_count, key=pair_count.get)
    # # pick one in lexicographically order.
    return max(pair_count, key=lambda p: (pair_count[p], p))


# update_sequence is imported from cs336_basics.utils


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    """"
    # key idea (not complete):
        Pre-tokenize corpus data into sequences, split by special regex (e.g. OpenAI)
        Init important data structures:
            - sequences: an ordered list sequences in corpus + its frequency
            - pair_count: frequency of  adjacent pair among sequences
            - pair_to_idx: reversed index of pair to a list of sequence indices where a pair is appeared
            - vocab: trained tokenizer vocabulary
            - merges: list of merges, ordered by creation
        Main loop (vocab_size - len(special_tokens))
            Find most common pair in the updated sequences.
            Record new merge and vocab
            Update step: (find all sequences needed to be update)
                Remove old pairs from pair_count and reverse index
                Rewrite sequence with the new pair (be careful of overlapping neighbors)
                Add new pair to pair_count and reverse_index
                Update sequence list with new seq
        Add special tokens to the vocab
    # note on utf-8:
        * for this algorithm, all operation should be in bytestring (sequence: a list of bytestring, pair_count a tuple of bytestring)
        * bytestring is an utf-8 encoded of a character. 
    """

    def pre_tokenize_corpus(
        input_file: str | os.PathLike, regex_pattern: str, special_tokens: List[str]
    ):
        """Transform corpus in to tokenized bytes-array based on a regex pattern
        Returns: an ordered list of [sequence: frequency]

        NOTE: by updating middle item of a list. use linked-link (which list may have already implemented in Python)
        """
        chunks = []
        logger.debug("Starting to pre-process corpus into sequences")
        with open(input_file, "rb") as file_io:
            chunk_size = 2 * mp.cpu_count() + 1
            chunks = find_chunk_boundaries(file_io, chunk_size, b"<|endoftext|>")
        logger.debug(
            f"Splitted corpus into {len(chunks) - 1} chunks using {mp.cpu_count()} cores"
        )

        tokenized_sequences = {}
        start = time.perf_counter()
        inputs = zip(
            repeat(input_file),
            repeat(special_tokens),
            repeat(regex_pattern),
            chunks[:-1],
            chunks[1:],
        )
        with mp.Pool(mp.cpu_count()) as p:
            results = list(
                tqdm(
                    p.imap(_process_chunk_star, inputs),
                    total=len(chunks) - 1,
                    desc="Tokenizing chunks",
                )
            )
            tokenized_sequences = dict(reduce(lambda d1, d2: d1 + d2, results))
        logger.debug(
            f"Processed {len(chunks) - 1} chunks into {len(tokenized_sequences)} sequences in {time.perf_counter() - start} seconds"
        )
        return list(tokenized_sequences.items())

    tokenized_sequences = pre_tokenize_corpus(
        input_path, GPT2_REGEX_PARSER, special_tokens
    )
    vocab: dict[int, bytes] = {}
    merges = []
    # use standard gpt2_bytes_to unicode instead
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pair_count = compute_pair_count(tokenized_sequences)
    pair_to_sequences_idx = compute_pair_to_sequence_idx(tokenized_sequences)

    num_merges = vocab_size - 256 - len(special_tokens)
    start = time.perf_counter()
    for i in range(num_merges):
        if i % 500 == 0:
            logger.debug(
                f"At merge {i}: Time elapsed: {time.perf_counter() - start:.3f} seconds"
            )

        if not pair_count:
            logger.warning(f"No more pairs to merge. Stopping early at {i} merges.")
            break

        best_pair = find_most_common_pairs(pair_count)
        merges.append(best_pair)
        vocab[len(vocab)] = bytes(best_pair[0] + best_pair[1])

        # .copy() because pair_to_sequences_idx
        seq_idx_to_be_updated = pair_to_sequences_idx[best_pair].copy()
        for seq_idx in seq_idx_to_be_updated:
            old_seq, frequency = tokenized_sequences[seq_idx]

            for old_pair in zip(old_seq, old_seq[1:]):
                pair_count[old_pair] -= frequency
                if pair_count[old_pair] == 0:
                    pair_count.pop(old_pair)
                pair_to_sequences_idx[old_pair].discard(seq_idx)

            new_seq = update_sequence(old_seq, best_pair)
            for new_pair in zip(new_seq, new_seq[1:]):
                if new_pair in pair_count:
                    pair_count[new_pair] += frequency
                else:
                    pair_count[new_pair] = frequency
                pair_to_sequences_idx[new_pair].add(seq_idx)

            tokenized_sequences[seq_idx] = (new_seq, frequency)

    # [i] because bytes only accepts a list / iterable.
    # If a number is passed, it will init an array of zero size i instead
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")

    return vocab, merges
