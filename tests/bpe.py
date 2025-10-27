import os
import collections
import regex as re
from typing import List
from itertools import repeat
import multiprocessing as mp
from typing import BinaryIO
from functools import reduce

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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


def fn_process_chunk(filepath, special_tokens, regex_pattern, start, end) -> collections.Counter:
    """ Process chunk file[start] - file[end] into a frequency of array bytes count
    e.g {b'[h, e, l, l, o]: 1, b[w, o, r, l, d]: 2}

    Why? so we can speed up merge. In a naive impl, we construct an array.
    And words may repeat multiple times in a corpus, causing uncessary scans. 
    """
    res = collections.Counter()
    special_token_regex = f"({ '|'.join(re.escape(tok) for tok in special_tokens)})"

    with open(filepath, "rb") as file_io:
        file_io.seek(start)
        chunk = file_io.read(end - start).decode("utf-8", errors="ignore")
        split_chunks_by_tokens = re.split(special_token_regex, chunk)
        for c in split_chunks_by_tokens:
            if c in special_tokens:
                updated_token = (c.encode("utf-8"),)
                res[updated_token] = res.get(updated_token, 0) + 1
            else:
                for word in re.finditer(regex_pattern, string=c):
                    # b'hello' ---> [b'h', b'e', b'l', b'l', b'o']
                    updated_token = tuple(bytes([i]) for i in word.group().encode("utf-8"))
                    res[updated_token] = res.get(updated_token, 0) + 1
    return res

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
    OPENAI_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def pre_tokenize_corpus(input_file: str | os.PathLike, regex_pattern: str, special_tokens: List[str]):
        """Transform corpuse in to tokenized bytesarray based on a regex pattern (original paper using " " but it's limited).
        because it would separate case dog!, dog. and dog into 3 separate tokens (although it has the same semantic meaning)

        For example:
            Hello world --> [b'Hello', b' ', b'world'] --> [['h','e','l','l,'o'], [' ', 'w', 'o','r','l','d']]

        NOTE: combine special tokens into regex pattern to remove 
        """
        cpu_cores = mp.cpu_count()
        chunks = []
        with open(input_path, "rb") as file_io:
            chunk_size = 2 * cpu_cores + 1
            chunks = find_chunk_boundaries(file_io, chunk_size, b"<|endoftext|>")
        print(f"Splitted corpus into {len(chunks)} chunks")
    
        with mp.Pool(mp.cpu_count()) as p:
            results = p.starmap(fn_process_chunk, zip(repeat(input_path), repeat(special_tokens), repeat(regex_pattern), chunks[:-1], chunks[1:]))
            merged_dict = dict(reduce(lambda d1, d2: d1 + d2, results))
        return merged_dict

    vocabs = {i: bytes([i]) for i in range(256)}
    # tuple (bytestr, bytestr)
    merges = []

    # word tuple ---> frequency
    word_freqs = pre_tokenize_corpus(input_path, OPENAI_PAT, special_tokens)

    # counter of pair --> frequency
    pair_stats = collections.Counter()

    # pair ---> set of word tuple to be updated after each while loop. It's used
    # to update both pair_stats and word_freqs
    pair_to_words_index: dict = {}


    # build merge of all pairs one time
    # merge
    for token, freq in word_freqs.items():
        for pair in zip(token, token[1:]):
            pair_stats[pair] = pair_stats.get(pair, 0) + freq
            if pair not in pair_to_words_index:
                pair_to_words_index[pair] = set()
            else:
                pair_to_words_index[pair] |= set({token})

    while len(vocabs) < vocab_size - len(special_tokens):

        # Find pair with the most frequency
        best_pair, freq = max(pair_stats.items(), key=lambda x: (x[1], x[0]))
        merged_pair = best_pair[0] + best_pair[1]

        # incremental update word_stats and pair stats, and  pair_to_words_index with new pair
        # best_pair = ab
        # words_freqs = {[a,b,a]: 2, [a,b,a,b,c]: 3, [d,e]:1} ===> {[ab,a ]: 2, [ab, ab, c]:3}
        # pair_stats: swap all key with new pair
        #   old: [a,b]: 2+3*2=8 [b,a]: 5, [b,c] = 3 , [d,e]:1
        #   new: [ab]: 8, [b,a]:5 [b,c] 3 -- delete [ab]
        #   new: [b,a]: 5, [b,c]:3, [d,e]: 1
        # pair_to_words = 
        #   old:  
        #      [a,b]: set({aba, ababc})
        #      [b,a]: set({aba, ababc})
        #      [b,c]: set({ababc})
        #      [d,e]: set({de})
        #   new:
        #     delete pair_to_words[a,b]
        #     add [ab,a] : set({aba, ababc})
        #      [ab,a] : set({aba, ababc})
        #      [b,a]: set({aba, ababc})
        #      [b,c]: set({ababc})
        #      [d,e]: set({de})
        # Implementation strategy: Decrement/INcrement
        # Decrement all keys related to old_token and best_pair in word_freqs, pair_stats
        tokens_to_be_updated = pair_to_words_index.pop(best_pair)
        del pair_stats[best_pair]
        for old_token in tokens_to_be_updated:
            
            print(f"Is old_token in word_freqs: {True if old_token in word_freqs else False}")
            # update new word into word stats
            i, new_token = 0, []
            while i < len(old_token):
                if i < len(old_token) - 1 and old_token[i] == best_pair[0] and old_token[i+1] == best_pair[1]:
                    new_token.append(merged_pair)
                    i+=2
                else:
                    new_token.append(old_token[i])
                    i+=1
            new_token = tuple(new_token)
            print(f"New token: {new_token}\nold token: {old_token}\n=====")
            word_freqs[new_token] = word_freqs[old_token]

            # update pair stats
            # before : a b c d e e  new_pair = d e
            # after  : a b c (d e), e 
            for new_pair in zip(new_token, new_token[1:]):
                pair_stats[new_pair] = pair_stats.get(new_pair, 0) + freq
                if new_pair not in pair_to_words_index:
                    pair_to_words_index[new_pair] = set()
                else:
                    pair_to_words_index[new_pair] |= set({new_token})
            del word_freqs[old_token]


        # add to final result
        vocabs[len(vocabs)] = merged_pair
        merges.append(best_pair)

    for i in range(len(special_tokens)):
        vocabs[len(vocabs) + i] = special_tokens[i].encode("utf-8")

    return vocabs, merges
