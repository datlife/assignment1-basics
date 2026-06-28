from functools import lru_cache

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def update_sequence(seq: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """ Update a sequence with a new pair, keeping overlapping neighbors (e.g. aaaa) in mind.
    """
    i, j = 0, 1
    new_seq = []
    while j < len(seq):
        if (seq[i], seq[j]) == best_pair:
            new_seq.append(seq[i] + seq[j])
            i += 2 
            j += 2
        else:
            new_seq.append(seq[i])
            i += 1
            j += 1

    # edge case
    if i == len(seq) - 1:
        new_seq.append(seq[i])

    return tuple(new_seq)

def compute_compression_ratio(original_text: str, encoded_ids: list[int]) -> float:
    if not encoded_ids:
        return 0.0
    return len(original_text.encode('utf-8')) / len(encoded_ids)