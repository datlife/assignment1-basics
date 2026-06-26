import argparse
import json
import logging
import time

from cs336_basics.bpe import run_train_bpe
from tests.common import gpt2_bytes_to_unicode
logger = logging.getLogger(__name__)

gpt2_byte_decoder = gpt2_bytes_to_unicode()
def decode(bpe_bytes):
    """" Convert a byte sequence into string. Since bpe_bytes is not utf-8, we convert
    it back to printable string byte by byte
    """
    global gpt2_byte_decoder
    return "".join(gpt2_byte_decoder[b] for b in bpe_bytes)



def main():  
    parser =  argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)-6s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )  

    logging.info("Starting BPE training")
    start = time.perf_counter()
    vocab, merges = run_train_bpe(
        # input_path="./data/TinyStoriesV2-GPT4-train.txt",
        input_path="./tests/fixtures/tinystories_sample_5M.txt",
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    elapsed = time.perf_counter() - start

    # writing vocab merges to disk

    with open("./vocab.json", "w") as f:
       json.dump({decode(bytestring): k for k, bytestring in vocab.items()}, f, indent=4, ensure_ascii=False) 

    logger.info("Finished BPE training in %.2f seconds", elapsed)
    logger.info("Vocab size: %d", len(vocab))
    logger.info("Number of merges: %d", len(merges))

    ## compute compression stats

if __name__ == "__main__":
    main()