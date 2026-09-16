# Session log

Append-only, newest first — `info (YYYY-MM-DD)`. Undated entries elsewhere predate this log.

* `RopeLayer` done, `test_rope` passing, committed `completed rope` — AI-assisted; high-level idea understood (rotate each adjacent pair by position × per-pair frequency), tensor mechanics (`rearrange (pair xy)`, `unbind`, `stack`, gathered cos/sin broadcast) not yet owned; no session-start drill today; next: fix the string, run drill set v3 (RoPE + einops), then softmax → attention (2026-09-16)
* SiLU + `SwiGLULayer` done, `test_silu_matches_pytorch` and `test_swiglu` passing, committed `completed swiglu` (5/13 model tests green); best bug: SiLU written as `x / (1 + sigmoid(-x))` — confused `sigmoid(-x)` with `e^-x`; runner-up: einsum letter `o` tagged two different axes on the `w2` down-projection; drills: v1 shapes 6/6, works-or-error 11/12 (miss: elementwise `gate (1,9) * x (1,3)`); next: attention (lecture 04 unwatched) starting with softmax (2026-09-06)
* RMSNormLayer done, `test_rmsnorm` passing after a 3-step bug journey (global reduction → `mean(x)` instead of `sqrt(mean(x²))` → fixed); shape drills exposed dim/keepdim weakness — drill file at `../notes/concepts/tensor-ops.md`; next: SiLU + SwiGLU (2026-09-01)
* EmbeddingLayer done, `test_embedding` passing; best bug: einsum on a lookup — embedding is indexing, not contraction; next: Linear/RMSNorm (2026-08-30)
* BPE tokenizer + training-loop work (`bpe.py`, `train.py`, `train_tiny_stories.py`) — backfilled from git (2026-07-02)
* Ledger created; Unit 1 BPE pre-map + first intuitions — backfilled from git (2026-06-28)

# Unit 1: BPE Training , Tokenize

## BPE My Random Intutions
* Why do we have to stop at pair? can it be a triplet or qualet ? how would it affect the BPE training , encoding/decoding and inference speed?
* This is text embedding. How will it translate to other data sources like images, audios, genome sequencing, health signals?
* How to calculate compression ratio? (whole textt decode utf-8 / encoded BYE text ())

##  Pre-map 
* Objective: What must this implementation eventually achieve?
* Inputs/Outputs: What data flows through this module?
* Expected Invariants: What conditions must always remain true (e.g., deterministic tie-breaking, byte-level token reversibility)?
* Expected Hurdles: What do I anticipate will be the hardest part?

## 2. The Confusion Queue & Friction Log
* [ ] Active Question: (e.g., Why does ZeRO-2 shard optimizer states but not parameters?)
* Failed Mental Model / Best Bug:
  - What I assumed: 
  - What actually happened: 
  - Why my model was wrong: 
  - Corrected rule of thumb: 

## 3. The Feynman Compression
* Summarize the core concept in 10 lines or less as if teaching a peer.
* Key Architecture Table (For Scaling/Systems topics):


## 4. Cold Retrieval & Recall Questions (Fill at the very END of session)
* Closed-Book Concept Explanation:
* Closed-Book Pseudocode:
* Spaced Review Questions (Write 3-5 core Q/A pairs to quiz yourself later):
  - Q1: 
  - A1:


# Unit 2: Implement Transformer Model and Training Loop

### Random Thoughts:
* Implement Auto-Research based on new training loop
* LLM Surgery? can I split / combine a trained LLM to enhance its intelligence?
* LLM Needs? Go through the whole stack and identify potential business gaps
* Explain evolution of Transformer model: from original paper to latest ideas (reasoning about identified gaps + how people solved it)

### Pre-map
Goal:

Implement the core Transformer language model components and .
Expected learning experiences:
* understanding of decoder-only pre-norm transformer block and design choices:
* hyperparam tuning 
* resource accounting

Likely invariants:


Questions before studying (? from handnotes and assignment)
* How distribution of vocab is like?
* What is d_model
* What is vanishing gradient problem for deep architecture? 
* how GLU provide linear path for gradients while retaining non-linear capabilities? (paper-ref)
* How to maximize GPU utilization with torch ops
* Row  vs Column Vectors
* What is block-diagonal matrix
* How did position embeddings get lost during vanilla self-attention mechanism? (https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)


### Lecture inputs
Relevant lectures:
| Lecture    | Useful part                     | Why it matters                                    |
| ---------- | ------------------------------- | ------------------------------------------------- |
| Lecture 03 | Transformer architecture        | Overall model structure                           |
| Lecture 04 | Attention                       | Q/K/V, causal masking, multi-head attention       |
| Lecture 05 | Training / efficiency intuition | Helps reason about shape, memory, and performance |

Useful harvested ideas go here:
* The residual stream is the main state passed through the Transformer.
* Attention mixes information across sequence positions.
* The MLP transforms each token position independently.
* Causal masking enforces auto-progressive prediction.
* Shape invariants are the best debugging tool for Transformer implementation.
* Most Transformer bugs are dimension, masking, or residual-order bugs.
* Embedding layer = just a lookup table of learned vectors — the input stage of every LLM (2026-08-30)

### Confusion queue
* [ ] Distrbution of vocab. How's it related to top_p, top_K and other params in LLM
* [ ] GPU Utilization through built-in Torch ops for elements of batch, point-wise and attention headers
* [ ] Einsum notation — when is it the right tool vs. not? Know it's a
      contraction (sums over shared labels); embedding lookup turned out not
      to need it (indexing, not contraction) — what's the general rule for
      picking einsum vs. plain indexing/broadcast ops? (2026-08-30)
* [ ] Why pre-norm?
* [ ] RoPE mechanics, left as a code comment: what does
      `rearrange(x, "... seq (pair xy) -> ... seq pair xy", xy=2)` followed by
      `unbind(dim=-1)` actually do to the 64 numbers of one token? Is it the
      same as `x[..., 0::2]`, `x[..., 1::2]`? Would `(xy pair)` pair different
      elements? (2026-09-16)
* [ ] RoPE: why `register_buffer(..., persistent=False)` for cos/sin instead
      of a Parameter or recomputing per forward? What is saved in the state
      dict and what is not? (2026-09-16)
* [ ] RoPE: the gathered `cos` is `(..., seq, pair)` and `x0` is
      `(..., seq, pair)` — write the right-aligned columns. What if
      `token_positions` is `(seq,)` instead of `(batch, seq)`? (2026-09-16)
* [ ] einops `einsum`/`rearrange`/`reduce`: I can read them, not write them
      cold. When is `rearrange` a view vs a copy? (2026-09-16)
* [ ] Why does rotating q and k by position make `q·k` depend only on the
      *relative* position? Derive for one pair on paper. (2026-09-16)
* [ ] why SwiGLU over FNN + ReLU — refresher received 2026-09-06 (gate decouples
      "whether to pass" from "what to pass"; content branch stays linear so the
      gradient path is only scaled, never squashed; Shazeer 2020 offers no
      mechanistic explanation). Still to do: write the answer in my own words.
* [ ] einsum letter discipline — when is it safe to reuse `i`/`o` across lines
      of one forward? Rule for now: name axes by what they are (`m`, `f`) so a
      letter can never tag the wrong side of a stored `(out, in)` weight.
      (2026-09-06)
* [ ] Motivation behind Xavier weight initialization — why that particular
      scaling, and when it's the right init to reach for vs. alternatives
      (2026-08-30)
* [ ] What does a raw token ID *mean* at the embedding stage — how should I
      make sense of an integer ID before it's mapped to its learned vector?
      (2026-08-30)
* [ ] dim vs keepdim on reductions — got the rms shape wrong twice in drills
      (`(64,)`, then a scalar, before deriving `(4, 12, 1)`). Rule: reduction
      kills the named axis; keepdim leaves a size-1 stub for broadcasting.
      Practice via `../notes/concepts/tensor-ops.md` drills. (2026-09-01)
* [ ] nn.Parameter initialization patterns — why `torch.empty` + `copy_` in
      set_weights? When is `nn.init.*` the right tool? First RMSNorm draft
      carried a dead `self.weights` attribute. (2026-09-01)

### Test ideas

* [ ] attention probabilities sum to 1 over allowed keys
* [ ] model behaves deterministically in eval mode
* [ ] tiny overfit test on a small batch
* [ ] softmax rows sum to 1 along the chosen dim — `x.softmax(-1).sum(-1)` was
      all ones in the 2026-09-05 drill; reuse as the first attention probe.
      (2026-09-06)

### Implementation plan

1. Inspect assignment tests before writing code: 
    - linear module class (test_linear), 
    - embedding (test_embedding), 
    - rmsnorm (test_rmsnorm)
    - position-wise ffn (SwiGLU = Swish + Gated Linear Unit) - bias term (test_swiglu)
    - Relative Positional Embeddings
    - Scaled Dot-product attention:
        (softmax): this is home! :D unormed vector --> normalized distribution
        * masking 
        * scaled_dot_product_attention
    - Multi-Head Self Attention (OG idea)
        * Causal masking: prevent model from attending to future tokens (what if it can?)


2. Implement smallest module first.
6. Compose Transformer block.
7. Compose full model.
8. Run assignment tests.
9. Add tiny overfit experiment if useful.
10. Write retrieval summary.

### Design decisions (why not)

* RoPE precomputes `cos`/`sin` once in `__init__` as non-persistent buffers
  (2026-09-16, AI-assisted). Why: no per-forward trig, tables move with
  `.to(device)` like parameters but are not trained and not checkpointed.
  Why not compute per forward: cost every step for a table that depends only
  on `max_seq_len`, `d_k`, `theta`. Understood at the level of "what", not
  yet "why it must be a buffer" — see confusion queue.
* SwiGLU holds three raw `nn.Parameter`s via `_init_2d_weights` instead of
  three `LinearLayer` submodules (2026-09-06). Why: one shared init helper,
  flat state dict (`w1`, `w2`, `w3`). Why not submodules: they would have
  given the forward as three layer calls and reused `set_weights` (with its
  `no_grad`), which is exactly where two of today's bugs came from. Revisit
  when building the Transformer block — whichever layout the block uses for
  attention projections should win for consistency.

### Bugs / failed mental models

#### Bug: EmbeddingLayer.forward treated as matmul (2026-08-30)

What I assumed:
Embedding lookup is a matrix multiplication, same pattern as LinearLayer —
tried `torch.einsum("bs,sd->bd", x, embedding_matrix)`.

What happened:
`s` was reused as a label for two different axes (`x`'s sequence_length vs.
`embedding_matrix`'s vocab_size). Even where sizes coincidentally matched,
einsum contracts (sums over) any shared label, so the output silently
collapsed the sequence axis instead of preserving it — wrong shape and wrong
values, no error raised.

Corrected rule:
Embedding is a lookup (`table[idx]`), not a contraction. Not every op in an
LLM is matmul. Advanced/fancy indexing has its own shape rule:
`table[idx].shape == idx.shape + table.shape[1:]` — idx's entire shape
(any rank) is preserved and a new trailing axis (`d_model`) is appended. Every
scalar token ID in the N-D input box gets replaced in place by its vector.
Equivalence worth remembering: this is the same result you'd get by
one-hot-encoding each token ID and matmul-ing against the table — indexing is
just the fast path for that.

Test added: `test_embedding` (existing, now passing).

#### Bug: RMSNorm — three shape/semantics slips in one forward (2026-09-01)

What I assumed:
(1) `torch.sum(x**2)` would reduce per token. (2) After fixing the axes, that
`torch.mean(x, dim=-1, keepdim=True)` was the RMS statistic.

What happened:
(1) `sum` with no `dim` reduced over ALL axes → one scalar RMS shared by the
whole batch; output shape `(64,)` instead of `(4, 12, 64)`. (2) Dividing by
`mean(x)` — near zero for a centered vector — blew values up ~700×. A
misnamed variable (`x_normed` for what was really the rms) hid that the first
draft never divided `x` by anything at all.

Corrected rule:
Unpack the layer's name right-to-left: RMS = Root of Mean of Squares —
`sqrt(mean(x², dim=-1, keepdim=True) + eps)`, with eps INSIDE the sqrt.
The rms is per-token `(batch, seq, 1)`; the gain g is per-feature
`(d_model,)` — duals over the same grid. Name variables by what they hold,
not by what comes next.

Test: `test_rmsnorm` passing; invariant probe g=ones → per-token RMS ≈ 1.

#### Bug: SiLU as `x / (1 + sigmoid(-x))` (2026-09-06)

What I assumed:
`torch.sigmoid(-x)` is `e^-x`, so `x / (1 + sigmoid(-x))` is `x / (1 + e^-x)`.
The comment above the code also labeled the *sigmoid* formula "silu".

What happened:
All 10 elements mismatched by up to 0.06; the test compared against
`F.silu`. `sigmoid(-x)` is `1 / (1 + e^x)`, a number in (0, 1), not `e^-x`.

Corrected rule:
SiLU is `x * sigmoid(x)` — "x, gated by its own sigmoid". Before running,
check the tails: SiLU(large negative) ≈ 0, SiLU(large positive) ≈ x,
SiLU(0) = 0. The wrong version returns `x / 2` for large negative x. Write
the comment *after* the code passes, and make it say what the code does.

Test: `test_silu_matches_pytorch` passing.


### Retrieval Q&A (2026-08-30)

Q: Why can't `EmbeddingLayer.forward` be a `torch.matmul` of `x` against the
embedding table?
A: `x` holds raw integer token IDs, not one-hot vectors — there's no shared
dimension to contract against `vocab_size`. Matmul would require reusing an
axis label with a different meaning on each side and would sum away the
sequence axis instead of keeping one vector per token.

Q: If `idx` has shape `(batch, seq, k)` and `table` has shape
`(vocab_size, d_model)`, what is `table[idx].shape` and why?
A: `(batch, seq, k, d_model)`. Advanced indexing preserves `idx`'s full shape
as-is (nothing is replaced or collapsed) and appends `table.shape[1:]`.

### Retrieval Q&A (2026-09-01)

Q: In RMSNorm with input `(batch, seq, d_model)`, what are the shapes of the
rms statistic and the gain g — and what is each "one number per"?
A: rms is `(batch, seq, 1)` — one number per token, computed across its
features (`keepdim` leaves the stub for broadcasting the divide). g is
`(d_model,)` — one number per feature, shared by every token in every batch.

Q: `x` has shape `(4, 12, 64)`. Shapes of `x.sum(dim=0)` and
`x.sum(dim=0, keepdim=True)`?
A: `(12, 64)` and `(1, 12, 64)`. A reduction kills the axis you name and
leaves the rest; keepdim keeps a size-1 stub in its place.

Q: Where does eps sit in the RMSNorm formula, and why there?
A: Inside the sqrt — `sqrt(mean(x²) + eps)` — so the statistic can never be
zero before you divide by it.

### Retrieval Q&A (2026-09-06)

Q: Write SiLU in words and give the three tail checks.
A: `x` times `sigmoid(x)` — x gated by its own sigmoid. SiLU(0) = 0;
SiLU(x) ≈ x for large positive x; SiLU(x) ≈ 0 for large negative x (the
`sigmoid(-x)` bug returned `x / 2` there).


Q: For `x` of shape `(4, 12, 3)` and gate `g` of shape `(4, 12, 9)`, does
`g * x` work?
A: No. Right-align: 9 vs 3, neither is 1. Gate and content live in `d_ff`
space; `x` lives in `d_model` space. Elementwise ops need the same
right-alignment check as matmul, and "one number per what?" on both sides.

### Retrieval Q&A (2026-09-16)

Q: `x` is `(4, 12, 64)`. Shape after
`rearrange(x, "... seq (pair xy) -> ... seq pair xy", xy=2)`, and which two
elements of a token end up in pair 0?
A: `(4, 12, 32, 2)`. The last axis is split with `xy` fastest, so pair 0 holds
elements 0 and 1, pair 1 holds 2 and 3 — adjacent pairing. `(xy pair)` would
instead pair element 0 with element 32 (the halves convention).

Q: The RoPE angle table is `(max_seq_len, d_k/2)`. What is
`table[token_positions].shape` for `token_positions` of shape `(4, 12)`, and
why does it broadcast against `x0`?
A: `(4, 12, 32)` — the embedding lookup rule, `idx.shape + table.shape[1:]`.
`x0` is also `(4, 12, 32)` (one number per token per pair), so the elementwise
rotation lines up column for column with no stub needed.

Q: In one sentence, what does RoPE do to a query vector and why to keys too?
A: It rotates each adjacent coordinate pair by `position × frequency[pair]`;
applying the same rotation to keys makes `q·k` depend only on the distance
between the two positions, so attention scores carry relative position.

### Retrieval

Explain from memory:

Pseudocode from memory:

What I still cannot explain:
