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
* Implement Autoresearch based on new training loop
* LLM Surgergy? can I split / combine a trained LLM to enhance its intelligence?
* LLM Needs? Go through the whole stack and identify potential business gaps
* Explain evolution of Transformer model: from original paper to latest ideas (reasoning about identified gaps + how people solved it)

### Pre-map
Goal:

Implement the core Transformer language model components and .
Expected learning experiences:
* understanding of decoder-only pre-norm transformer block and design choices:
* hyperparam tuning 
* resource accounting
 * 

Likely invariants:


Questions before studying (? from handnotes and assignment)
* How distribution of vocab is like?
* What is d_model
* What is vanishing gradient problem for deep architecture? how GLU provide linear path for gradients while retaining non-linear capabilities? (paper-ref)
* How to maximize GPU utilization with torch ops
* Row  vs Column Vectors
* What is block-diagnomnal matrix
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
* Causal masking enforces autoregressive prediction.
* Shape invariants are the best debugging tool for Transformer implementation.
* Most Transformer bugs are dimension, masking, or residual-order bugs.

### Confusion queue
* [ ] Distrbution of vocab. How's it related to top_p, top_K and other params in LLM
* [ ] GPU Utilization through built-in Torch ops for elements of batch, point-wise and attention headers
* [ ] Einsum notation
* [ ] Why pre-norm?
* [ ] why SwiGLU over FNN + ReLU

### Test ideas

* [ ] attention probabilities sum to 1 over allowed keys
* [ ] model behaves deterministically in eval mode
* [ ] tiny overfit test on a small batch

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
3. Add shape tests after each module.
4. Implement attention without optimization first.
5. Add causal mask test.
6. Compose Transformer block.
7. Compose full model.
8. Run assignment tests.
9. Add tiny overfit experiment if useful.
10. Write retrieval summary.

### Bugs / failed mental models

#### Bug: <short name>

What I assumed:

What happened:

Corrected rule:

Test added:

### Retrieval

Explain from memory:

Pseudocode from memory:

What I still cannot explain:
