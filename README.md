# Transformer RNN GRU Comparison

### Experiments
For the experiments the following parameters were used:
- `MAX_SEQ_LEN = 60`
- `MIN_TOK_FREQ = 2`
- `epochs = 20`
- `batch_size = 128`
- `lr = 1e-3`
- `betas = (0.9, 0.99)`
- `clip = 5`
- `n_heads = 4`
- `dim_embedding = 196`
- `dim_hidden = 256`
- `dropout = 0.1`
__*Experiment 1*__
- `n_layers = 3`
__*Experiment 2*__
- `n_layers = 6`


### Translation Performance
| n_layers | Model      | top-10           | top-5            | top-1            | bleu              | ppl              |
|----------|------------|------------------|------------------|------------------|-------------------|------------------|
| 3        | GRU        | 0.934            | 0.908            | 0.750            | **0.2**               | **2.376**        |
|          | RNN        | 0.839            | 0.786            | 0.586            | 0.0               | 7.178            |
|          | Transformer| **0.958**        | **0.940**        | **0.795**        | 0.175             | 3.127            |
|----------|------------|------------------|------------------|------------------|-------------------|------------------|
| 6        | GRU        | 0.926            | 0.898            | 0.736            | 0.3               | 3.294            |
|          | RNN        | 0.810            | 0.752            | 0.554            | 0.1               | 8.749            |
|          | Transformer| **0.960**        | **0.942**        | **0.802**        | **0.375**         | **2.320**        |

- Overall Accuracy: Across top-10, top-5, and top-1 accuracy metrics, the 6-layer Transformer model achives the highest scores (0.960, 0.942, and 0.802, respectively).
- BLEU Score: The Transformer with 6 layers has the highest BLEU score of 0.375. This reflects its ability to generate translations closest to human reference translations.
- Perplexity: The 3-layer GRU model exhibits the lowest perplexity at 2.376, but the 6-layer Transformer follows closely with a perplexity of 2.320, balancing model complexity and predictive precision effectively.

- Layer Depth Impact: Increasing layers from 3 to 6 significantly improves the Transformer's performance. This is partly due to its self-attention mechanism which scales better with depth than RNNs. In contrast, GRU and Vanilla RNN struggle with additional layers due to issues like vanishing gradients, hindering their ability to scale with additional layers.
- Potential Impact of Sequence Length: Increasing the maximum sequence length (MAX_SEQ_LEN) could increase the performance gap between Transformer and the RNN-based models.
