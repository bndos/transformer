"""## Transformer models.

Here you have to code the Full Transformer and Decoder-Only Transformer
architectures. It is divided in three parts:
* Attention layers (done individually)
* Encoder and decoder layers (done individually)
* Full Transformer: gather the encoder and decoder layers (done individually)

The Transformer (or "Full Transformer") is presented in the paper:
[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf). The
[illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
blog can help understand how the architecture works.
Also [the annontated transformer]
(https://nlp.seas.harvard.edu/2018/04/03/attention.html) to have an idea of
how to code this architecture. We encourage you to use `torch.einsum` and the
`einops` library as much as you can. It will make your code simpler.
"""

import torch
from torch import nn
import einops
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Positional encoding module.

    This PE module comes from:
    Pytorch. (2021). LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT.
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """Create a positional encoding module."""
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).to(DEVICE)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        ).to(DEVICE)
        pe = torch.zeros(max_len, 1, d_model).to(DEVICE)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward pass of the positional encoding module.

        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = einops.rearrange(x, "b s e -> s b e")
        x = x + self.pe[: x.size(0)]
        x = einops.rearrange(x, "s b e -> b s e")
        return self.dropout(x)


def attention(
    q: torch.FloatTensor,
    k: torch.FloatTensor,
    v: torch.FloatTensor,
    mask: torch.BoolTensor = None,
    dropout: nn.Dropout = None,
) -> tuple:
    """Compute multihead scaled dot-product attention.

    From theprojected queries, keys and values.

    Args
    ----
        q: Batch of queries.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        k: Batch of keys.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        v: Batch of values.
            Shape of [batch_size, seq_len_2, n_heads, dim_model].
        mask: Prevent tokens to attend to some other tokens (for
            padding or autoregressive attention).
            Attention is prevented where the mask is `True`.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2],
            or broadcastable to that shape.
        dropout: Dropout layer to use.

    Output
    ------
        y: Multihead scaled dot-attention between the queries, keys and values.
            Shape of [batch_size, seq_len_1, n_heads, dim_model].
        attn: Computed attention between the keys and the queries.
            Shape of [batch_size, n_heads, seq_len_1, seq_len_2].
    """
    d = q.size(-1)
    attn = torch.einsum("b s h d, b t h d -> b h s t", q, k)
    attn = attn / math.sqrt(d)

    if mask is not None:
        attn = attn.masked_fill(mask == True, -1e9)

    attn = torch.nn.functional.softmax(attn, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    y = torch.einsum("b h s t, b t h d -> b s h d", attn, v)

    return y, attn


class MultiheadAttention(nn.Module):
    """Multihead attention module.

    Can be used as a self-attention and cross-attention layer.
    The queries, keys and values are projected into multiple heads
    before computing the attention between those tensors.

    Parameters
    ----------
        dim: Dimension of the input tokens.
        n_heads: Number of heads. `dim` must be divisible by `n_heads`.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float,
    ):
        """Init multihead attention."""
        super().__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        # self.heads_q = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(n_heads)])
        # self.heads_k = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(n_heads)])
        # self.heads_v = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(n_heads)])

        self.W_o = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        key_padding_mask: torch.BoolTensor = None,
        attn_mask: torch.BoolTensor = None,
    ) -> torch.FloatTensor:
        """Compute the scaled multi-head attention.

        Form the input queries, keys and values.

        Project those queries, keys and values before feeding them
        to the `attention` function.

        The masks are boolean masks. Tokens are prevented to attends to
        positions where the mask is `True`.

        Args
        ----
            q: Batch of queries.
                Shape of [batch_size, seq_len_1, dim_model].
            k: Batch of keys.
                Shape of [batch_size, seq_len_2, dim_model].
            v: Batch of values.
                Shape of [batch_size, seq_len_2, dim_model].
            key_padding_mask: Prevent attending to padding tokens.
                Shape of [batch_size, seq_len_2].
            attn_mask: Prevent attending to subsequent tokens.
                Shape of [seq_len_1, seq_len_2].

        Output
        ------
            y: Computed multihead attention.
                Shape of [batch_size, seq_len_1, dim_model].
        """
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = einops.rearrange(
            q, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim
        )
        k = einops.rearrange(
            k, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim
        )
        v = einops.rearrange(
            v, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim
        )

        if attn_mask is not None:
            attn_mask = einops.repeat(
                attn_mask, "s1 s2 -> b h s1 s2", b=q.size(0), h=self.n_heads
            )
        if key_padding_mask is not None:
            key_padding_mask = einops.repeat(
                key_padding_mask,
                "b s2 -> b h s1 s2",
                h=self.n_heads,
                s1=q.size(1),
            )
            attn_mask = (
                key_padding_mask
                if attn_mask is None
                else attn_mask | key_padding_mask
            )

        y, attn = attention(
            q=q, k=k, v=v, mask=attn_mask, dropout=self.dropout
        )

        y = einops.rearrange(y, "b s h d -> b s (h d)")
        y = self.W_o(y)

        return y


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer.

    Parameters
    ----------
        d_model: The dimension of decoders inputs/outputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, d_ff: int, nhead: int, dropout: float):
        """Init decoder."""
        super().__init__()

        self.masked_attention = MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attention = MultiheadAttention(d_model, nhead, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(
        self,
        src: torch.FloatTensor,
        tgt: torch.FloatTensor,
        tgt_mask_attn: torch.BoolTensor,
        src_key_padding_mask: torch.BoolTensor,
        tgt_key_padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Decode the next target tokens based on the previous tokens.

        Args
        ----
            src: Batch of source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in
                                  src sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt
                                  sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y:  Batch of sequence of embeddings representing the predicted
                target tokens
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        tgt_attn = self.masked_attention(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask_attn,
            key_padding_mask=tgt_key_padding_mask,
        )

        tgt = tgt + self.dropout1(tgt_attn)
        tgt = self.norm1(tgt)

        if src is not None:
            tgt_src_attn = self.multihead_attention(
                q=tgt, k=src, v=src, key_padding_mask=src_key_padding_mask
            )
            tgt = tgt + self.dropout2(tgt_src_attn)
            tgt = self.norm2(tgt)

        ffn_out = self.ffn(tgt)

        tgt = tgt + self.dropout3(ffn_out)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    """Implementation of the transformer decoder stack.

    Parameters
    ----------
        d_model: The dimension of decoders inputs/outputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_decoder_layers: Number of stacked decoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_decoder_layer: int,
        nhead: int,
        dropout: float,
    ):
        """Init decoder."""
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, d_ff, nhead, dropout)
                for _ in range(num_decoder_layer)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.FloatTensor,
        tgt: torch.FloatTensor,
        tgt_mask_attn: torch.BoolTensor,
        src_key_padding_mask: torch.BoolTensor,
        tgt_key_padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Decode the source sequence by sequentially passing.

        the encoded source sequence and the target sequence through the decoder
        stack.

        Args
        ----
            src: Batch of encoded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src
                                  sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt
                                  sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y:  Batch of sequence of embeddings representing the predicted
                target tokens
                Shape of [batch_size, tgt_seq_len, dim_model].
        """
        output = tgt
        for layer in self.layers:
            output = layer(
                src,
                output,
                tgt_mask_attn,
                src_key_padding_mask,
                tgt_key_padding_mask,
            )

        return output


class TransformerEncoderLayer(nn.Module):
    """Single encoder layer.

    Parameters
    ----------
        d_model: The dimension of input tokens.
        dim_feedforward: Hidden dimension of the feedforward networks.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        nhead: int,
        dropout: float,
    ):
        """Init encoder."""
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(
        self, src: torch.FloatTensor, key_padding_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        """Encode the input. Does not attend to masked inputs.

        Args
        ----
            src: Batch of embedded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source tokens.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        src = src + self.dropout1(attn)
        src = self.norm1(src)

        ffn_out = self.ffn(src)
        src = src + self.dropout2(ffn_out)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Implementation of the transformer encoder stack.

    Parameters
    ----------
        d_model: The dimension of encoders inputs.
        dim_feedforward: Hidden dimension of the feedforward networks.
        num_encoder_layers: Number of stacked encoders.
        nheads: Number of heads for each multi-head attention.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        nheads: int,
        dropout: float,
    ):
        """Init encoder."""
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, dim_feedforward, nheads, dropout
                )
                for _ in range(num_encoder_layers)
            ]
        )

    def forward(
        self, src: torch.FloatTensor, key_padding_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        """Encode the source sequence by sequentially passing.

        the source sequence through the encoder stack.

        Args
        ----
            src: Batch of embedded source sentences.
                Shape of [batch_size, src_seq_len, dim_model].
            key_padding_mask: Mask preventing attention to padding tokens.
                Shape of [batch_size, src_seq_len].

        Output
        ------
            y: Batch of encoded source sequence.
                Shape of [batch_size, src_seq_len, dim_model].
        """
        output = src
        for layer in self.layers:
            output = layer(output, key_padding_mask)

        return output


class Transformer(nn.Module):
    """Implementation of a Transformer based on the paper.

    https://arxiv.org/pdf/1706.03762.pdf

    Parameters
    ----------
        d_model: The dimension of encoders/decoders inputs/ouputs.
        nhead: Number of heads for each multi-head attention.
        num_encoder_layers: Number of stacked encoders.
        num_decoder_layers: Number of stacked encoders.
        dim_feedforward: Hidden dimension of the feedforward networks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        """Init transformer."""
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model, dim_feedforward, num_encoder_layers, nhead, dropout
        )
        self.decoder = TransformerDecoder(
            d_model, dim_feedforward, num_decoder_layers, nhead, dropout
        )

    def forward(
        self,
        src: torch.FloatTensor,
        tgt: torch.FloatTensor,
        tgt_mask_attn: torch.BoolTensor,
        src_key_padding_mask: torch.BoolTensor,
        tgt_key_padding_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Compute next token embeddings.

        Args
        ----
            src: Batch of source sequences.
                Shape of [batch_size, src_seq_len, dim_model].
            tgt: Batch of target sequences.
                Shape of [batch_size, tgt_seq_len, dim_model].
            tgt_mask_attn: Mask to prevent attention to subsequent tokens.
                Shape of [tgt_seq_len, tgt_seq_len].
            src_key_padding_mask: Mask to prevent attention to padding in src
                                  sequence.
                Shape of [batch_size, src_seq_len].
            tgt_key_padding_mask: Mask to prevent attention to padding in tgt
                                  sequence.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y: Next token embeddings, given the previous target tokens and the
               source tokens.
               Shape of [batch_size, tgt_seq_len, dim_model].
        """
        memory = self.encoder(src, key_padding_mask=src_key_padding_mask)
        return self.decoder(
            memory,
            tgt,
            tgt_mask_attn,
            src_key_padding_mask,
            tgt_key_padding_mask,
        )


class TranslationTransformer(nn.Module):
    """Basic Transformer encoder and decoder for a translation task.

    Manage the masks creation, and the token embeddings.
    Position embeddings can be learnt with a standard `nn.Embedding` layer.

    Parameters
    ----------
        n_tokens_src: Number of tokens in the source vocabulary.
        n_tokens_tgt: Number of tokens in the target vocabulary.
        n_heads: Number of heads for each multi-head attention.
        dim_embedding: Dimension size of the word embeddings
                       (for both language).
        dim_hidden: Dimension size of the feedforward layers
            (for both the encoder and the decoder).
        n_layers: Number of layers in the encoder and decoder.
        dropout: Dropout rate.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
    """

    def __init__(
        self,
        n_tokens_src: int,
        n_tokens_tgt: int,
        n_heads: int,
        dim_embedding: int,
        dim_hidden: int,
        n_layers: int,
        dropout: float,
        src_pad_idx: int,
        tgt_pad_idx: int,
    ):
        """Init the translation transformer."""
        super().__init__()

        self.src_embeddings = nn.Embedding(
            n_tokens_src, dim_embedding, src_pad_idx
        )
        self.src_pad_idx = src_pad_idx

        self.tgt_embeddings = nn.Embedding(
            n_tokens_tgt, dim_embedding, tgt_pad_idx
        )
        self.tgt_pad_idx = tgt_pad_idx

        self.positional_encoding = PositionalEncoding(dim_embedding, dropout)

        self.transformer = Transformer(
            dim_embedding, n_heads, n_layers, n_layers, dim_hidden, dropout
        )

        self.out_layer = nn.Linear(dim_embedding, n_tokens_tgt)

    def forward(
        self, source: torch.LongTensor, target: torch.LongTensor
    ) -> torch.FloatTensor:
        """Predict the target tokens logites based on the source tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, seq_len_src].
            target: Batch of target sentences.
                Shape of [batch_size, seq_len_tgt].

        Output
        ------
            y: Distributions over the next token for all tokens in each
                sentences.
                Those need to be the logits only, do not apply a softmax
                because
                it will be done in the loss computation for numerical
                stability.
                See
                Shape of [batch_size, seq_len_tgt, n_tokens_tgt].
        """
        src = self.positional_encoding(self.src_embeddings(source))
        tgt = self.positional_encoding(self.tgt_embeddings(target))

        tgt_mask_attn = torch.triu(
            torch.ones(target.size(1), target.size(1), device=DEVICE),
            diagonal=1,
        ).bool()

        src_key_padding_mask = source == self.src_pad_idx
        tgt_key_padding_mask = target == self.tgt_pad_idx

        y = self.transformer(
            src, tgt, tgt_mask_attn, src_key_padding_mask, tgt_key_padding_mask
        )

        y = self.out_layer(y)

        return y


class GenerativeDecoderLayer(nn.Module):
    """Single decoder layer for a Decoder-Only Transformer model."""

    def __init__(self, d_model, d_ff, nhead, dropout):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask_attn):
        # Masked self-attention
        attn_output = self.self_attention(
            tgt, tgt, tgt, attn_mask=tgt_mask_attn
        )
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        # Feed-forward
        ffn_output = self.ffn(tgt)
        tgt = self.norm2(tgt + self.dropout2(ffn_output))
        return tgt


class TransformerDecoderOnly(nn.Module):
    """Decoder-Only Transformer model."""

    def __init__(self, d_model, nhead, num_decoder_layers, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GenerativeDecoderLayer(d_model, d_ff, nhead, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, tgt_mask_attn):
        for layer in self.layers:
            tgt = layer(tgt, tgt_mask_attn)
        return self.norm(tgt)


class GenerativeTransformer(nn.Module):
    """A simple wrapper for the Decoder-Only Transformer model."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_decoder = TransformerDecoderOnly(
            d_model, nhead, num_layers, d_ff, dropout
        )
        self.out_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt):
        tgt_mask_attn = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=DEVICE), diagonal=1
        ).bool()

        tgt = self.pos_encoder(self.embedding(tgt))
        tgt = self.transformer_decoder(tgt, tgt_mask_attn)

        return self.out_layer(tgt)
