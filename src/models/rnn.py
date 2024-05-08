"""RNN model architecture for the translation task."""
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNNCell(nn.Module):
    """A single RNN layer.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        dropout: Dropout rate.

    Important note: This layer does not exactly the same thing as nn.RNNCell does.
    PyTorch implementation is only doing one simple pass over one token for each batch.
    This implementation is taking the whole sequence of each batch and provide the
    final hidden state along with the embeddings of each token in each sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ):
        """Initialize the RNN layer."""
        super().__init__()

        self.hidden_size = hidden_size

        # See pytorch definition:
        # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.Wih = nn.Linear(input_size, hidden_size, device=DEVICE)
        self.Whh = nn.Linear(hidden_size, hidden_size, device=DEVICE)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.Tanh()

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor) -> tuple:
        """Forward pass of the RNN layer.

        Go through all the sequence in x, iteratively updating
        the hidden state h.

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state.
                Shape of [batch_size, hidden_size].

        Output
        ------
            y: Token embeddings.
                Shape of [batch_size, seq_len, hidden_size].
            h: Last hidden state.
                Shape of [batch_size, hidden_size].
        """
        batch_size, seq_len, input_size = x.shape
        y = torch.zeros([batch_size, seq_len, self.hidden_size], device=DEVICE)

        for t in range(seq_len):
            input = x[:, t, :]
            w_input = self.Wih(input)
            w_hidden = self.Whh(h)
            h = self.act(w_input + w_hidden)
            y[:, t, :] = self.dropout(h)

        return y, h


class RNN(nn.Module):
    """Implementation of an RNN.

    based on https://pytorch.org/docs/stable/generated/torch.nn.RNN.html.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        num_layers: Number of layers (RNNCell or GRUCell).
        dropout: Dropout rate.
        model_type: Either 'RNN' or 'GRU', to select which model we want.
            This parameter can be removed if you decide to use the module `GRU`.
            Indeed, `GRU` should have exactly the same code as this module,
            but with `GRUCell` instead of `RNNCell`. We let the freedom for you
            to decide at which level you want to specialise the modules (either
            in `TranslationRNN` by creating a `GRU` or a `RNN`, or in `RNN`
            by creating a `GRUCell` or a `RNNCell`).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        model_type: str,
    ):
        """Initialize the RNN."""
        super().__init__()

        self.hidden_size = hidden_size
        model_class = RNNCell if model_type == "RNN" else GRUCell

        self.layers = nn.ModuleList()
        self.layers.append(model_class(input_size, hidden_size, dropout))
        for i in range(1, num_layers):
            self.layers.append(model_class(hidden_size, hidden_size, dropout))

    def forward(
        self, x: torch.FloatTensor, h: torch.FloatTensor = None
    ) -> tuple:
        """Pass the input sequence through all the RNN cells.

        Returns the output and the final hidden state of each RNN layer

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Hidden state for each RNN layer.
                Can be None, in which case an initial hidden state is created.
                Shape of [batch_size, n_layers, hidden_size].

        Output
        ------
            y: Output embeddings for each token after the RNN layers.
                Shape of [batch_size, seq_len, hidden_size].
            h: Final hidden state.
                Shape of [batch_size, n_layers, hidden_size].
        """
        input = x
        h = (
            torch.zeros(
                [x.shape[0], len(self.layers), self.hidden_size],
                device=x.device,
            )
            if h is None
            else h
        )
        final_h = torch.zeros_like(h, device=x.device)
        for l in range(len(self.layers)):
            input, h_out = self.layers[l](input, h[:, l, :])
            final_h[:, l, :] = h_out

        return input, final_h


"""GRU."""


class GRU(nn.Module):
    """Implementation of a GRU based on https://pytorch.org/docs/stable/generated/torch.nn.GRU.html.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        num_layers: Number of layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        """Initialize the GRU."""
        super().__init__()

        self.hidden_size = hidden_size

        self.layers = nn.ModuleList()
        self.layers.append(GRUCell(input_size, hidden_size, dropout))
        for _ in range(1, num_layers):
            self.layers.append(GRUCell(hidden_size, hidden_size, dropout))

    def forward(
        self, x: torch.FloatTensor, h: torch.FloatTensor = None
    ) -> tuple:
        """Forward pass of the GRU.

        Args
        ----
            x: Input sequence
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state for each layer.
                If 'None', then an initial hidden state (a zero filled tensor)
                is created.
                Shape of [batch_size, n_layers, hidden_size].

        Output
        ------
            output:
                Shape of [batch_size, seq_len, hidden_size].
            h_n: Final hidden state.
                Shape of [batch_size, n_layers, hidden size].
        """
        input = x
        h = (
            torch.zeros(
                [x.shape[0], len(self.layers), self.hidden_size],
                device=x.device,
            )
            if h is None
            else h
        )
        final_h = torch.zeros_like(h, device=x.device)

        for l in range(len(self.layers)):
            input, h_out = self.layers[l](input, h[:, l, :])
            final_h[:, l, :] = h_out

        return input, final_h


class GRUCell(nn.Module):
    """A single GRU layer.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ):
        """Initialize the GRU layer."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_r = nn.Linear(
            input_size + hidden_size, hidden_size, device=DEVICE
        )
        self.W_z = nn.Linear(
            input_size + hidden_size, hidden_size, device=DEVICE
        )
        self.W_h = nn.Linear(
            input_size + hidden_size, hidden_size, device=DEVICE
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor) -> tuple:
        """Forward pass of the GRU layer.

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state.
                Shape of [batch_size, hidden_size].

        Output
        ------
            y: Token embeddings.
                Shape of [batch_size, seq_len, hidden_size].
            h: Last hidden state.
                Shape of [batch_size, hidden_size].
        """
        batch_size, seq_len, input_size = x.shape
        y = torch.zeros([batch_size, seq_len, self.hidden_size], device=DEVICE)

        for t in range(seq_len):
            input = x[:, t, :]
            combined = torch.cat((input, h), dim=1)

            r = torch.sigmoid(self.W_r(combined))
            z = torch.sigmoid(self.W_z(combined))

            combined_r = torch.cat((input, r * h), dim=1)
            h_tilde = torch.tanh(self.W_h(combined_r))

            h = (1 - z) * h + z * h_tilde
            y[:, t, :] = self.dropout(h)

        return y, h


"""### Translation RNN."""


class TranslationRNN(nn.Module):
    """Basic RNN encoder and decoder for a translation task.

    It can run as a vanilla RNN or a GRU-RNN.

    Parameters
    ----------
        n_tokens_src: Number of tokens in the source vocabulary.
        n_tokens_tgt: Number of tokens in the target vocabulary.
        dim_embedding: Dimension size of the word embeddings (for both language).
        dim_hidden: Dimension size of the hidden layers in the RNNs
            (for both the encoder and the decoder).
        n_layers: Number of layers in the RNNs.
        dropout: Dropout rate.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
        model_type: Either 'RNN' or 'GRU', to select which model we want.
    """

    def __init__(
        self,
        n_tokens_src: int,
        n_tokens_tgt: int,
        dim_embedding: int,
        dim_hidden: int,
        n_layers: int,
        dropout: float,
        src_pad_idx: int,
        tgt_pad_idx: int,
        model_type: str,
    ):
        """Initialize the TranslationRNN."""
        super().__init__()
        self.src_embeddings = nn.Embedding(
            n_tokens_src, dim_embedding, src_pad_idx
        )
        self.tgt_embeddings = nn.Embedding(
            n_tokens_tgt, dim_embedding, tgt_pad_idx
        )

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.encoder = RNN(
            dim_embedding, dim_hidden, n_layers, dropout, model_type
        )
        self.norm = nn.LayerNorm(dim_hidden)
        self.decoder = RNN(
            dim_embedding, dim_hidden, n_layers, dropout, model_type
        )
        self.out_layer = nn.Linear(dim_hidden, n_tokens_tgt)

    def forward(
        self, source: torch.LongTensor, target: torch.LongTensor
    ) -> torch.FloatTensor:
        """Predict the target tokens logits based on the source tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, src_seq_len].
            target: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len].

        Output
        ------
            y: Distributions over the next token for all tokens in each
                sentences. Those need to be the logits only, do not apply a
                softmax because it will be done in the loss computation for
                numerical stability.
                See
                https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                for more informations.
                Shape of [batch_size, tgt_seq_len, n_tokens_tgt].
        """
        source = torch.fliplr(source)

        src_emb = self.src_embeddings(source)
        out, hidden = self.encoder(src_emb)

        hidden = self.norm(hidden)

        # teacher forcing
        tgt_emb = self.tgt_embeddings(target)
        y, hidden = self.decoder(tgt_emb, hidden)

        y = self.out_layer(y)

        return y
