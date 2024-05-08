"""Preprocessing module.

The tokenizers are objects that are able to divide a python string into a
list of tokens (words, punctuations, special tokens...) as a list of strings.

The special tokens are used for a particular reasons:
* *<unk>*: Replace an unknown word in the vocabulary by this default token
* *<pad>*: Virtual token used to as padding token so a batch of sentences can
           have a unique length
* *<bos>*: Token indicating the beggining of a sentence in the target sequence
* *<eos>*: Token indicating the end of a sentence in the target sequence

## Datasets

Functions and classes to build the vocabularies and the torch datasets.
The vocabulary is an object able to transform a string token into the id
(an int) of that token in the vocabulary.
"""


import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]


def get_data(data_path: str = "data", filename: str = "fra.txt") -> tuple:
    """Get the dataset from the data path."""
    df = pd.read_csv(
        data_path + "/fra.txt",
        sep="\t",
        names=["english", "french", "attribution"],
    )
    train = [(en, fr) for en, fr in zip(df["english"], df["french"])]
    train, valid = train_test_split(train, test_size=0.1, random_state=0)

    return train, valid


class TranslationDataset(Dataset):
    """Dataset for the translation task.

    Each sample is a pair of sentences, one in english and one in french.
    """

    def __init__(
        self,
        dataset: list,
        en_vocab: Vocab,
        fr_vocab: Vocab,
        en_tokenizer,
        fr_tokenizer,
    ):
        """Initialize the dataset."""
        super().__init__()

        self.dataset = dataset
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer

    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        """Return a sample.

        Args
        ----
            index: Index of the sample.

        Output
        ------
            en_tokens: English tokens of the sample, as a LongTensor.
            fr_tokens: French tokens of the sample, as a LongTensor.
        """
        # Get the strings
        en_sentence, fr_sentence = self.dataset[index]

        # To list of words
        # We also add the beggining-of-sentence and end-of-sentence tokens
        en_tokens = ["<bos>"] + self.en_tokenizer(en_sentence) + ["<eos>"]
        fr_tokens = ["<bos>"] + self.fr_tokenizer(fr_sentence) + ["<eos>"]

        # To list of tokens
        en_tokens = self.en_vocab(en_tokens)  # list[int]
        fr_tokens = self.fr_vocab(fr_tokens)

        return torch.LongTensor(en_tokens), torch.LongTensor(fr_tokens)


def yield_tokens(dataset, tokenizer, lang):
    """Tokenize the whole dataset and yield the tokens."""
    assert lang in ("en", "fr")
    sentence_idx = 0 if lang == "en" else 1

    for sentences in dataset:
        sentence = sentences[sentence_idx]
        tokens = tokenizer(sentence)
        yield tokens


def build_vocab(dataset: list, en_tokenizer, fr_tokenizer, min_freq: int):
    """Return two vocabularies, one for each language."""
    en_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, en_tokenizer, "en"),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    en_vocab.set_default_index(
        en_vocab["<unk>"]
    )  # Default token for unknown words

    fr_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, fr_tokenizer, "fr"),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    fr_vocab.set_default_index(fr_vocab["<unk>"])

    return en_vocab, fr_vocab


def preprocess(
    dataset: list,
    en_tokenizer,
    fr_tokenizer,
    max_words: int,
) -> list:
    r"""Preprocess the dataset.

    Remove samples where at least one of the sentences are too long.
    Those samples takes too much memory.
    Also remove the pending '\n' at the end of sentences.
    """
    filtered = []

    for en_s, fr_s in dataset:
        if (
            len(en_tokenizer(en_s)) >= max_words
            or len(fr_tokenizer(fr_s)) >= max_words
        ):
            continue

        en_s = en_s.replace("\n", "")
        fr_s = fr_s.replace("\n", "")

        filtered.append((en_s, fr_s))

    return filtered


def build_datasets(
    max_sequence_length: int,
    min_token_freq: int,
    en_tokenizer,
    fr_tokenizer,
    train: list,
    val: list,
) -> tuple:
    """Build the training, validation and testing datasets.

    It takes care of the vocabulary creation.

    Args
    ----
        - max_sequence_length: Maximum number of tokens in each sequences.
            Having big sequences increases dramatically the VRAM taken
            during training.
        - min_token_freq: Minimum number of occurences each token must have
            to be saved in the vocabulary. Reducing this number increases
            the vocabularies's size.
        - en_tokenizer: Tokenizer for the english sentences.
        - fr_tokenizer: Tokenizer for the french sentences.
        - train and val: List containing the pairs (english, french) sentences.


    Output
    ------
        - (train_dataset, val_dataset): Tuple of the two TranslationDataset
                                        objects.
    """
    datasets = [
        preprocess(samples, en_tokenizer, fr_tokenizer, max_sequence_length)
        for samples in [train, val]
    ]

    en_vocab, fr_vocab = build_vocab(
        datasets[0], en_tokenizer, fr_tokenizer, min_token_freq
    )

    datasets = [
        TranslationDataset(
            samples, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer
        )
        for samples in datasets
    ]

    return datasets


def generate_batch(
    data_batch: list, src_pad_idx: int, tgt_pad_idx: int
) -> tuple:
    """Add padding to the given batch.

    All the samples are of the same size.

    Args
    ----
        data_batch: List of samples.
            Each sample is a tuple of LongTensors of varying size.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.

    Output
    ------
        en_batch: Batch of tokens for the padded english sentences.
            Shape of [batch_size, max_en_len].
        fr_batch: Batch of tokens for the padded french sentences.
            Shape of [batch_size, max_fr_len].
    """
    en_batch, fr_batch = [], []
    for en_tokens, fr_tokens in data_batch:
        en_batch.append(en_tokens)
        fr_batch.append(fr_tokens)

    en_batch = pad_sequence(
        en_batch, padding_value=src_pad_idx, batch_first=True
    )
    fr_batch = pad_sequence(
        fr_batch, padding_value=tgt_pad_idx, batch_first=True
    )
    return en_batch, fr_batch
