"""# Greedy search.

Here you have to implement a geedy search to generate a target translation
from a trained model and an input source string.
The next token will simply be the most probable one.
"""

from collections import defaultdict
from itertools import takewhile

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.data.preprocess import build_datasets, generate_batch, get_data
from src.models.rnn import TranslationRNN
from src.models.transformer import TranslationTransformer
from torch.utils.data import DataLoader
from torchinfo import summary
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.metrics import bleu_score

en_tokenizer, fr_tokenizer = (
    get_tokenizer("spacy", language="en"),
    get_tokenizer("spacy", language="fr"),
)

train, valid = get_data()


def greedy_search(
    model: nn.Module,
    source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    src_tokenizer,
    device: str,
    max_sentence_length: int,
) -> str:
    """Do a greedy search to produce probable translations.

    Args
    ----
        model: The translation model. Assumes it produces logits score
               (before softmax).
        source: The sentence to translate.
        src_vocab: The source vocabulary.
        tgt_vocab: The target vocabulary.
        device: Device to which we make the inference.
        max_target: Maximum number of target sentences we keep at the end of
                    each stage.
        max_sentence_length: Maximum number of tokens for the translated
                             sentence.

    Output
    ------
        sentence: The translated source sentence.
    """
    src_tokens = ["<bos>"] + src_tokenizer(source) + ["<eos>"]
    src_tokens = src_vocab(src_tokens)

    tgt_tokens = ["<bos>"]
    tgt_tokens = tgt_vocab(tgt_tokens)

    # To tensor and add unitary batch dimension
    src_tokens = torch.LongTensor(src_tokens).to(device)
    tgt_tokens = torch.LongTensor(tgt_tokens).unsqueeze(dim=0).to(device)
    target_probs = torch.FloatTensor([1]).to(device)
    model.to(device)

    EOS_IDX = tgt_vocab["<eos>"]

    with torch.no_grad():
        batch_size, n_tokens = tgt_tokens.shape
        # Get next beams
        src = einops.repeat(src_tokens, "t -> b t", b=tgt_tokens.shape[0])
        predicted = model.forward(src, tgt_tokens)
        predicted = torch.softmax(predicted, dim=-1)
        probs, predicted = predicted[:, -1].topk(k=1, dim=-1)

    sentences = []

    for tgt_sentence in tgt_tokens:
        tgt_sentence = list(tgt_sentence)[1:]  # Remove <bos> token
        tgt_sentence = list(takewhile(lambda t: t != EOS_IDX, tgt_sentence))
        tgt_sentence = " ".join(tgt_vocab.lookup_tokens(tgt_sentence))
        sentences.append(tgt_sentence)

    sentences = [beautify(s) for s in sentences]
    # Join the sentences with their likelihood
    sentences = [(s, p.item()) for s, p in zip(sentences, target_probs)]
    # Sort the sentences by their likelihood
    sentences = [(s, p) for s, p in sorted(sentences, key=lambda k: k[1], reverse=True)]

    return sentences


def beautify(sentence: str) -> str:
    """Removes useless spaces."""
    punc = {".", ",", ";"}
    for p in punc:
        sentence = sentence.replace(f" {p}", p)

    links = {"-", "'"}
    for l in links:
        sentence = sentence.replace(f"{l} ", l)
        sentence = sentence.replace(f" {l}", l)

    return sentence


def indices_terminated(target: torch.FloatTensor, eos_token: int) -> tuple:
    """Split the target sentences between the terminated and the non-terminated
    sentence. Return the indices of those two groups.

    Args
    ----
        target: The sentences.
            Shape of [batch_size, n_tokens].
        eos_token: Value of the End-of-Sentence token.

    Output
    ------
        terminated: Indices of the terminated sentences (who's got the eos_token).
            Shape of [n_terminated, ].
        non-terminated: Indices of the unfinished sentences.
            Shape of [batch_size-n_terminated, ].
    """
    terminated = [i for i, t in enumerate(target) if eos_token in t]
    non_terminated = [i for i, t in enumerate(target) if eos_token not in t]
    return torch.LongTensor(terminated), torch.LongTensor(non_terminated)


def append_beams(
    target: torch.FloatTensor, beams: torch.FloatTensor
) -> torch.FloatTensor:
    """Add the beam tokens to the current sentences.
    Duplicate the sentences so one token is added per beam per batch.

    Args
    ----
        target: Batch of unfinished sentences.
            Shape of [batch_size, n_tokens].
        beams: Batch of beams for each sentences.
            Shape of [batch_size, n_beams].

    Output
    ------
        target: Batch of sentences with one beam per sentence.
            Shape of [batch_size * n_beams, n_tokens+1].
    """
    batch_size, n_beams = beams.shape
    n_tokens = target.shape[1]

    target = einops.repeat(
        target, "b t -> b c t", c=n_beams
    )  # [batch_size, n_beams, n_tokens]
    beams = beams.unsqueeze(dim=2)  # [batch_size, n_beams, 1]

    target = torch.cat((target, beams), dim=2)  # [batch_size, n_beams, n_tokens+1]
    target = target.view(
        batch_size * n_beams, n_tokens + 1
    )  # [batch_size * n_beams, n_tokens+1]
    return target


def beam_search(
    model: nn.Module,
    source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    src_tokenizer,
    device: str,
    beam_width: int,
    max_target: int,
    max_sentence_length: int,
) -> list:
    """Do a beam search to produce probable translations.

    Args
    ----
        model: The translation model. Assumes it produces linear score
               (before softmax).
        source: The sentence to translate.
        src_vocab: The source vocabulary.
        tgt_vocab: The target vocabulary.
        device: Device to which we make the inference.
        beam_width: Number of top-k tokens we keep at each stage.
        max_target: Maximum number of target sentences we keep at the end of
                    each stage.
        max_sentence_length: Maximum number of tokens for the translated
                             sentence.

    Output
    ------
        sentences: List of sentences orderer by their likelihood.
    """
    src_tokens = ["<bos>"] + src_tokenizer(source) + ["<eos>"]
    src_tokens = src_vocab(src_tokens)

    tgt_tokens = ["<bos>"]
    tgt_tokens = tgt_vocab(tgt_tokens)

    # To tensor and add unitary batch dimension
    src_tokens = torch.LongTensor(src_tokens).to(device)
    tgt_tokens = torch.LongTensor(tgt_tokens).unsqueeze(dim=0).to(device)
    target_probs = torch.FloatTensor([1]).to(device)
    model.to(device)

    EOS_IDX = tgt_vocab["<eos>"]
    with torch.no_grad():
        while tgt_tokens.shape[1] < max_sentence_length:
            batch_size, n_tokens = tgt_tokens.shape

            # Get next beams
            src = einops.repeat(src_tokens, "t -> b t", b=tgt_tokens.shape[0])
            predicted = model.forward(src, tgt_tokens)
            predicted = torch.softmax(predicted, dim=-1)
            probs, predicted = predicted[:, -1].topk(k=beam_width, dim=-1)

            # Separe between terminated sentences and the others
            idx_terminated, idx_not_terminated = indices_terminated(tgt_tokens, EOS_IDX)
            idx_terminated, idx_not_terminated = (
                idx_terminated.to(device),
                idx_not_terminated.to(device),
            )

            tgt_terminated = torch.index_select(tgt_tokens, dim=0, index=idx_terminated)
            tgt_probs_terminated = torch.index_select(
                target_probs, dim=0, index=idx_terminated
            )

            filter_t = lambda t: torch.index_select(t, dim=0, index=idx_not_terminated)
            tgt_others = filter_t(tgt_tokens)
            tgt_probs_others = filter_t(target_probs)
            predicted = filter_t(predicted)
            probs = filter_t(probs)

            # Add the top tokens to the previous target sentences
            tgt_others = append_beams(tgt_others, predicted)

            # Add padding to terminated target
            padd = torch.zeros(
                (len(tgt_terminated), 1), dtype=torch.long, device=device
            )
            tgt_terminated = torch.cat((tgt_terminated, padd), dim=1)

            # Update each target sentence probabilities
            tgt_probs_others = torch.repeat_interleave(tgt_probs_others, beam_width)
            tgt_probs_others *= probs.flatten()
            tgt_probs_terminated *= 0.999  # Penalize short sequences overtime

            # Group up the terminated and the others
            target_probs = torch.cat((tgt_probs_others, tgt_probs_terminated), dim=0)
            tgt_tokens = torch.cat((tgt_others, tgt_terminated), dim=0)

            # Keep only the top `max_target` target sentences
            if target_probs.shape[0] <= max_target:
                continue

            target_probs, indices = target_probs.topk(k=max_target, dim=0)
            tgt_tokens = torch.index_select(tgt_tokens, dim=0, index=indices)

    sentences = []
    for tgt_sentence in tgt_tokens:
        tgt_sentence = list(tgt_sentence)[1:]  # Remove <bos> token
        tgt_sentence = list(takewhile(lambda t: t != EOS_IDX, tgt_sentence))
        tgt_sentence = " ".join(tgt_vocab.lookup_tokens(tgt_sentence))
        sentences.append(tgt_sentence)

    sentences = [beautify(s) for s in sentences]

    # Join the sentences with their likelihood
    sentences = [(s, p.item()) for s, p in zip(sentences, target_probs)]
    # Sort the sentences by their likelihood
    sentences = [(s, p) for s, p in sorted(sentences, key=lambda k: k[1], reverse=True)]

    return sentences


def print_logs(dataset_type: str, logs: dict):
    """Print the logs.

    Args
    ----
        dataset_type: Either "Train", "Eval", "Test" type.
        logs: Containing the metric's name and value.
    """
    desc = [f"{name}: {value:.2f}" for name, value in logs.items()]
    desc = "\t".join(desc)
    desc = f"{dataset_type} -\t" + desc
    desc = desc.expandtabs(5)
    print(desc)


def topk_accuracy(
    real_tokens: torch.FloatTensor,
    probs_tokens: torch.FloatTensor,
    k: int,
    tgt_pad_idx: int,
) -> torch.FloatTensor:
    """Compute the top-k accuracy.
    We ignore the PAD tokens.

    Args
    ----
        real_tokens: Real tokens of the target sentence.
            Shape of [batch_size * n_tokens].
        probs_tokens: Tokens probability predicted by the model.
            Shape of [batch_size * n_tokens, n_target_vocabulary].
        k: Top-k accuracy threshold.
        src_pad_idx: Source padding index value.

    Output
    ------
        acc: Scalar top-k accuracy value.
    """
    total = (real_tokens != tgt_pad_idx).sum()

    _, pred_tokens = probs_tokens.topk(k=k, dim=-1)  # [batch_size * n_tokens, k]
    real_tokens = einops.repeat(
        real_tokens, "b -> b k", k=k
    )  # [batch_size * n_tokens, k]

    good = (pred_tokens == real_tokens) & (real_tokens != tgt_pad_idx)
    acc = good.sum() / total
    return acc


def loss_batch(
    model: nn.Module,
    source: torch.LongTensor,
    target: torch.LongTensor,
    config: dict,
) -> dict:
    """Compute the metrics associated with this batch.
    The metrics are:
        - loss
        - top-1 accuracy
        - top-5 accuracy
        - top-10 accuracy

    Args
    ----
        model: The model to train.
        source: Batch of source tokens.
            Shape of [batch_size, n_src_tokens].
        target: Batch of target tokens.
            Shape of [batch_size, n_tgt_tokens].
        config: Additional parameters.

    Output
    ------
        metrics: Dictionnary containing evaluated metrics on this batch.
    """
    device = config["device"]
    loss_fn = config["loss"].to(device)
    metrics = dict()

    source, target = source.to(device), target.to(device)
    target_in, target_out = target[:, :-1], target[:, 1:]

    # Loss
    pred = model(source, target_in)  # [batch_size, n_tgt_tokens-1, n_vocab]
    pred = pred.view(-1, pred.shape[2])  # [batch_size * (n_tgt_tokens - 1), n_vocab]
    target_out = target_out.flatten()  # [batch_size * (n_tgt_tokens - 1),]
    metrics["loss"] = loss_fn(pred, target_out)
    metrics["ppl"] = torch.exp(metrics["loss"])

    # Accuracy - we ignore the padding predictions
    for k in [1, 5, 10]:
        metrics[f"top-{k}"] = topk_accuracy(target_out, pred, k, config["tgt_pad_idx"])

    return metrics


def eval_model(
    model: nn.Module,
    dataloader: DataLoader,
    config: dict,
    calculate_bleu: bool = False,
    val_dataset=None,
) -> dict:
    """Evaluate the model on the given dataloader."""
    device = config["device"]
    logs = defaultdict(list)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for source, target in dataloader:
            metrics = loss_batch(model, source, target, config)
            for name, value in metrics.items():
                logs[name].append(value.cpu().item())

    for name, values in logs.items():
        logs[name] = np.mean(values)

    if calculate_bleu and val_dataset is not None:
        bleu_logs = []
        for source, target in val_dataset:
            pred, _ = beam_search(
                model,
                source,
                config["src_vocab"],
                config["tgt_vocab"],
                config["src_tokenizer"],
                device,
                beam_width=10,
                max_target=100,
                max_sentence_length=config["max_sequence_length"],
            )[0]
            bleu_logs.append(bleu_score([pred.split()], [[target.split()]]))

        logs["bleu"] = np.mean(bleu_logs)

    return logs


def train_model(model: nn.Module, config: dict):
    """Train the model in a teacher forcing manner."""
    train_loader, val_loader = config["train_loader"], config["val_loader"]
    train_dataset, val_dataset = (
        train_loader.dataset.dataset,
        val_loader.dataset.dataset,
    )
    optimizer = config["optimizer"]
    clip = config["clip"]
    device = config["device"]

    columns = ["epoch"]
    for mode in ["train", "validation"]:
        columns += [
            f"{mode} - {colname}"
            for colname in ["source", "target", "predicted", "likelihood"]
        ]
    log_table = wandb.Table(columns=columns)

    print(f'Starting training for {config["epochs"]} epochs, using {device}.')
    for e in range(config["epochs"]):
        print(f"\nEpoch {e+1}")

        model.to(device)
        model.train()
        logs = defaultdict(list)

        for batch_id, (source, target) in enumerate(train_loader):
            optimizer.zero_grad()

            metrics = loss_batch(model, source, target, config)
            loss = metrics["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            for name, value in metrics.items():
                logs[name].append(
                    value.cpu().item()
                )  # Don't forget the '.item' to free the cuda memory

            if batch_id % config["log_every"] == 0:
                for name, value in logs.items():
                    logs[name] = np.mean(value)

                train_logs = {f"Train - {m}": v for m, v in logs.items()}
                wandb.log(train_logs)
                logs = defaultdict(list)

        val_source, val_target = val_dataset[torch.randint(len(val_dataset), (1,))]
        val_pred, val_prob = beam_search(
            model,
            val_source,
            config["src_vocab"],
            config["tgt_vocab"],
            config["src_tokenizer"],
            device,  # It can take a lot of VRAM
            beam_width=10,
            max_target=100,
            max_sentence_length=config["max_sequence_length"],
        )[0]

        train_source, train_target = train_dataset[
            torch.randint(len(train_dataset), (1,))
        ]
        train_pred, train_prob = beam_search(
            model,
            train_source,
            config["src_vocab"],
            config["tgt_vocab"],
            config["src_tokenizer"],
            device,  # It can take a lot of VRAM
            beam_width=10,
            max_target=100,
            max_sentence_length=config["max_sequence_length"],
        )[0]

        # Logs
        if len(logs) != 0:
            for name, value in logs.items():
                logs[name] = np.mean(value)
            logs["bleu"] = bleu_score([train_pred.split()], [[train_target.split()]])
            train_logs = {f"Train - {m}": v for m, v in logs.items()}
        else:
            logs = {m.split(" - ")[1]: v for m, v in train_logs.items()}

        print_logs("Train", logs)

        logs = eval_model(model, val_loader, config)
        logs["bleu"] = bleu_score([val_pred.split()], [[val_target.split()]])
        print(
            f"Validation BLEU: {logs['bleu']:.2f} "
            f"Val pred: {val_pred} Val target: {val_target}"
        )
        print_logs("Eval", logs)
        val_logs = {f"Validation - {m}": v for m, v in logs.items()}

        print(val_source)
        print(val_pred)

        logs = {**train_logs, **val_logs}  # Merge dictionnaries
        wandb.log(logs)  # Upload to the WandB cloud

        # Table logs
        data = [
            e + 1,
            train_source,
            train_target,
            train_pred,
            train_prob,
            val_source,
            val_target,
            val_pred,
            val_prob,
        ]
        log_table.add_data(*data)

    final_eval = eval_model(
        model,
        val_loader,
        config,
        calculate_bleu=True,
        val_dataset=val_dataset[:2],
    )
    wandb.log({"Final evaluation - " + k: v for k, v in final_eval.items()})
    # Log the table at the end of the training
    wandb.log({"Model predictions": log_table})


"""Training the models"""

# Instanciate the datasets
# MAX_SEQ_LEN = 8
# MIN_TOK_FREQ = 20
MAX_SEQ_LEN = 60
MIN_TOK_FREQ = 2
train_dataset, val_dataset = build_datasets(
    MAX_SEQ_LEN,
    MIN_TOK_FREQ,
    en_tokenizer,
    fr_tokenizer,
    train,
    valid,
)


print(f"English vocabulary size: {len(train_dataset.en_vocab):,}")
print(f"French vocabulary size: {len(train_dataset.fr_vocab):,}")

print(f"\nTraining examples: {len(train_dataset):,}")
print(f"Validation examples: {len(val_dataset):,}")

# Build the model, the dataloaders, optimizer and the loss function
# Log every hyperparameters and arguments into the config dictionnary

config = {
    # General parameters
    "epochs": 10,
    "batch_size": 128,
    "lr": 1e-3,
    "betas": (0.9, 0.99),
    "clip": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Model parameters
    "n_tokens_src": len(train_dataset.en_vocab),
    "n_tokens_tgt": len(train_dataset.fr_vocab),
    "n_heads": 4,
    "dim_embedding": 196,
    "dim_hidden": 256,
    "n_layers": 6,
    "dropout": 0.1,
    "model_type": "RNN",
    # Others
    "max_sequence_length": MAX_SEQ_LEN,
    "min_token_freq": MIN_TOK_FREQ,
    "src_vocab": train_dataset.en_vocab,
    "tgt_vocab": train_dataset.fr_vocab,
    "src_tokenizer": en_tokenizer,
    "tgt_tokenizer": fr_tokenizer,
    "src_pad_idx": train_dataset.en_vocab["<pad>"],
    "tgt_pad_idx": train_dataset.fr_vocab["<pad>"],
    "seed": 0,
    "log_every": 50,  # Number of batches between each wandb logs
}

torch.manual_seed(config["seed"])

config["train_loader"] = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=lambda batch: generate_batch(
        batch, config["src_pad_idx"], config["tgt_pad_idx"]
    ),
)

config["val_loader"] = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=lambda batch: generate_batch(
        batch, config["src_pad_idx"], config["tgt_pad_idx"]
    ),
)

weight_classes = torch.ones(config["n_tokens_tgt"], dtype=torch.float)
weight_classes[config["tgt_vocab"]["<unk>"]] = 0.1  # Lower the importance of that class
config["loss"] = nn.CrossEntropyLoss(
    weight=weight_classes,
    ignore_index=config["tgt_pad_idx"],  # We do not have to learn those
)

config_rnn = config.copy()
config_gru = config.copy()
config_gru["model_type"] = "GRU"

model_rnn = TranslationRNN(
    config_rnn["n_tokens_src"],
    config_rnn["n_tokens_tgt"],
    config_rnn["dim_embedding"],
    config_rnn["dim_hidden"],
    config_rnn["n_layers"],
    config_rnn["dropout"],
    config_rnn["src_pad_idx"],
    config_rnn["tgt_pad_idx"],
    config_rnn["model_type"],
)

model_gru = TranslationRNN(
    config_gru["n_tokens_src"],
    config_gru["n_tokens_tgt"],
    config_gru["dim_embedding"],
    config_gru["dim_hidden"],
    config_gru["n_layers"],
    config_gru["dropout"],
    config_gru["src_pad_idx"],
    config_gru["tgt_pad_idx"],
    config_gru["model_type"],
)

model_transformer = TranslationTransformer(
    config["n_tokens_src"],
    config["n_tokens_tgt"],
    config["n_heads"],
    config["dim_embedding"],
    config["dim_hidden"],
    config["n_layers"],
    config["dropout"],
    config["src_pad_idx"],
    config["tgt_pad_idx"],
)

config_rnn["optimizer"] = optim.Adam(
    model_rnn.parameters(),
    lr=config_rnn["lr"],
    betas=config_rnn["betas"],
)

config_gru["optimizer"] = optim.Adam(
    model_gru.parameters(),
    lr=config_gru["lr"],
    betas=config_gru["betas"],
)

config["optimizer"] = optim.Adam(
    model_transformer.parameters(),
    lr=config["lr"],
    betas=config["betas"],
)


def wandb_train_model(model: nn.Module, config: dict, model_type: str, model_size: str):
    summary(
        model,
        input_size=[
            (config["batch_size"], config["max_sequence_length"]),
            (config["batch_size"], config["max_sequence_length"]),
        ],
        dtypes=[torch.long, torch.long],
        depth=3,
    )

    with wandb.init(
        config=config,
        project="INF8225 - TP3",  # Title of your project
        group=f"Transformer-Rnn-Gru - {model_size}",
        name=f"{model_type}",
        save_code=True,
    ):
        train_model(model, config)

    sentence = "It is possible to try your work here."

    preds = beam_search(
        model,
        sentence,
        config["src_vocab"],
        config["tgt_vocab"],
        config["src_tokenizer"],
        config["device"],
        beam_width=10,
        max_target=100,
        max_sentence_length=config["max_sequence_length"],
    )[:5]

    for i, (translation, likelihood) in enumerate(preds):
        print(f"{i}. ({likelihood*100:.5f}%) \t {translation}")

    torch.save(model.state_dict(), f"{model_type}_model.pth")


wandb_train_model(model_transformer, config, "Transfomer", "small")
wandb_train_model(model_rnn, config_rnn, "RNN", "small")
wandb_train_model(model_gru, config_gru, "GRU", "small")
