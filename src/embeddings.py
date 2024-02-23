from dataclasses import dataclass, field

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.file_utils import ModelOutput
from encoder_model import Encoder

import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
import math
import random
import wandb

# Setup logger for simplicity
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    query_max_len: int = 32
    doc_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        doc = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(doc[0], list):
            doc = sum(doc, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            doc,
            padding=True,
            truncation=True,
            max_length=self.doc_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "doc": d_collated}


@dataclass
class DataArguments:
    train_data: str = field(default=None)
    test_data: str = field(default=None)
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=32,
    )

    doc_max_len: int = field(default=128)

    max_example_num_per_dataset: int = field(
        default=100000000,
    )
    max_example_num_per_dataset_eval: int = field(default=100000000)

    query_instruction_for_retrieval: str = field(default=None)
    doc_instruction_for_retrieval: str = field(default=None)
    max_eval_group_size: int = field(default=None)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(default="cls")
    normalized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True)
    margin: float = field(default=0.5)
    synthetic_data: bool = field(default=False)


@dataclass
class ModelArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field()
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)


class EmbeddingDataset(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer, test=False):
        self.dataset = datasets.load_dataset(
            "json", data_files=args.train_data, split="train"
        )

        if test:
            self.dataset = datasets.load_dataset(
                "json", data_files=args.test_data, split="train"
            )

            if args.max_example_num_per_dataset_eval is not None:
                self.dataset = self.dataset.select(
                    range(min(len(self.dataset), args.max_example_num_per_dataset_eval))
                )
        if args.max_example_num_per_dataset is not None:
            self.dataset = self.dataset.select(
                range(min(len(self.dataset), args.max_example_num_per_dataset))
            )

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]["query"]
        if self.args.query_instruction_for_retrieval is not None:
            sel = self.args.query_instruction_for_retrieval + query

        documents = []
        pos = random.choice(self.dataset[item]["pos"])
        documents.append(pos)

        if len(self.dataset[item]["neg"]) < self.args.train_group_size - 1:
            num = math.ceil(
                (self.args.train_group_size - 1) / len(self.dataset[item]["neg"])
            )
            negs = random.sample(
                self.dataset[item]["neg"] * num, self.args.train_group_size - 1
            )
        else:
            negs = random.sample(
                self.dataset[item]["neg"], self.args.train_group_size - 1
            )
        documents.extend(negs)

        return query, documents


class SyntheticEmbeddingDataset(Dataset):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        test=False,
        dataset=None,
    ):
        if dataset is None:
            self.dataset = datasets.load_dataset(
                "json", data_files=args.train_data, split="train"
            )
        else:
            self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args

        self.total_len = len(self.dataset)

        if args.max_example_num_per_dataset is not None:
            self.dataset = self.dataset.select(
                range(min(len(self.dataset), args.max_example_num_per_dataset))
            )

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        passage = self.dataset[item]["passage"]
        pos = self.dataset[item]["old_query"]
        neg = self.dataset[item]["new_queries"]

        return passage, [pos] + neg


class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model to %s", output_dir)
        self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model.forward(**inputs)

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


parser = HfArgumentParser((ModelArgs, DataArguments, CustomTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: ModelArgs
data_args: DataArguments
training_args: CustomTrainingArguments

logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
logger.info("Training/evaluation params %s", training_args)
logger.info("Model params %s", model_args)
logger.info("Data params %s", data_args)

# Set seed
set_seed(training_args.seed)

num_labels = 1
tokenizer = AutoTokenizer.from_pretrained(
    (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    ),
    cache_dir=model_args.cache_dir,
    use_fast=True,
)
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    cache_dir=model_args.cache_dir,
)
logger.info("Config: %s", config)

model = Encoder(
    model_name=model_args.model_name_or_path,
    normalized=training_args.normalized,
    sentence_pooling_method=training_args.sentence_pooling_method,
    temperature=training_args.temperature,
    margin=training_args.margin,
)

if training_args.fix_position_embedding:
    for k, v in model.named_parameters():
        if "position_embeddings" in k:
            logging.info(f"Freeze the parameters for {k}")
            v.requires_grad = False

if training_args.synthetic_data:
    dataset = datasets.load_dataset(
        "json", data_files=data_args.train_data, split="train"
    )
    split = dataset.train_test_split(test_size=0.2)
    train_dataset = SyntheticEmbeddingDataset(
        args=data_args, tokenizer=tokenizer, dataset=split["train"]
    )
    test_dataset = SyntheticEmbeddingDataset(
        args=data_args, tokenizer=tokenizer, test=True, dataset=split["test"]
    )
else:
    train_dataset = EmbeddingDataset(args=data_args, tokenizer=tokenizer)
    test_dataset = EmbeddingDataset(args=data_args, tokenizer=tokenizer, test=True)


def compute_metrics(pred):
    # q_reps, doc_reps, loss, scores, labels
    loss = {"eval-loss": pred.predictions[2].mean().item()}
    wandb.log(loss)

    return loss


trainer = BiTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=EmbedCollator(
        tokenizer,
        query_max_len=data_args.query_max_len,
        doc_max_len=data_args.doc_max_len,
    ),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

# Training
wandb.init()
trainer.train()
trainer.save_model()
