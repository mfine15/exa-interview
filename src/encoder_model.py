from dataclasses import dataclass
import random
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch import Tensor, nn
import torch
from typing import Optional, Dict, Tuple, List, Union, cast
from transformers.file_utils import ModelOutput
import numpy as np
from torch.nn import functional as F
import wandb


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    docs_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    labels: Optional[Tensor] = None


class Encoder(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = None,
        normalized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 0.02,
        margin: float = 0.5,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        self.normalized = normalized
        self.margin = margin
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.config = self.model.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        docs_out = self.model(**features, return_dict=True)
        docs_reps = self.sentence_embedding(
            docs_out.last_hidden_state, features["attention_mask"]
        )
        docs_reps = torch.nn.functional.normalize(docs_reps, dim=-1)
        return docs_reps.contiguous()

    def compute_similarity(self, q_reps, docs_reps):
        try:
            if len(docs_reps.size()) == 2:
                return torch.matmul(q_reps, docs_reps.transpose(0, 1))
            return torch.matmul(q_reps, docs_reps.transpose(-2, -1))
        except RuntimeError as e:
            raise e

    def compute_loss(self, scores, target):
        positive_scores = scores[:, 0]

        # Mask to ignore the positive document when computing the max of negative scores
        mask = torch.ones_like(scores)
        mask[:, 0] = 0
        # Apply the mask and compute the maximum negative score for each query
        masked_scores = mask * scores
        max_negative_scores = torch.topk(masked_scores, min(3, scores.shape[1]), dim=1)[
            0
        ].mean(dim=1)
        losses = F.relu((positive_scores - max_negative_scores) + self.margin)

        if random.random() < 0.001:

            data = {
                "max_neg": wandb.Histogram(max_negative_scores.detach().cpu().numpy()),
                "max_pos": wandb.Histogram(positive_scores.detach().cpu().numpy()),
                "margin": self.margin,
            }
            wandb.log(data)

        return losses.mean()

    def forward(self, query: Dict[str, Tensor] = None, doc: Dict[str, Tensor] = None):
        q_reps = self.encode(query)
        docs_reps = self.encode(doc)

        group_size = docs_reps.size(0) // q_reps.size(0)
        q_reps_view = q_reps[:, None, :]

        scores = self.compute_similarity(q_reps, docs_reps) / self.temperature  # B B*G
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * group_size
        # loss = self.compute_loss(scores, target)
        loss = self.cross_entropy(scores, target)

        out = EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            docs_reps=docs_reps,
            labels=target,
        )
        # print([(k, v.shape) for k, v in out.items()])
        # print(
        #     "shapes",
        #     out.loss.shape,
        #     out.scores.shape,
        #     out.q_reps.shape,
        #     out.docs_reps.shape,
        #     out.labels.shape,
        # )
        return out

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu() for k, v in state_dict.items()}
        )
        self.model.save_pretrained(output_dir, state_dict=state_dict)


class TrainedEncoder(Encoder):
    def __init__(
        self,
        model_name: str = None,
        normalized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 0.02,
        device: str = "cuda",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.pooling_method = sentence_pooling_method
        super().__init__(
            model_name,
            normalized,
            sentence_pooling_method,
            temperature,
        )
        self.model.to(self.device)

    def pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor = None
    ):
        if self.pooling_method == "cls":
            return last_hidden_state[:, 0]
        elif self.pooling_method == "mean":
            s = torch.sum(
                last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1
            )
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d

    @torch.no_grad()
    def encode(
        self,
        sentences,
        batch_size=1024,
        show_progress_bar=False,
        convert_to_tensor=True,
        **kwargs
    ):
        self.model.eval()
        all_embeddings = []
        for start_index in tqdm(
            range(0, len(sentences), batch_size),
            desc="Inference Embeddings",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences[start_index : start_index + batch_size]

            if (
                isinstance(sentences_batch[0], dict) and "text" in sentences_batch[0]
            ):  # hack for MTEB
                sentences_batch = [s["text"] for s in sentences_batch]

            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return all_embeddings

    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: int = 1024,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        convert_to_tensor=True,
    ) -> np.ndarray:
        """
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        """
        return self.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )

    def encode_corpus(self, corpus, **kwargs) -> np.ndarray:
        """
        This function will be used for retrieval task
        encode corpus for retrieval task
        """
        return self.encode(corpus, **kwargs)
