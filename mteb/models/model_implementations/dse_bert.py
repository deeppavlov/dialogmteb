from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel,
)

from mteb.models import ModelMeta
from mteb.models.model_meta import ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs, PromptType


class PSCBert(BertPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128):
        super().__init__(config)
        self.bert = BertModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False),
        )

    def forward(self, input_ids, attention_mask, task_type):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            """
            When both query and reponse are single-turn sentence, input_ids are in shape
            Batch_Size * 2 * Max_Sequence_Length

            When query is multi-turn dialogue and reponse is single-turn sentence, input_ids are in shape
            Batch_Size * (Num_of_turn + 1) * Max_Sequence_Length

            See 'prepare_pairwise_input_multiturn_concatenate()' and 'prepare_pairwise_input()' in training.py for more details

            The last index of the second dimension always stands for the response, the rest stands for the query
            """
            if input_ids.shape[1] == 2:
                input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
                attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)
            else:
                batch_size = input_ids.shape[0]
                input_ids_1 = input_ids[:, :-1, :].view(batch_size, -1)
                input_ids_2 = input_ids[:, -1, :]
                attention_mask_1 = attention_mask[:, :-1, :].view(batch_size, -1)
                attention_mask_2 = attention_mask[:, -1, :]

            # mean embeddings
            bert_output_1 = self.bert.forward(
                input_ids=input_ids_1, attention_mask=attention_mask_1
            )
            bert_output_2 = self.bert.forward(
                input_ids=input_ids_2, attention_mask=attention_mask_2
            )
            attention_mask_1 = attention_mask_1.unsqueeze(-1)
            attention_mask_2 = attention_mask_2.unsqueeze(-1)
            mean_output_1 = torch.sum(
                bert_output_1[0] * attention_mask_1, dim=1
            ) / torch.sum(attention_mask_1, dim=1)
            mean_output_2 = torch.sum(
                bert_output_2[0] * attention_mask_2, dim=1
            ) / torch.sum(attention_mask_2, dim=1)

            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2

    # pass BERT embedding through the contrastive heads to get logits
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    # calculate the embedding of an input sentence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        bert_output = self.bert.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(
            attention_mask, dim=1
        )
        return embeddings


class PSCRoberta(RobertaPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.emb_size = self.roberta.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False),
        )

    # calculate the embedding of an input sentence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        bert_output = self.roberta.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(
            attention_mask, dim=1
        )
        return embeddings


class DSEWrapper(SentenceTransformerEncoderWrapper):
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        *,
        model_class: type[PSCBert | PSCRoberta],
        embed_dim: int | None = None,
        **kwargs,
    ) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = model_class.from_pretrained(model, revision=revision)
        self.model.eval()
        self.model.to(device)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        embs = []
        for batch in tqdm(inputs, desc="Encoding"):
            inputs = self.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.model.get_mean_embeddings(
                inputs["input_ids"], inputs["attention_mask"]
            )
            embs.append(output.cpu().detach().numpy())
        return np.vstack(embs)


dse_bert_base = ModelMeta(
    loader=DSEWrapper,
    loader_kwargs={
        "model_class": PSCBert,
    },
    name="aws-ai/dse-bert-base",
    revision="918ad931256ade24add8b1840a710e9e96bc9b40",
    release_date="2022-07-10",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=23_440_896,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/aws-ai/dse-bert-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)

dse_bert_large = ModelMeta(
    loader=DSEWrapper,
    loader_kwargs={
        "model_class": PSCBert,
    },
    name="aws-ai/dse-bert-large",
    revision="7100d49b06f74ea00f6acebe4f74d342ec9b30d2",
    release_date="2022-07-10",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=31254528,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=1024,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/aws-ai/dse-bert-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)


dse_roberta_base = ModelMeta(
    loader=DSEWrapper,
    loader_kwargs={
        "model_class": PSCRoberta,
    },
    name="aws-ai/dse-roberta-base",
    revision="bdb4005e439cd26d3736a7a45f56737d7f7cd47c",
    release_date="2022-07-11",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=38603520,
    memory_usage_mb=None,
    max_tokens=514,
    embed_dim=768,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/aws-ai/dse-roberta-base",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)


dse_roberta_large = ModelMeta(
    loader=DSEWrapper,
    loader_kwargs={
        "model_class": PSCRoberta,
    },
    name="aws-ai/dse-roberta-large",
    revision="1a3a00fb5a1e4a552f1c75a2a537eff09bea777c",
    release_date="2022-07-11",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=51471360,
    memory_usage_mb=None,
    max_tokens=514,
    embed_dim=1024,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers"],
    reference="https://huggingface.co/aws-ai/dse-roberta-large",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=None,
    training_datasets=None,
    adapted_from=None,
    superseded_by=None,
    modalities=["text"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    output_dtypes=None,
    extra_requirements_groups=None,
)
