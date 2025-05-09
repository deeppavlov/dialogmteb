from __future__ import annotations

import logging
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from mteb.abstasks import TaskMetadata
from mteb.model_meta import ModelMeta, ScoringFunction
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.bge_models import bge_training_data
from mteb.requires_package import requires_package
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class Model2VecModel(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        **kwargs,
    ) -> None:
        """Wrapper for Model2Vec models.

        Args:
            model_name: The Model2Vec model to load from HuggingFace Hub.
            **kwargs: Additional arguments to pass to the wrapper.
        """
        requires_package(self, "model2vec", model_name, "pip install 'mteb[model2vec]'")
        from model2vec import StaticModel  # type: ignore

        self.model_name = model_name
        self.model = StaticModel.from_pretrained(self.model_name)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        sentences = [text for batch in inputs for text in batch["text"]]
        return self.model.encode(sentences).astype(np.float32)


m2v_base_glove_subword = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/M2V_base_glove_subword",
    languages=["eng_Latn"],
    open_weights=True,
    revision="5f4f5ca159b7321a8b39739bba0794fa0debddf4",
    release_date="2024-09-21",
    n_parameters=int(103 * 1e6),
    memory_usage_mb=391,
    max_tokens=np.inf,  # Theoretically infinite
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/M2V_base_glove_subword",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)


m2v_base_glove = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/M2V_base_glove",
    languages=["eng_Latn"],
    open_weights=True,
    revision="38ebd7f10f71e67fa8db898290f92b82e9cfff2b",
    release_date="2024-09-21",
    n_parameters=int(102 * 1e6),
    memory_usage_mb=391,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/M2V_base_glove",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

m2v_base_output = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/M2V_base_output",
    languages=["eng_Latn"],
    open_weights=True,
    revision="02460ae401a22b09d2c6652e23371398329551e2",
    release_date="2024-09-21",
    n_parameters=int(7.56 * 1e6),
    memory_usage_mb=29,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/M2V_base_output",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

m2v_multilingual_output = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/M2V_multilingual_output",
    languages=["eng_Latn"],
    open_weights=True,
    revision="2cf4ec4e1f51aeca6c55cf9b93097d00711a6305",
    release_date="2024-09-21",
    n_parameters=int(128 * 1e6),
    memory_usage_mb=489,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/M2V_multilingual_output",
    use_instructions=False,
    adapted_from="sentence-transformers/LaBSE",
    superseded_by=None,
    training_datasets=None,
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

potion_base_2m = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/potion-base-2M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="86db093558fbced2072b929eb1690bce5272bd4b",
    release_date="2024-10-29",
    n_parameters=int(2 * 1e6),
    memory_usage_mb=7,
    max_tokens=np.inf,
    embed_dim=64,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/potion-base-2M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

potion_base_4m = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/potion-base-4M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="81b1802ada41afcd0987a37dc15e569c9fa76f04",
    release_date="2024-10-29",
    n_parameters=int(3.78 * 1e6),
    memory_usage_mb=14,
    max_tokens=np.inf,
    embed_dim=128,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/potion-base-4M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

potion_base_8m = ModelMeta(
    loader=Model2VecModel,
    name="minishlab/potion-base-8M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="dcbec7aa2d52fc76754ac6291803feedd8c619ce",
    release_date="2024-10-29",
    n_parameters=int(7.56 * 1e6),
    memory_usage_mb=29,
    max_tokens=np.inf,
    embed_dim=256,
    license="mit",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["NumPy", "Sentence Transformers"],
    reference="https://huggingface.co/minishlab/potion-base-8M",
    use_instructions=False,
    adapted_from="BAAI/bge-base-en-v1.5",
    superseded_by=None,
    training_datasets=bge_training_data,  # distilled
    public_training_code="https://github.com/MinishLab/model2vec",
    public_training_data=None,
)

pubmed_bert_100k = ModelMeta(
    loader=Model2VecModel,
    name="NeuML/pubmedbert-base-embeddings-100K",
    languages=["eng_Latn"],
    open_weights=True,
    revision="bac5e3b12fb8c650e92a19c41b436732c4f16e9e",
    release_date="2025-01-03",
    n_parameters=1 * 1e5,
    memory_usage_mb=0,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-100K",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-100K#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_500k = ModelMeta(
    loader=Model2VecModel,
    name="NeuML/pubmedbert-base-embeddings-500K",
    languages=["eng_Latn"],
    open_weights=True,
    revision="34ba71e35c393fdad7ed695113f653feb407b16b",
    release_date="2025-01-03",
    n_parameters=5 * 1e5,
    memory_usage_mb=2,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-500K",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-500K#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_1m = ModelMeta(
    loader=Model2VecModel,
    name="NeuML/pubmedbert-base-embeddings-1M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="2b7fed222594708da6d88bcda92ae9b434b7ddd1",
    release_date="2025-01-03",
    n_parameters=1 * 1e6,
    memory_usage_mb=2,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-1M",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-1M#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_2m = ModelMeta(
    loader=Model2VecModel,
    name="NeuML/pubmedbert-base-embeddings-2M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="1d7bbe04d6713e425161146bfdc71473cbed498a",
    release_date="2025-01-03",
    n_parameters=1.95 * 1e6,
    memory_usage_mb=7,
    max_tokens=np.inf,
    embed_dim=64,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-2M",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-2M#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)

pubmed_bert_8m = ModelMeta(
    loader=Model2VecModel,
    name="NeuML/pubmedbert-base-embeddings-8M",
    languages=["eng_Latn"],
    open_weights=True,
    revision="387d350015e963744f4fafe56a574b7cd48646c9",
    release_date="2025-01-03",
    n_parameters=7.81 * 1e6,
    memory_usage_mb=30,
    max_tokens=np.inf,
    embed_dim=256,
    license="apache-2.0",
    similarity_fn_name="cosine",
    framework=["NumPy"],
    reference="https://huggingface.co/NeuML/pubmedbert-base-embeddings-8M",
    use_instructions=False,
    adapted_from="NeuML/pubmedbert-base-embeddings",
    superseded_by=None,
    training_datasets={},
    public_training_code="https://huggingface.co/NeuML/pubmedbert-base-embeddings-8M#training",
    public_training_data="https://pubmed.ncbi.nlm.nih.gov/download/",
)
