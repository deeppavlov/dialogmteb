from typing import Any

from mteb.models import ModelMeta
from mteb.models.model_meta import ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper


class SimcCSE(SentenceTransformerEncoderWrapper):
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device: str | None = None,
        model_prompts: dict[str, str] | None = None,
        *,
        embed_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.modules import (
            Pooling,
            Transformer,
        )

        transformer = Transformer(model)
        pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="cls")
        model = SentenceTransformer(
            modules=[
                transformer,
                pooling,
            ]
        )
        super().__init__(
            model,
            device=device,
            model_prompts=model_prompts,
            embed_dim=embed_dim,
        )


sup_simcse_bert_base_uncased = ModelMeta(
    loader=SimcCSE,
    name="princeton-nlp/sup-simcse-bert-base-uncased",
    revision="2d82fab19ac3a73a20dd20333d27eb8a52d6e97f",
    release_date="2021-04-21",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=23440896,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "JAX", "Transformers"],
    reference="https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased",
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
sup_simcse_roberta_base = ModelMeta(
    loader=SimcCSE,
    name="princeton-nlp/sup-simcse-roberta-base",
    revision="4bf73c6b5df517f74188c5e9ec159b2208c89c08",
    release_date="2021-04-21",
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
    framework=["PyTorch", "JAX", "Transformers"],
    reference="https://huggingface.co/princeton-nlp/sup-simcse-roberta-base",
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
sup_simcse_roberta_large = ModelMeta(
    loader=SimcCSE,
    name="princeton-nlp/sup-simcse-roberta-large",
    revision="96d164d9950b72f4ce179cb1eb3414de0910953f",
    release_date="2021-04-21",
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
    framework=["PyTorch", "JAX", "Transformers"],
    reference="https://huggingface.co/princeton-nlp/sup-simcse-roberta-large",
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
sup_simcse_bert_large_uncased = ModelMeta(
    loader=SimcCSE,
    name="princeton-nlp/sup-simcse-bert-large-uncased",
    revision="fee654985cba2109906c28abb1967a2f4d4e316f",
    release_date="2021-04-21",
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
    framework=["PyTorch", "JAX", "Transformers"],
    reference="https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased",
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
