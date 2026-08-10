from typing import Any

from mteb.models import ModelMeta
from mteb.models.model_meta import ScoringFunction
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper


class TODBert(SentenceTransformerEncoderWrapper):
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


tod_bert_mlm = ModelMeta(
    loader=TODBert,
    name="TODBERT/TOD-BERT-MLM-V1",
    revision="34178a6c57ace7efbf9423aae288804eb163f326",
    release_date="2020-07-11",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=23442432,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "TensorFlow", "JAX", "Transformers"],
    reference="https://huggingface.co/TODBERT/TOD-BERT-MLM-V1",
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
tod_bert_jnt = ModelMeta(
    loader=TODBert,
    name="TODBERT/TOD-BERT-JNT-V1",
    revision="903797e92f97b5e61a1142636b2d604682a1032c",
    release_date="2020-07-11",
    languages=None,
    n_parameters=None,
    n_active_parameters_override=None,
    n_embedding_parameters=23442432,
    memory_usage_mb=None,
    max_tokens=512,
    embed_dim=768,
    license=None,
    open_weights=None,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "TensorFlow", "JAX", "Transformers"],
    reference="https://huggingface.co/TODBERT/TOD-BERT-JNT-V1",
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
