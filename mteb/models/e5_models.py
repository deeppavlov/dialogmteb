from __future__ import annotations

from mteb.model_meta import (
    ModelMeta,
    ScoringFunction,
)
from mteb.models.sentence_transformer_wrapper import sentence_transformers_loader
from mteb.types import PromptType

E5_PAPER_RELEASE_DATE = "2024-02-08"
XLMR_LANGUAGES = [
    "afr_Latn",
    "amh_Latn",
    "ara_Latn",
    "asm_Latn",
    "aze_Latn",
    "bel_Latn",
    "bul_Latn",
    "ben_Latn",
    "ben_Beng",
    "bre_Latn",
    "bos_Latn",
    "cat_Latn",
    "ces_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "ell_Latn",
    "eng_Latn",
    "epo_Latn",
    "spa_Latn",
    "est_Latn",
    "eus_Latn",
    "fas_Latn",
    "fin_Latn",
    "fra_Latn",
    "fry_Latn",
    "gle_Latn",
    "gla_Latn",
    "glg_Latn",
    "guj_Latn",
    "hau_Latn",
    "heb_Latn",
    "hin_Latn",
    "hin_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jpn_Latn",
    "jav_Latn",
    "kat_Latn",
    "kaz_Latn",
    "khm_Latn",
    "kan_Latn",
    "kor_Latn",
    "kur_Latn",
    "kir_Latn",
    "lat_Latn",
    "lao_Latn",
    "lit_Latn",
    "lav_Latn",
    "mlg_Latn",
    "mkd_Latn",
    "mal_Latn",
    "mon_Latn",
    "mar_Latn",
    "msa_Latn",
    "mya_Latn",
    "nep_Latn",
    "nld_Latn",
    "nob_Latn",
    "orm_Latn",
    "ori_Latn",
    "pan_Latn",
    "pol_Latn",
    "pus_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Latn",
    "san_Latn",
    "snd_Latn",
    "sin_Latn",
    "slk_Latn",
    "slv_Latn",
    "som_Latn",
    "sqi_Latn",
    "srp_Latn",
    "sun_Latn",
    "swe_Latn",
    "swa_Latn",
    "tam_Latn",
    "tam_Taml",
    "tel_Latn",
    "tel_Telu",
    "tha_Latn",
    "tgl_Latn",
    "tur_Latn",
    "uig_Latn",
    "ukr_Latn",
    "urd_Latn",
    "urd_Arab",
    "uzb_Latn",
    "vie_Latn",
    "xho_Latn",
    "yid_Latn",
    "zho_Hant",
    "zho_Hans",
]

MULTILINGUAL_E5_CITATION = """
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}
"""

E5_CITATION = """
@article{wang2022text,
  title={Text Embeddings by Weakly-Supervised Contrastive Pre-training},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2212.03533},
  year={2022}
}
"""

model_prompts = {
    PromptType.query.value: "query: ",
    PromptType.passage.value: "passage: ",
}

E5_TRAINING_DATA = {
    # from 4.2 in https://arxiv.org/pdf/2212.03533
    # also pre-training data from a variety of sources (stackexchange, semantic scholar, reddit, CC, ...)
    "MSMARCO": ["train"],
    "MSMARCOHardNegatives": ["train"],
    "NanoMSMARCORetrieval": ["train"],
    "MSMARCO-PL": ["train"],  # translation not trained on
    "mMARCO-NL": ["train"],  # translation not trained on
    "NQ": ["train"],
    "NQHardNegatives": ["train"],
    "NanoNQRetrieval": ["train"],
    "NQ-PL": ["train"],  # translation not trained on
    "NQ-NL": ["train"],  # translation not trained on
}

ME5_TRAINING_DATA = {
    **E5_TRAINING_DATA,
    "FEVER": ["train"],
    "FEVERHardNegatives": ["train"],
    "FEVER-NL": ["train"],  # translation not trained on
    "FEVER-PL": ["train"],  # translation not trained on
    "HotpotQA": ["train"],
    "HotpotQAHardNegatives": ["train"],
    "HotpotQA-PL": ["train"],  # translation not trained on
    "HotpotQA-NL": ["train"],  # translation not trained on
    "MIRACLRetrieval": ["train"],
    "MIRACLRetrievalHardNegatives": ["train"],
    "MIRACLReranking": ["train"],
    "MrTidyRetrieval": ["train"],
}

e5_mult_small = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/multilingual-e5-small",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="fd1525a9fd15316a2d503bf26ab031a61d056e98",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=118_000_000,
    memory_usage_mb=449,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/multilingual-e5-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,  # couldn't find
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
    adapted_from="microsoft/Multilingual-MiniLM-L12-H384",
    citation=MULTILINGUAL_E5_CITATION,
)

e5_mult_base = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/multilingual-e5-base",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="d13f1b27baf31030b7fd040960d60d909913633f",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=278_000_000,
    memory_usage_mb=1061,
    embed_dim=768,
    license="mit",
    max_tokens=514,
    reference="https://huggingface.co/intfloat/multilingual-e5-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    adapted_from="FacebookAI/xlm-roberta-base",
    training_datasets=ME5_TRAINING_DATA,
    citation=MULTILINGUAL_E5_CITATION,
)

e5_mult_large = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/multilingual-e5-large",
    languages=XLMR_LANGUAGES,
    open_weights=True,
    revision="ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=560_000_000,
    memory_usage_mb=2136,
    embed_dim=1024,
    license="mit",
    max_tokens=514,
    reference="https://huggingface.co/intfloat/multilingual-e5-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=ME5_TRAINING_DATA,
    adapted_from="FacebookAI/xlm-roberta-large",
    citation=MULTILINGUAL_E5_CITATION,
)

e5_eng_small_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/e5-small-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="dca8b1a9dae0d4575df2bf423a5edb485a431236",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=33_000_000,
    memory_usage_mb=127,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/e5-small-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    adapted_from="intfloat/e5-small",
    training_datasets=E5_TRAINING_DATA,
    citation=E5_CITATION,
)

e5_eng_small = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/e5-small",
    languages=["eng_Latn"],
    open_weights=True,
    revision="e272f3049e853b47cb5ca3952268c6662abda68f",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=33_000_000,
    memory_usage_mb=127,
    embed_dim=384,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/e5-small",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_TRAINING_DATA,
    adapted_from="sentence-transformers/all-MiniLM-L6-v2",
    citation=E5_CITATION,
)

e5_eng_base_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/e5-base-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="1c644c92ad3ba1efdad3f1451a637716616a20e8",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=109_000_000,
    memory_usage_mb=418,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/e5-base-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    superseded_by=None,
    adapted_from="intfloat/e5-base",
    citation=E5_CITATION,
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_TRAINING_DATA,
)

e5_eng_large_v2 = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/e5-large-v2",
    languages=["eng_Latn"],
    open_weights=True,
    revision="b322e09026e4ea05f42beadf4d661fb4e101d311",
    release_date=E5_PAPER_RELEASE_DATE,
    n_parameters=335_000_000,
    memory_usage_mb=1278,
    embed_dim=1024,
    license="mit",
    max_tokens=514,
    reference="https://huggingface.co/intfloat/e5-large-v2",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    superseded_by=None,
    adapted_from="intfloat/e5-large",
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_TRAINING_DATA,
    citation=E5_CITATION,
)

e5_large = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/e5-large",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4dc6d853a804b9c8886ede6dda8a073b7dc08a81",
    release_date="2022-12-26",
    n_parameters=335_000_000,
    memory_usage_mb=1278,
    embed_dim=1024,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/e5-large",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    superseded_by="intfloat/e5-large-v2",
    adapted_from="google-bert/bert-large-uncased-whole-word-masking",
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_TRAINING_DATA,
    citation=E5_CITATION,
)

e5_base = ModelMeta(
    loader=sentence_transformers_loader,
    loader_kwargs=dict(
        model_prompts=model_prompts,
    ),
    name="intfloat/e5-base",
    languages=["eng-Latn"],
    open_weights=True,
    revision="b533fe4636f4a2507c08ddab40644d20b0006d6a",
    release_date="2022-12-26",
    n_parameters=109_000_000,
    memory_usage_mb=418,
    embed_dim=768,
    license="apache-2.0",
    max_tokens=512,
    reference="https://huggingface.co/intfloat/e5-base",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    superseded_by="intfloat/e5-base-v2",
    adapted_from="google-bert/bert-base-uncased",
    public_training_code=None,
    public_training_data=None,
    training_datasets=E5_TRAINING_DATA,
    citation=E5_CITATION,
)
