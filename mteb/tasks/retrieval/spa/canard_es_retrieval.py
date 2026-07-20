from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsCanard(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsCanard",
        description="canard",
        reference="https://huggingface.co/datasets/DeepPavlov/canard_es",
        dataset={
            "path": "DeepPavlov/canard_es",
            "revision": "601d3d582e0123998c9c1400fb150bce8a92bb88",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["spa-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2019-12-31"),
        domains=[],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["Canard"],
    )
