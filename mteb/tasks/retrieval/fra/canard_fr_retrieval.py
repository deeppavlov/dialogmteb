from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FrCanard(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FrCanard",
        description="canard",
        reference="https://huggingface.co/datasets/DeepPavlov/canard_fr",
        dataset={
            "path": "DeepPavlov/canard_fr",
            "revision": "e1ea7582a3e62c18262a28935a6a579f45a303e1",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
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
