from mteb.abstasks import AbsTaskReranking
from mteb.abstasks.task_metadata import TaskMetadata


class FrWebLINXCandidatesReranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="FrWebLINXCandidatesReranking",
        description="WebLINX is a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. The reranking task focuses on finding relevant elements at every given step in the trajectory.",
        reference="https://huggingface.co/datasets/DeepPavlov/weblinx_fr",
        dataset={
            "path": "DeepPavlov/weblinx_fr",
            "revision": "352cf5e4cd129145d0b6a3c0188947647ff6da2c",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=[
            "validation",
            "test_iid",
            "test_cat",
            "test_geo",
            "test_vis",
            "test_web",
        ],
        eval_langs=["fra-Latn"],
        main_score="mrr_at_10",
        date=("2023-03-01", "2023-10-30"),
        domains=["Academic", "Web", "Written"],
        task_subtypes=["Code retrieval", "Conversational retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation="",
        adapted_from=["WebLINXCandidatesReranking"],
    )
