from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FrWizardOfWikipedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FrWizardOfWikipedia",
        description="WizardOfWikipedia",
        reference="https://huggingface.co/datasets/DeepPavlov/wizard_of_wikipedia_fr",
        dataset={
            "path": "DeepPavlov/wizard_of_wikipedia_fr",
            "revision": "0c37af2e1d0e776a8d63d86b42fc4f90b19a2811",
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
        adapted_from=["WiardOfWikipedia"],
    )
