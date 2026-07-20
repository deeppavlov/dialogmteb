from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class EsWizardOfWikipedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="EsWizardOfWikipedia",
        description="WizardOfWikipedia",
        reference="https://huggingface.co/datasets/DeepPavlov/wizard_of_wikipedia_es",
        dataset={
            "path": "DeepPavlov/wizard_of_wikipedia_es",
            "revision": "88c838bec63a0dd421778d896127a4cf31c8a228",
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
        adapted_from=["WiardOfWikipedia"],
    )
