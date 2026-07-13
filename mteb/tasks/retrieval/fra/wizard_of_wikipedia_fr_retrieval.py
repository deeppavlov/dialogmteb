from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FrWizardOfWikipedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FrWizardOfWikipedia",
        description="WizardOfWikipedia",
        reference=None,
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
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{elgohary-etal-2019-unpack,
  address = {Hong Kong, China},
  author = {Elgohary, Ahmed  and
Peskov, Denis  and
Boyd-Graber, Jordan},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  doi = {10.18653/v1/D19-1605},
  editor = {Inui, Kentaro  and
Jiang, Jing  and
Ng, Vincent  and
Wan, Xiaojun},
  month = nov,
  pages = {5918--5924},
  publisher = {Association for Computational Linguistics},
  title = {Can You Unpack That? Learning to Rewrite Questions-in-Context},
  url = {https://aclanthology.org/D19-1605/},
  year = {2019},
}
""",
        adapted_from=["WiardOfWikipedia"],
    )
