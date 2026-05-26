from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class WiardOfWikipedia(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="WiardOfWikipedia",
        description="WiardOfWikipedia",
        reference=None,
        dataset={
            "path": "DeepPavlov/wizard_of_wikipedia",
            "revision": "a806e8f492e91cdcfe5a86ff6fa5cefaf2dcf11c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=[],
        task_subtypes=[],
        license=None,
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{dinan2019wizardwikipediaknowledgepoweredconversational,
    title={Wizard of Wikipedia: Knowledge-Powered Conversational agents}, 
    author={Emily Dinan and Stephen Roller and Kurt Shuster and Angela Fan and Michael Auli and Jason Weston},
    year={2019},
    eprint={1811.01241},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/1811.01241}, 
}
""",
    )
