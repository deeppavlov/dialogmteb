from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class MSMARCOv2(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MSMARCOv2",
        dataset={
            "path": "mteb/msmarco-v2",
            "revision": "b1663124850d305ab7c470bb0548acf8e2e7ea43",
        },
        description="MS MARCO is a collection of datasets focused on deep learning in search",
        reference="https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train", "dev", "dev2"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=[
            "Encyclopaedic",
            "Academic",
            "Blog",
            "News",
            "Medical",
            "Government",
            "Reviews",
            "Non-fiction",
            "Social",
            "Web",
        ],
        task_subtypes=["Question answering"],
        license="msr-la-nc",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{DBLP:journals/corr/NguyenRSGTMD16,
          author    = {Tri Nguyen and
                       Mir Rosenberg and
                       Xia Song and
                       Jianfeng Gao and
                       Saurabh Tiwary and
                       Rangan Majumder and
                       Li Deng},
          title     = {{MS} {MARCO:} {A} Human Generated MAchine Reading COmprehension Dataset},
          journal   = {CoRR},
          volume    = {abs/1611.09268},
          year      = {2016},
          url       = {http://arxiv.org/abs/1611.09268},
          archivePrefix = {arXiv},
          eprint    = {1611.09268},
          timestamp = {Mon, 13 Aug 2018 16:49:03 +0200},
          biburl    = {https://dblp.org/rec/journals/corr/NguyenRSGTMD16.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }
        }""",
        adapted_from=["MSMARCO"],
    )
