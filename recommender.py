from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer


def _score_candidates(candidates, corpus_feature, time_decay_weight, encoder):
    if len(candidates) == 0:
        return candidates
    candidate_feature = encoder.encode([paper.summary for paper in candidates])
    sim = encoder.similarity(candidate_feature, corpus_feature)
    scores = (sim * time_decay_weight).sum(axis=1) * 10
    for score, candidate in zip(scores, candidates):
        candidate.score = score.item()
    return sorted(candidates, key=lambda x: x.score, reverse=True)


def rerank_paper(
    candidate_arxiv,
    candidate_bio,
    candidate_journal,
    corpus: list[dict],
    model: str = "avsolatorio/GIST-small-Embedding-v0",
):
    if (
        len(candidate_arxiv) == 0
        and len(candidate_bio) == 0
        and len(candidate_journal) == 0
    ):
        return candidate_arxiv, candidate_bio, candidate_journal

    encoder = SentenceTransformer(model)
    corpus = sorted(
        corpus,
        key=lambda x: datetime.strptime(x["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
        reverse=True,
    )
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper["data"]["abstractNote"] for paper in corpus])

    candidate_arxiv = _score_candidates(
        candidate_arxiv, corpus_feature, time_decay_weight, encoder
    )
    candidate_bio = _score_candidates(
        candidate_bio, corpus_feature, time_decay_weight, encoder
    )
    candidate_journal = _score_candidates(
        candidate_journal, corpus_feature, time_decay_weight, encoder
    )

    return candidate_arxiv, candidate_bio, candidate_journal
