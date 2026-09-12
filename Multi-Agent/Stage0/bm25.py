from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


BM25_RETRIEVER_VERSION = "stage0-bm25-v1"
BM25_K1 = 1.2
BM25_B = 0.75


_SLOT_QUERY_TERMS: dict[str, str] = {
    "observed_threat_reality": "observed threat activity trend incidents prevalence attack change",
    "threat_capability_or_mechanism": "threat capability technique mechanism adversary behavior attack method",
    "exploitation_signal": "exploitation vulnerability exploited attack signal affected product weakness",
    "mitigation_relation_or_control": "mitigation control defensive technique protection countermeasure effectiveness",
    "mitigation_implementation": "deployment implementation maturity feasibility adoption operational implementation",
    "technical_literature": "technical research study evidence evaluation method results literature",
}


def tokenize(text: str) -> list[str]:
    """Deterministic lexical tokenizer used by the Stage 0 BM25 ranker."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_slot_query(
    *,
    threat_name: str,
    pmt_name: str,
    gap_direction: str,
    slot: str,
) -> str:
    try:
        slot_terms = _SLOT_QUERY_TERMS[slot]
    except KeyError as exc:
        raise ValueError(f"Unknown evidence slot for BM25 query: {slot}") from exc
    return (
        f"threat {threat_name} mitigation {pmt_name} "
        f"forecast gap {gap_direction} {slot_terms}"
    )


@dataclass(frozen=True)
class BM25Result:
    index: int
    score: float


def rank_bm25(
    documents: Iterable[str],
    query: str,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[BM25Result]:
    """Return deterministic Okapi BM25 scores in descending order."""
    tokenized_documents = [tokenize(document) for document in documents]
    if not tokenized_documents:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return [BM25Result(index=i, score=0.0) for i in range(len(tokenized_documents))]

    n_docs = len(tokenized_documents)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_documents) / n_docs
    avg_doc_len = avg_doc_len or 1.0

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    idf = {
        term: math.log(1.0 + (n_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in set(query_terms)
    }

    results: list[BM25Result] = []
    for index, tokens in enumerate(tokenized_documents):
        frequencies = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if not tf:
                continue
            denominator = tf + k1 * (1.0 - b + b * doc_len / avg_doc_len)
            score += idf[term] * (tf * (k1 + 1.0)) / denominator
        results.append(BM25Result(index=index, score=score))

    return sorted(results, key=lambda item: (-item.score, item.index))
