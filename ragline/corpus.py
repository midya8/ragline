"""Deterministic corpus and chunking for RAGLine.

The built-in corpus is generated from fixed templates so a seed reproduces it
bit for bit. Chunks are paragraph windows with stable ids; every answer RAGLine
produces cites chunk ids from this layer.
"""

from __future__ import annotations

import random

TOPICS = {
    "photosynthesis": [
        "Photosynthesis converts light energy into chemical energy in {org}.",
        "The light-dependent reactions of photosynthesis occur in the {mem}.",
        "Chlorophyll absorbs light most strongly in the blue and red parts of the spectrum.",
        "The Calvin cycle fixes carbon dioxide into {sugar} using ATP and NADPH.",
        "Stomata regulate gas exchange for photosynthesis in {org}.",
    ],
    "databases": [
        "A B-tree index keeps {rec} sorted and allows logarithmic lookup time.",
        "Write-ahead logging records changes before they are applied to {rec}.",
        "Normalization reduces redundancy by decomposing {rec} into related tables.",
        "A query planner chooses join order using statistics about {rec}.",
        "Transactions group operations so they commit or roll back as a unit.",
    ],
    "networking": [
        "TCP provides ordered, reliable delivery using {mech} and retransmission.",
        "DNS resolves hostnames to IP addresses through a hierarchy of {mech}.",
        "TLS establishes an encrypted channel using a handshake with {mech}.",
        "Routers forward packets between networks based on {mech} tables.",
        "HTTP is a stateless request-response protocol layered on {mech}.",
    ],
    "astronomy": [
        "Redshift measures how much light from {obj} is stretched by expansion.",
        "A light-year is the distance light travels in one {unit}.",
        "Main-sequence stars fuse hydrogen into helium in their {part}.",
        "Exoplanets are detected by transit dips and radial-velocity wobbles of {obj}.",
        "The cosmic microwave background is residual radiation from the early {unit}.",
    ],
}

FILLERS = {
    "org": ["chloroplasts", "plant cells", "leaf tissue"],
    "mem": ["thylakoid membranes", "grana"],
    "sugar": ["glyceraldehyde-3-phosphate", "glucose precursors"],
    "rec": ["rows", "pages", "records"],
    "mech": ["acknowledgements", "certificates", "servers", "prefix"],
    "obj": ["distant galaxies", "host stars", "quasars"],
    "unit": ["year", "universe"],
    "part": ["cores", "interiors"],
}


def build_corpus(seed: int = 0, copies: int = 1) -> list[dict]:
    """Expand the templates into a deterministic list of {doc_id, topic, text}."""
    rng = random.Random(seed)
    docs = []
    for topic, templates in TOPICS.items():
        for copy in range(copies):
            for i, tpl in enumerate(templates):
                text = tpl
                for slot, options in FILLERS.items():
                    needle = "{" + slot + "}"
                    if needle in text:
                        text = text.replace(needle, rng.choice(options))
                docs.append({"doc_id": f"{topic}-{copy}-{i}", "topic": topic, "text": text})
    rng.shuffle(docs)
    return docs


def chunk_documents(docs: list[dict], window: int = 2) -> list[dict]:
    """Group consecutive same-topic documents into chunks of up to `window`
    documents. Chunks keep their source doc ids for citation checks."""
    chunks = []
    for doc in docs:
        if chunks and chunks[-1]["topic"] == doc["topic"] and len(chunks[-1]["doc_ids"]) < window:
            chunks[-1]["doc_ids"].append(doc["doc_id"])
            chunks[-1]["text"] += " " + doc["text"]
        else:
            chunks.append({
                "chunk_id": f"c{len(chunks)}",
                "topic": doc["topic"],
                "doc_ids": [doc["doc_id"]],
                "text": doc["text"],
            })
    return chunks
