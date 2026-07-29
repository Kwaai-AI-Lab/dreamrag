#!/usr/bin/env python3
"""
build_corpus_graph.py — Extract entities from corpus and build knowledge graph.

Processes documents to:
1. Extract named entities (simple pattern matching)
2. Create entity nodes in the graph
3. Link entities to documents
4. Build basic relations from co-occurrence
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from uuid import UUID

from graph import GraphStore, EntityNode, entity_id, clean_entity_name
from corpus_retrieval_analysis import CorpusLoader


def extract_entities_simple(text: str, entity_type: str = "Unknown") -> List[Tuple[str, str]]:
    """
    Simple entity extraction using regex patterns.
    Returns list of (entity_name, entity_type) tuples.
    """
    entities = []

    # Person names: Capitalized words (simple heuristic)
    # Pattern: Capitalized Word(s) appearing multiple times
    name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    person_candidates = re.findall(name_pattern, text)

    # Count occurrences and keep entities mentioned 2+ times
    entity_counts = {}
    for candidate in person_candidates:
        if len(candidate) > 2 and candidate not in ["The", "And", "For", "With"]:
            entity_counts[candidate] = entity_counts.get(candidate, 0) + 1

    for candidate, count in entity_counts.items():
        if count >= 2:  # Mentioned multiple times
            entities.append((clean_entity_name(candidate), "Person"))

    # Organizations: Common patterns
    org_patterns = [
        r'\b([A-Z][a-z]+(?:\s+(?:Inc|Corp|Ltd|University|College|Institute|Organization|Company|Department)\.?))\b',
    ]
    for pattern in org_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 3:
                entities.append((clean_entity_name(match), "Organization"))

    # Locations: Country/City names (hardcoded set)
    locations = [
        "United States", "Europe", "France", "Germany", "Russia", "China",
        "Japan", "India", "Brazil", "Canada", "Australia", "Mexico",
        "London", "Paris", "Berlin", "Moscow", "Tokyo", "New York",
    ]
    for loc in locations:
        if loc in text:
            entities.append((loc, "Location"))

    # Concepts: Common noun phrases
    concept_patterns = [
        r'\b((?:theory|principle|law|method|algorithm|concept|process|system|phenomenon)\s+of\s+[a-z]+)\b',
    ]
    for pattern in concept_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 3:
                entities.append((clean_entity_name(match), "Concept"))

    # Remove duplicates
    return list(set(entities))


def build_entity_graph(
    corpus_path: str,
    graph_db_path: str,
    max_docs: int = None,
) -> Dict[int, str]:
    """
    Build knowledge graph from corpus documents.

    Returns:
        chunk_id_to_doc_id: Mapping for retrieval
    """
    print(f"Building graph from corpus: {corpus_path}")
    print("=" * 80)

    # Load corpus
    loader = CorpusLoader(corpus_path)
    documents = loader.load_documents()

    if max_docs:
        docs_to_process = list(documents.items())[:max_docs]
        print(f"Processing {len(docs_to_process)} documents (max_docs={max_docs})")
    else:
        docs_to_process = list(documents.items())
        print(f"Processing {len(docs_to_process)} documents")

    # Create graph
    graph = GraphStore(graph_db_path)
    chunk_id_to_doc_id = {}
    entity_mentions: Dict[str, Set[str]] = {}  # entity_name -> set of doc_ids
    chunk_counter = 1

    # Process each document
    for doc_id, text in docs_to_process:
        print(f"\n[{doc_id}]")

        # Extract entities
        entities = extract_entities_simple(text)
        print(f"  Found {len(entities)} unique entities")

        if not entities:
            continue

        # Add entities to graph
        for entity_name, entity_type in entities:
            eid = entity_id(entity_name, entity_type)

            # Create/update entity node
            node = EntityNode(
                id=eid,
                name=entity_name,
                entity_type=entity_type,
                description="",
                embedding=[],  # Empty for now
                mention_count=1,
                first_chunk_id=chunk_counter,
            )

            graph.upsert_entity(node)

            # Track mentions for relation building
            if entity_name not in entity_mentions:
                entity_mentions[entity_name] = set()
            entity_mentions[entity_name].add(doc_id)

        # Link chunk to entities
        chunk_ids_for_doc = [eid for eid, _ in [(entity_id(n, t), t) for n, t in entities]]
        graph.link_chunk(chunk_counter, chunk_ids_for_doc)
        chunk_id_to_doc_id[chunk_counter] = doc_id

        chunk_counter += 1

    # Build relations from co-occurrence
    print("\n" + "=" * 80)
    print("Building relations from co-occurrence...")

    relation_count = 0
    for topic_dir in Path(corpus_path).iterdir():
        if not topic_dir.is_dir() or topic_dir.name == "QA-Trackers":
            continue

        docs_dir = topic_dir / "documents"
        if not docs_dir.exists():
            continue

        for doc_file in sorted(docs_dir.iterdir()):
            if doc_file.name.startswith('.'):
                continue

            doc_id = f"{topic_dir.name}|{doc_file.name}"
            if doc_id not in documents:
                continue

            text = documents[doc_id]
            entities = extract_entities_simple(text)
            entity_ids = [entity_id(n, t) for n, t in entities]

            # Create relations between co-occurring entities
            for i, src_id in enumerate(entity_ids):
                for dst_id in entity_ids[i + 1 :]:
                    if src_id != dst_id:
                        graph.upsert_relation(
                            src_id,
                            dst_id,
                            "co_occurs",
                            chunk_counter,
                        )
                        relation_count += 1

    print(f"✓ Created {relation_count} relations")

    # Print graph stats
    print("\n" + "=" * 80)
    print("GRAPH STATISTICS")
    print("=" * 80)
    print(f"Entities:  {graph.node_count()}")
    print(f"Relations: {graph.relation_count()}")
    print(f"Documents mapped: {len(set(chunk_id_to_doc_id.values()))}")

    graph.close()
    return chunk_id_to_doc_id


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build knowledge graph from corpus")
    parser.add_argument(
        "--corpus-path",
        default="/Users/christophermayfield/Desktop/Corpus_Final_Review",
        help="Path to Corpus_Final_Review",
    )
    parser.add_argument(
        "--graph-db",
        default="./data/store/corpus_graph.db",
        help="Output graph database path",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (for testing)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.graph_db).parent.mkdir(parents=True, exist_ok=True)

    # Build graph
    chunk_to_doc = build_entity_graph(
        corpus_path=args.corpus_path,
        graph_db_path=args.graph_db,
        max_docs=args.max_docs,
    )

    # Save chunk mapping
    mapping_path = Path(args.graph_db).with_suffix(".mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(chunk_to_doc, f, indent=2)

    print(f"\n✓ Graph saved to {args.graph_db}")
    print(f"✓ Mapping saved to {mapping_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
