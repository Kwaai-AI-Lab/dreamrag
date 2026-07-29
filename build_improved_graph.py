#!/usr/bin/env python3
"""
build_improved_graph.py — Build high-quality knowledge graph with improved extraction.

Uses:
1. Better entity extraction (GLiNER + spaCy + patterns + deduplication)
2. Semantic relation extraction
3. Confidence-weighted scoring
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from improved_extraction import (
    ImprovedEntityExtractor,
    deduplicate_entities,
    ExtractedEntity,
    ExtractedRelation,
)
from graph import GraphStore, EntityNode, entity_id, clean_entity_name
from corpus_retrieval_analysis import CorpusLoader
from gliner_ner import DEFAULT_URL


def build_improved_graph(
    corpus_path: str,
    graph_db_path: str,
    max_docs: int = None,
    gliner_url: Optional[str] = DEFAULT_URL,
    prefer_inprocess_gliner: bool = False,
) -> Dict[int, str]:
    """Build knowledge graph with improved entity and relation extraction."""
    print(f"Building improved graph from corpus: {corpus_path}")
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

    # Initialize extractors
    extractor = ImprovedEntityExtractor(
        gliner_url=gliner_url,
        prefer_inprocess_gliner=prefer_inprocess_gliner,
    )
    print(
        f"Entity extractor initialized "
        f"(GLiNER available: {extractor.gliner_available}, "
        f"spaCy available: {extractor.spacy_available})"
    )

    # Create graph
    graph = GraphStore(graph_db_path)
    chunk_id_to_doc_id = {}

    # Track statistics
    entity_mentions: Dict[str, Set[str]] = {}
    relation_counts = defaultdict(int)
    chunk_counter = 1

    # Phase 1: Extract entities and build graph
    print("\n" + "=" * 80)
    print("PHASE 1: Entity Extraction")
    print("=" * 80)

    all_entities: List[ExtractedEntity] = []
    doc_entities: Dict[str, List[ExtractedEntity]] = {}

    for i, (doc_id, text) in enumerate(docs_to_process):
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(docs_to_process)}]")

        # Extract entities
        entities = extractor.extract_entities(text)

        if not entities:
            continue

        # Deduplicate within document
        unique_entities = deduplicate_entities(entities)
        doc_entities[doc_id] = unique_entities
        all_entities.extend(unique_entities)

        # Add to graph
        for entity in unique_entities:
            eid = entity_id(entity.name, entity.entity_type)

            node = EntityNode(
                id=eid,
                name=entity.name,
                entity_type=entity.entity_type,
                description="",
                embedding=[],
                mention_count=1,
                first_chunk_id=chunk_counter,
                confidence=entity.confidence,
            )

            graph.upsert_entity(node)

            # Track mentions
            if entity.name not in entity_mentions:
                entity_mentions[entity.name] = set()
            entity_mentions[entity.name].add(doc_id)

        # Link chunk to entities
        chunk_ids_for_doc = [entity_id(e.name, e.entity_type) for e in unique_entities]
        graph.link_chunk(chunk_counter, chunk_ids_for_doc)
        chunk_id_to_doc_id[chunk_counter] = doc_id

        chunk_counter += 1

    print(f"✓ Extracted {len(all_entities)} entity mentions from {len(doc_entities)} documents")

    # Phase 2: Global deduplication
    print("\n" + "=" * 80)
    print("PHASE 2: Global Entity Deduplication")
    print("=" * 80)

    global_dedup = deduplicate_entities(all_entities)
    print(f"✓ Deduplicated to {len(global_dedup)} unique entities")

    # Phase 3: Extract relations
    print("\n" + "=" * 80)
    print("PHASE 3: Semantic Relation Extraction")
    print("=" * 80)

    relation_count = 0
    for i, (doc_id, text) in enumerate(docs_to_process):
        if (i + 1) % 30 == 0:
            print(f"  [{i + 1}/{len(docs_to_process)}]")

        if doc_id not in doc_entities:
            continue

        # Extract relations
        relations = extractor.extract_relations(text, doc_entities[doc_id])

        for rel in relations:
            src_id = entity_id(rel.src_name, rel.src_type)
            dst_id = entity_id(rel.dst_name, rel.dst_type)

            if src_id != dst_id:
                # Use relation confidence as evidence weight
                for _ in range(max(1, int(rel.confidence * 10))):
                    graph.upsert_relation(src_id, dst_id, rel.relation_type, chunk_counter)
                    relation_count += 1
                    relation_counts[rel.relation_type] += 1

    print(f"✓ Extracted {relation_count} semantic relations")

    # Phase 4: Co-occurrence relations (lighter weight)
    print("\n" + "=" * 80)
    print("PHASE 4: Co-occurrence Relations")
    print("=" * 80)

    cooccurrence_count = 0
    for doc_id, entities in doc_entities.items():
        entity_ids = [entity_id(e.name, e.entity_type) for e in entities]

        # Create co-occurrence relations with lower weight
        for i, src_id in enumerate(entity_ids):
            for dst_id in entity_ids[i + 1 :]:
                if src_id != dst_id:
                    # Single link (lower confidence than semantic relations)
                    graph.upsert_relation(src_id, dst_id, "co_occurs", chunk_counter)
                    cooccurrence_count += 1
                    relation_counts["co_occurs"] += 1

    print(f"✓ Created {cooccurrence_count} co-occurrence relations")

    # Print statistics
    print("\n" + "=" * 80)
    print("FINAL GRAPH STATISTICS")
    print("=" * 80)
    print(f"Entities:           {graph.node_count()}")
    print(f"Total relations:    {graph.relation_count()}")
    print(f"Documents mapped:   {len(set(chunk_id_to_doc_id.values()))}")
    print()
    print("Relation breakdown:")
    for rel_type in sorted(relation_counts.keys(), key=lambda x: relation_counts[x], reverse=True):
        count = relation_counts[rel_type]
        pct = (count / sum(relation_counts.values())) * 100
        print(f"  {rel_type:<20} {count:>8} ({pct:>5.1f}%)")

    graph.close()
    return chunk_id_to_doc_id


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build improved knowledge graph")
    parser.add_argument(
        "--corpus-path",
        default="/Users/christophermayfield/Desktop/Corpus_Final_Review",
        help="Path to Corpus_Final_Review",
    )
    parser.add_argument(
        "--graph-db",
        default="./data/store/corpus_graph_improved.db",
        help="Output graph database path",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (for testing)",
    )
    parser.add_argument(
        "--gliner-url",
        default=DEFAULT_URL,
        help="GLiNER NER server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--inprocess-gliner",
        action="store_true",
        help="Prefer in-process GLiNER over HTTP server",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.graph_db).parent.mkdir(parents=True, exist_ok=True)

    # Build graph
    chunk_to_doc = build_improved_graph(
        corpus_path=args.corpus_path,
        graph_db_path=args.graph_db,
        max_docs=args.max_docs,
        gliner_url=args.gliner_url,
        prefer_inprocess_gliner=args.inprocess_gliner,
    )

    # Save chunk mapping
    mapping_path = Path(args.graph_db).with_suffix(".mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(chunk_to_doc, f, indent=2)

    print(f"\n✓ Improved graph saved to {args.graph_db}")
    print(f"✓ Mapping saved to {mapping_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
