"""
ingestion.py — Document ingestion pipeline: chunk → embed → upload + knowledge graph extraction.

Mirrors the logic in ingestion.rs from kwaai-rag.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional

logger = logging.getLogger(__name__)


# ── Dataclasses (mirrors Rust structs) ───────────────────────────────────────

@dataclass
class GraphIngestConfig:
    store: Any                          # GraphStore (thread-safe wrapper expected)
    inference_url: str = ""
    inference_urls: list[str] = field(default_factory=list)
    model: str = "default"
    workers: int = 1
    entity_types: list[str] = field(default_factory=list)
    no_relations: bool = False
    context_window: int = 1
    gliner_client: Any = None           # Optional GliNERClient
    entity_centric: bool = False
    ec_refine_threshold: float = 0.0
    ec_refine_budget: int = 50
    ec_refine_only: bool = False
    chunk_batch: int = 1

    def effective_urls(self) -> list[str]:
        return self.inference_urls if self.inference_urls else [self.inference_url]


@dataclass
class IngestConfig:
    embed: Any                          # EmbedClient
    chunk_cfg: Any = None               # ChunkConfig
    upload_batch_size: int = 64
    graph: Optional[GraphIngestConfig] = None
    doc_meta: dict[str, str] = field(default_factory=dict)
    doc_schema: Any = None              # Optional DocSchema


@dataclass
class IngestionResult:
    chunks_ingested: int
    vectors_uploaded: int


@dataclass
class _ChunkResult:
    chunk_id: int
    entities: list
    relations: list
    embeddings: list[list[float]]


# ── Public entry point ────────────────────────────────────────────────────────

async def ingest_text(
    cfg: IngestConfig,
    meta: Any,                          # MetaStore
    doc_name: str,
    text: str,
    upload_fn: Callable[[list[tuple[int, list[float]]]], Awaitable[int]],
    progress: Optional[Callable[[int, int], None]] = None,
) -> IngestionResult:
    """Ingest a document: chunk → embed → upload + store metadata.

    upload_fn receives batches of (chunk_id, embedding) pairs and returns
    the number of vectors successfully stored.
    """
    from chunker import split_text  # local import to match original crate boundary

    raw_chunks = split_text(text, doc_name, cfg.chunk_cfg, cfg.doc_schema)
    chunks = _apply_doc_meta(raw_chunks, doc_name, cfg.doc_meta)
    total = len(chunks)
    logger.info("ingesting document doc=%s chunks=%d", doc_name, total)

    if not chunks:
        return IngestionResult(chunks_ingested=0, vectors_uploaded=0)

    metas: list = []
    ids: list[int] = []
    total_uploaded = 0
    ingested_at = meta.now_rfc3339()

    for batch_start in range(0, total, cfg.upload_batch_size):
        batch = chunks[batch_start: batch_start + cfg.upload_batch_size]

        embed_strings = [
            f"[{c.section_name}] {c.text}" if c.section_name else c.text
            for c in batch
        ]
        embeddings = await cfg.embed.embed_batch(embed_strings)

        vectors = [(c.id, emb) for c, emb in zip(batch, embeddings)]
        uploaded = await upload_fn(vectors)
        total_uploaded += uploaded

        for c in batch:
            metas.append({
                "doc_name": c.doc_name,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "surrounding": c.surrounding,
                "page_num": c.page_num,
                "ingested_at": ingested_at,
                "section_name": c.section_name,
                "skip_extraction": c.skip_extraction,
                "section_note": c.section_note,
                "section_type": c.section_type,
            })
            ids.append(c.id)

        if progress:
            progress(len(ids), total)
        logger.debug("batch uploaded done=%d total=%d", len(ids), total)

    meta.put_chunks(metas, ids)

    if cfg.graph:
        await _extract_and_store_entities(chunks, ids, cfg.embed, cfg.graph)

    return IngestionResult(chunks_ingested=total, vectors_uploaded=total_uploaded)


# ── Public graph-build entry point ───────────────────────────────────────────

async def extract_and_store_entities_pub(
    chunks: list,
    chunk_ids: list[int],
    embed: Any,
    graph_cfg: GraphIngestConfig,
    progress: Optional[Callable[[int, int, int, int], None]] = None,
) -> None:
    """Extract entities from all chunks and persist to the GraphStore.

    When graph_cfg.workers > 1, chunks are dispatched concurrently across all
    effective inference URLs (round-robin). The GraphStore is only written by a
    single drain task.

    progress(chunks_done, total_chunks, entity_count, relation_count)
    """
    if graph_cfg.entity_centric:
        await _extract_entity_centric(chunks, chunk_ids, embed, graph_cfg, progress)
        return

    if graph_cfg.ec_refine_only:
        print("  EC refine-only: skipping CC extraction, re-scoring existing entities")
        store = graph_cfg.store
        async with store.lock:
            store.sync_evidence()
            try:
                store.score_all_confidences()
            except Exception as e:
                logger.warning("confidence scoring failed: %s", e)
        if graph_cfg.ec_refine_threshold > 0.0:
            await _refine_low_confidence_entities(chunks, chunk_ids, embed, graph_cfg)
        return

    total = len(chunks)
    urls = graph_cfg.effective_urls()
    url_counter = 0
    url_lock = asyncio.Lock()
    workers = max(graph_cfg.workers, 1)
    chunk_batch = max(graph_cfg.chunk_batch, 1)
    sem = asyncio.Semaphore(workers)
    queue: asyncio.Queue[_ChunkResult] = asyncio.Queue(maxsize=workers * 4)

    # Snapshot Person entity genders for pronoun resolution.
    async with graph_cfg.store.lock:
        gender_context = [
            (e.name, e.gender)
            for e in graph_cfg.store.all_entities()
            if e.entity_type == "Person"
        ]

    async def _next_url() -> str:
        nonlocal url_counter
        async with url_lock:
            idx = url_counter % len(urls)
            url_counter += 1
        return urls[idx]

    # Drain task: writes results to graph sequentially.
    async def drain() -> None:
        done = 0
        while True:
            res = await queue.get()
            if res is None:  # sentinel
                break
            done += 1
            if not res.entities:
                if progress:
                    async with graph_cfg.store.lock:
                        nc = graph_cfg.store.node_count()
                        rc = graph_cfg.store.relation_count()
                    progress(done, total, nc, rc)
                continue

            async with graph_cfg.store.lock:
                graph = graph_cfg.store
                entity_ids_for_chunk = []
                for extracted, emb in zip(res.entities, res.embeddings):
                    clean = clean_extracted_name(extracted.name)
                    if clean is None:
                        continue
                    eid = _entity_id(clean, extracted.entity_type)
                    fields = {
                        k: _field_value(v, res.chunk_id)
                        for k, v in extracted.fields.items()
                        if v
                    }
                    description = _description_from_fields(clean, extracted.entity_type, fields)
                    if not description:
                        description = extracted.description
                    node = _make_entity_node(
                        eid, clean, extracted.entity_type, description, emb,
                        res.chunk_id, fields,
                    )
                    try:
                        graph.upsert_entity(node)
                        entity_ids_for_chunk.append(eid)
                    except Exception as e:
                        logger.warning("upsert_entity: %s", e)

                for rel in res.relations:
                    src = _resolve_entity_id(rel.from_, res.entities, graph)
                    dst = _resolve_entity_id(rel.to, res.entities, graph)
                    try:
                        graph.upsert_relation(src, dst, rel.relation, res.chunk_id)
                    except Exception as e:
                        logger.warning("upsert_relation: %s", e)

                try:
                    graph.link_chunk(res.chunk_id, entity_ids_for_chunk)
                except Exception as e:
                    logger.warning("link_chunk: %s", e)

                if progress:
                    progress(done, total, graph.node_count(), graph.relation_count())

    drain_task = asyncio.create_task(drain())

    async def process_chunk_batch(i: int) -> None:
        batch_end = min(i + chunk_batch, total)
        chunk = chunks[i]
        chunk_id = chunk_ids[i]

        if chunk.skip_extraction:
            logger.debug("skipping extraction for flagged section chunk_id=%d", chunk_id)
            await queue.put(_ChunkResult(chunk_id=chunk_id, entities=[], relations=[], embeddings=[]))
            return

        if graph_cfg.context_window > 0 or chunk_batch > 1:
            start = max(0, i - graph_cfg.context_window)
            end = min(batch_end + graph_cfg.context_window, total)
            center_type = chunk.section_type
            text = "\n\n[...]\n\n".join(
                c.text for c in chunks[start:end]
                if center_type.same_window_zone(c.section_type)
            )
        else:
            text = chunk.text

        url = await _next_url()
        from ner import extract_proper_noun_candidates, resolve_pronouns
        from graph import extract_from_text

        gliner_hints: list[str] = []
        if graph_cfg.gliner_client:
            gliner_hints = await graph_cfg.gliner_client.person_spans(text)

        candidates = extract_proper_noun_candidates(text)
        for span in gliner_hints:
            if span not in candidates:
                candidates.append(span)

        pronoun_map = resolve_pronouns(text, gender_context, gliner_hints)
        for _, name in pronoun_map:
            if name not in candidates:
                candidates.append(name)

        hints_opt = gliner_hints if gliner_hints else None
        et = [t for t in graph_cfg.entity_types]

        try:
            entities, relations = await extract_from_text(
                text, candidates, pronoun_map, chunk.section_note,
                url, graph_cfg.model, et, graph_cfg.no_relations, hints_opt,
            )
        except Exception as e:
            logger.warning("entity extraction error for chunk %d: %s", chunk_id, e)
            await queue.put(_ChunkResult(chunk_id=chunk_id, entities=[], relations=[], embeddings=[]))
            return

        if et:
            entities = [e for e in entities if any(t.lower() == e.entity_type.lower() for t in et)]

        if not entities:
            embeddings = []
        else:
            texts = [_entity_embed_text(e, chunk_id) for e in entities]
            try:
                embeddings = await embed.embed_batch(texts)
            except Exception as e:
                logger.warning("entity embedding error for chunk %d: %s", chunk_id, e)
                embeddings = []

        await queue.put(_ChunkResult(
            chunk_id=chunk_id,
            entities=entities,
            relations=relations,
            embeddings=embeddings,
        ))

    # Spawn tasks with semaphore throttling.
    tasks = []
    i = 0
    while i < total:
        async with sem:
            task = asyncio.create_task(process_chunk_batch(i))
            tasks.append(task)
        i += chunk_batch

    await asyncio.gather(*tasks)
    await queue.put(None)  # close drain
    await drain_task

    async with graph_cfg.store.lock:
        graph_cfg.store.sync_evidence()
        try:
            graph_cfg.store.score_all_confidences()
        except Exception as e:
            logger.warning("confidence scoring failed: %s", e)

    if graph_cfg.ec_refine_threshold > 0.0:
        await _refine_low_confidence_entities(chunks, chunk_ids, embed, graph_cfg)


# ── Private: sequential graph extraction ─────────────────────────────────────

async def _extract_and_store_entities(
    chunks: list,
    chunk_ids: list[int],
    embed: Any,
    graph_cfg: GraphIngestConfig,
) -> None:
    """Sequential extraction — used during ingest_text (no progress callback)."""
    from ner import extract_proper_noun_candidates
    from graph import extract_from_text

    total = len(chunks)
    cw = graph_cfg.context_window

    for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
        if chunk.skip_extraction:
            logger.debug("skipping extraction chunk_id=%d section=%s", chunk_id, chunk.section_name)
            continue

        if cw > 0:
            start = max(0, i - cw)
            end = min(i + cw + 1, total)
            center_type = chunk.section_type
            text = "\n\n[...]\n\n".join(
                c.text for c in chunks[start:end]
                if center_type.same_window_zone(c.section_type)
            )
        else:
            text = chunk.text

        et = list(graph_cfg.entity_types)
        candidates = extract_proper_noun_candidates(text)

        gliner_hints: list[str] = []
        if graph_cfg.gliner_client:
            gliner_hints = await graph_cfg.gliner_client.person_spans(text)

        hints_opt = gliner_hints if gliner_hints else None

        try:
            entities, relations = await extract_from_text(
                text, candidates, [], chunk.section_note,
                graph_cfg.inference_url, graph_cfg.model, et,
                graph_cfg.no_relations, hints_opt,
            )
        except Exception as e:
            logger.warning("entity extraction error for chunk %d: %s", chunk_id, e)
            continue

        if et:
            entities = [e for e in entities if any(t.lower() == e.entity_type.lower() for t in et)]

        if not entities:
            continue

        texts = [_entity_embed_text(e, chunk_id) for e in entities]
        try:
            embeddings = await embed.embed_batch(texts)
        except Exception as e:
            logger.warning("entity embedding error for chunk %d: %s", chunk_id, e)
            continue

        async with graph_cfg.store.lock:
            graph = graph_cfg.store
            entity_ids_for_chunk = []
            for extracted, emb in zip(entities, embeddings):
                eid = _entity_id(extracted.name, extracted.entity_type)
                fields = {
                    k: _field_value(v, chunk_id)
                    for k, v in extracted.fields.items()
                    if v
                }
                description = _description_from_fields(extracted.name, extracted.entity_type, fields)
                if not description:
                    description = extracted.description
                node = _make_entity_node(
                    eid, extracted.name, extracted.entity_type, description, emb, chunk_id, fields,
                )
                try:
                    graph.upsert_entity(node)
                    entity_ids_for_chunk.append(eid)
                except Exception as e:
                    logger.warning("upsert_entity failed: %s", e)

            for rel in relations:
                src = _resolve_entity_id(rel.from_, entities, graph)
                dst = _resolve_entity_id(rel.to, entities, graph)
                try:
                    graph.upsert_relation(src, dst, rel.relation, chunk_id)
                except Exception as e:
                    logger.warning("upsert_relation failed: %s", e)

            try:
                graph.link_chunk(chunk_id, entity_ids_for_chunk)
            except Exception as e:
                logger.warning("link_chunk failed: %s", e)

            logger.debug("graph updated chunk_id=%d entities=%d relations=%d",
                         chunk_id, len(entity_ids_for_chunk), len(relations))


# ── Entity-centric extraction ─────────────────────────────────────────────────

def _window_text(chunks: list, center: int, window: int) -> str:
    if window == 0:
        return chunks[center].text
    start = max(0, center - window)
    end = min(center + window + 1, len(chunks))
    center_type = chunks[center].section_type
    return "\n\n[...]\n\n".join(
        c.text for c in chunks[start:end]
        if center_type.same_window_zone(c.section_type)
    )


async def _extract_entity_centric(
    chunks: list,
    chunk_ids: list[int],
    embed: Any,
    graph_cfg: GraphIngestConfig,
    progress: Optional[Callable[[int, int, int, int], None]],
) -> None:
    from graph import extract_from_text

    MAX_SOURCE_CHUNKS = 3

    gliner = graph_cfg.gliner_client
    if gliner is None:
        logger.warning("--entity-centric requires --gliner-url; aborting entity-centric run")
        return

    total = len(chunks)
    cw = graph_cfg.context_window

    # Phase 1: GLiNER scan — build entity_name → [chunk_indices]
    entity_to_chunks: dict[str, list[int]] = defaultdict(list)
    for i in range(total):
        text = _window_text(chunks, i, cw)
        for span in await gliner.person_spans(text):
            entity_to_chunks[span].append(i)

    unique_names = list(entity_to_chunks.items())
    n_unique = len(unique_names)
    logger.info("entity-centric phase 1: %d unique spans in %d chunks", n_unique, total)
    print(f"  EC phase 1: {n_unique} unique GLiNER spans → one LLM call each")

    # Phase 2: per-entity LLM calls
    urls = graph_cfg.effective_urls()
    url_counter = 0
    url_lock = asyncio.Lock()
    workers = max(graph_cfg.workers, 1)
    no_relations = graph_cfg.no_relations
    et = list(graph_cfg.entity_types)
    sem = asyncio.Semaphore(workers)
    done_ctr = 0
    llm_calls = 0
    context_chars = 0

    async def _next_url() -> str:
        nonlocal url_counter
        async with url_lock:
            idx = url_counter % len(urls)
            url_counter += 1
        return urls[idx]

    async def process_entity(entity_name: str, source_indices: list[int]) -> None:
        nonlocal done_ctr, llm_calls, context_chars

        # Aggregate up to MAX_SOURCE_CHUNKS distinct windows.
        seen: set[int] = set()
        for ci in source_indices[:MAX_SOURCE_CHUNKS]:
            start = max(0, ci - cw)
            end = min(ci + cw + 1, total)
            seen.update(range(start, end))

        context = "\n\n[...]\n\n".join(chunks[ci].text for ci in sorted(seen))
        ctx_len = len(context)

        url = await _next_url()
        llm_calls += 1
        context_chars += ctx_len

        candidates = [entity_name]
        hints = [entity_name]
        et_refs = et

        try:
            entities, _ = await extract_from_text(
                context, candidates, [], None, url, graph_cfg.model,
                et_refs, no_relations, hints,
            )
        except Exception as e:
            logger.warning("ec extract error for '%s': %s", entity_name, e)
            return

        texts = [f"{e.name}: {e.description}" for e in entities]
        if not texts:
            return
        try:
            embeddings = await embed.embed_batch(texts)
        except Exception as e:
            logger.warning("ec embed error: %s", e)
            embeddings = []
        entities = entities[:len(embeddings)]

        async with graph_cfg.store.lock:
            graph = graph_cfg.store
            for extracted, emb in zip(entities, embeddings):
                clean = clean_extracted_name(extracted.name)
                if clean is None:
                    continue
                eid = _entity_id(clean, extracted.entity_type)
                fields = {
                    k: {"value": v, "evidence_chunk_ids": [], "confidence": 1.0}
                    for k, v in extracted.fields.items()
                }
                description = _description_from_fields(clean, extracted.entity_type, fields)
                if not description:
                    description = extracted.description
                node = _make_entity_node(eid, clean, extracted.entity_type, description, emb, 0, fields)
                try:
                    graph.upsert_entity(node)
                except Exception as e:
                    logger.warning("ec upsert: %s", e)

            done_ctr += 1
            if progress:
                progress(done_ctr, n_unique, graph.node_count(), graph.relation_count())

    tasks = []
    for entity_name, source_indices in unique_names:
        async with sem:
            tasks.append(asyncio.create_task(process_entity(entity_name, source_indices)))

    await asyncio.gather(*tasks)

    avg_ctx = context_chars // llm_calls if llm_calls else 0
    async with graph_cfg.store.lock:
        entity_count = graph_cfg.store.node_count()
    logger.info("entity-centric complete: %d calls, %d avg ctx chars, %d entities",
                llm_calls, avg_ctx, entity_count)
    print(f"  EC metrics: {llm_calls} LLM calls  |  {avg_ctx} avg context chars  |  {entity_count} entities")


# ── EC Refinement pass ────────────────────────────────────────────────────────

async def _refine_low_confidence_entities(
    chunks: list,
    chunk_ids: list[int],
    embed: Any,
    cfg: GraphIngestConfig,
) -> None:
    from graph import extract_from_text

    MAX_SOURCE_CHUNKS = 3

    id_to_index = {cid: i for i, cid in enumerate(chunk_ids)}
    total = len(chunks)
    cw = cfg.context_window
    threshold = cfg.ec_refine_threshold
    budget = max(cfg.ec_refine_budget, 1)
    et = list(cfg.entity_types)

    # Collect low-confidence targets.
    async with cfg.store.lock:
        candidates_raw = [
            (n.id, n.name, n.entity_type, list(n.evidence), n.confidence)
            for n in cfg.store.all_entities()
            if n.confidence < threshold and (
                not et or any(t.lower() == n.entity_type.lower() for t in et)
            )
        ]
    candidates_raw.sort(key=lambda x: x[4])
    targets = candidates_raw[:budget]

    if not targets:
        print(f"  EC refinement: 0 entities below threshold {threshold:.2f}")
        return

    print(
        f"  EC refinement: {len(targets)} entities below threshold {threshold:.2f} "
        f"→ escalating (budget={budget})"
    )

    urls = cfg.effective_urls()
    url_counter = 0
    url_lock = asyncio.Lock()
    workers = max(cfg.workers, 1)
    sem = asyncio.Semaphore(workers)

    async def _next_url() -> str:
        nonlocal url_counter
        async with url_lock:
            idx = url_counter % len(urls)
            url_counter += 1
        return urls[idx]

    async with cfg.store.lock:
        initial_entity_count = cfg.store.node_count()

    improved = 0
    confidence_delta_sum = 0.0

    for target_id, entity_name, _, evidence, old_conf in targets:
        source_indices = [id_to_index[cid] for cid in evidence if cid in id_to_index]
        if not source_indices:
            continue

        seen: set[int] = set()
        for ci in source_indices[:MAX_SOURCE_CHUNKS]:
            start = max(0, ci - cw)
            end = min(ci + cw + 1, total)
            seen.update(range(start, end))

        context = "\n\n[...]\n\n".join(chunks[ci].text for ci in sorted(seen))

        url = await _next_url()
        et_refs = et

        try:
            async with sem:
                entities, _ = await extract_from_text(
                    context, [entity_name], [], None, url, cfg.model,
                    et_refs, True, [entity_name],  # no_relations=True for field enrichment
                )
        except Exception as e:
            logger.warning("EC refinement error for '%s': %s", entity_name, e)
            continue

        entities = [
            e for e in entities
            if not et or any(t.lower() == e.entity_type.lower() for t in et)
        ]

        for extracted in entities:
            clean = clean_extracted_name(extracted.name)
            if clean is None:
                continue
            eid = _entity_id(clean, extracted.entity_type)
            embed_text = f"{clean}: {extracted.description}"
            try:
                embedding = (await embed.embed_batch([embed_text]))[0]
            except Exception as e:
                logger.warning("EC refinement embed error: %s", e)
                continue

            fields = {
                k: {"value": v, "evidence_chunk_ids": [], "confidence": 1.0}
                for k, v in extracted.fields.items()
            }
            description = _description_from_fields(clean, extracted.entity_type, fields)
            if not description:
                description = extracted.description

            node = _make_entity_node(eid, clean, extracted.entity_type, description, embedding, 0, fields)
            async with cfg.store.lock:
                try:
                    cfg.store.upsert_entity(node)
                except Exception as e:
                    logger.warning("EC refinement upsert error: %s", e)

        async with cfg.store.lock:
            new_conf = cfg.store.rescore_entity(target_id)

        if new_conf > old_conf + 0.01:
            improved += 1
            confidence_delta_sum += new_conf - old_conf

    async with cfg.store.lock:
        try:
            cfg.store.score_all_confidences()
        except Exception as e:
            logger.warning("EC refinement confidence persist failed: %s", e)
        final_entity_count = cfg.store.node_count()

    new_entities = max(0, final_entity_count - initial_entity_count)
    avg_delta = confidence_delta_sum / improved if improved else 0.0
    print(
        f"  EC refinement done: {improved}/{len(targets)} existing entities improved "
        f"(avg confidence ↑ +{avg_delta:.2f}), {new_entities} new entities discovered"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_entity_id(name: str, current_entities: list, graph: Any) -> int:
    """Resolve a relation endpoint to an entity ID.

    Priority: (1) current-chunk extraction, (2) existing graph node by name,
    (3) Unknown fallback.
    """
    for e in current_entities:
        if e.name.lower() == name.lower():
            return _entity_id(e.name, e.entity_type)
    node = graph.find_by_name(name)
    if node:
        return node.id
    return _entity_id(name, "Unknown")


def _apply_doc_meta(chunks: list, doc_name: str, doc_meta: dict[str, str]) -> list:
    """Prepend doc-level metadata prefix to each chunk's text."""
    if not doc_meta:
        return chunks
    doc_lower = doc_name.lower()
    prefix = next(
        (v for k, v in doc_meta.items() if k.lower() in doc_lower),
        None,
    )
    if prefix:
        for c in chunks:
            c.text = f"{prefix}\n\n{c.text}"
    return chunks


def _entity_embed_text(entity: Any, chunk_id: int) -> str:
    from graph import description_from_fields, FieldValue
    if entity.fields:
        fv_map = {
            k: FieldValue(value=v, evidence_chunk_ids=[chunk_id], confidence=1.0)
            for k, v in entity.fields.items()
            if v
        }
        desc = description_from_fields(entity.name, entity.entity_type, fv_map)
        if not desc:
            desc = entity.description
    else:
        desc = entity.description
    return f"{entity.name}: {desc}" if desc else entity.name


def _entity_id(name: str, entity_type: str) -> int:
    from graph import entity_id
    return entity_id(name, entity_type)


def _field_value(v: str, chunk_id: int) -> Any:
    from graph import FieldValue
    return FieldValue(value=v, evidence_chunk_ids=[chunk_id], confidence=1.0)


def _description_from_fields(name: str, entity_type: str, fields: dict) -> str:
    from graph import description_from_fields
    return description_from_fields(name, entity_type, fields)


def _make_entity_node(eid, name, entity_type, description, embedding, chunk_id, fields) -> Any:
    from graph import EntityNode
    return EntityNode(
        id=eid,
        name=name,
        entity_type=entity_type,
        description=description,
        embedding=embedding,
        mention_count=1,
        first_chunk_id=chunk_id,
        aliases=[],
        schema_type=None,
        evidence=[],
        gender=None,
        fields=fields,
        confidence=0.0,
    )


# ── Entity name filter ────────────────────────────────────────────────────────

_GENERIC_ROLE_BLOCKLIST = {
    "granny", "gran", "grandma", "grandfather", "grandpa", "gramps",
    "dad", "daddy", "father", "mother", "mom", "mum", "mama",
    "uncle", "auntie", "aunt", "cousin", "son", "daughter",
    "me", "i", "he", "she", "they", "we",
    "the narrator", "the author", "narrator", "author",
    "herrenvolk", "herrenvolkism", "apartheid",
    "coloured", "coloureds", "blacks", "whites", "white", "black",
    "indians", "africans", "europeans", "non-white", "non-whites",
    "non-european", "cape malay", "cape malay_indian", "pathan", "pathans",
    "xhosa", "slavic", "hungarian", "jewish", "aryan", "moslem", "muslim",
    "nationalist", "nationalists", "german", "french", "russian", "british",
    "english", "african", "indian", "arab", "arabs", "chinese", "boer",
    "bantu", "coolie", "coolies", "malay", "malays", "griqua", "hindu",
    "hindus", "irish", "japanese", "norwegian", "sikh", "turks", "zulus",
    "afrikaner", "afrikaners", "west indians", "south african", "cape coloured",
    "non-white muslim south africans", "socialist", "marxist", "labour",
    "communist", "fascist", "nazi", "nats", "native",
    "christmas", "eid", "eid mubarak", "islam", "ramadan", "victorian",
    "history", "science", "schooling", "mother tongue",
    "everything", "something", "nothing", "anything",
    "there", "here", "this", "that", "these", "those",
    "each", "every", "all", "none", "some", "any", "both", "one", "many",
    "such", "how", "when", "moreover", "sometime", "alas", "half", "apart",
    "being", "blot", "do", "everyone", "figure", "found", "great", "had",
    "hatless", "just", "later", "little", "much", "needless", "next", "now",
    "ob", "perh", "perhaps", "peru", "piccadilly", "regrettably", "several",
    "shyly", "soon", "still", "tell", "theoretically", "v1", "va", "whether",
    "wo", "worse", "poor abdul", "flash", "dandy", "lobo", "baby", "youth",
    "legless", "muddy", "polly", "tiny", "vic", "bill", "solly", "nina",
    "kismets", "zoology", "cadbury", "freubel",
    "south african indian", "head of british muslims", "non-white councillors",
    "prof", "prof.", "prof_", "gools", "rassools", "goldings", "killers",
    "stranglers", "royal family", "mr.", "mr_", "rev.", "rev_", "dr.", "dr_",
    "god", "allah", "lord", "devil", "fate", "nature", "y_allah", "y allah",
    "hadji", "haji", "hajj", "maulvi", "molvi", "imam", "sheikh",
    "black maria", "homer", "longfellow", "wordsworth", "robert browning",
    "robert louis stevenson", "john milton", "mark twain", "charles dickens",
    "shakespeare", "william shakespeare", "bernard shaw", "shaw", "chekov",
    "chekhov", "dostoevsky", "gogol", "gorki", "emile zola", "sinclair lewis",
    "steinbeck", "jack london", "damon runyon", "tarzan", "buck rogers",
    "buck jones", "hopalong cassidy", "roy rogers", "gene autry", "bob steele",
    "cobra woman", "brick bradford", "globi", "ali baba", "tsotsi", "banquo",
    "mephistopheles", "dorian gray", "pharaoh cheops", "hunchback of notre dame",
    "goofy", "captain america", "captain marvel", "captain britain",
    "superman", "batman", "spiderman", "spider-man", "hamlet", "cassandra",
    "mommy", "mummy", "then", "tb", "cac", "gandhian", "berlin hitler",
    "mom ayesha", "european native coloured indian malay griqua", "lot",
}

_ROLE_PREFIXES = (
    "uncle ", "auntie ", "aunt ", "granny ", "gran ", "grandpa ",
    "grandma ", "grandfather ", "grandmother ", "sis ", "boeta ", "boetie ",
)

_SENTENCE_STARTERS = {
    "when", "where", "while", "that", "this", "those", "these", "what",
    "which", "who", "whom", "whose", "how", "why", "if", "although",
    "because", "since", "after", "before", "as", "and", "but", "or",
    "nor", "so", "yet", "for", "the", "a", "an",
}

_TRAILING_JUNK = {
    "please", "thank", "thanks", "yes", "no", "too", "also", "only",
    "said", "asked", "replied", "told", "wrote", "was", "is", "are",
    "the", "a", "an", "and", "but", "or", "for", "to", "of", "in",
    "on", "at", "with", "from", "by", "as", "his", "her", "their",
}


def clean_extracted_name(raw: str) -> Optional[str]:
    """Filter and normalise an extracted entity name.

    Returns None if the name should be discarded (blocklist hit, role prefix,
    sentence starter, empty after normalisation). Returns the cleaned name otherwise.
    """
    name_lc = raw.strip().lower()
    if name_lc in _GENERIC_ROLE_BLOCKLIST:
        return None

    words = name_lc.split()
    word_count = len(words)

    if word_count > 7:
        return None
    if word_count <= 3 and any(name_lc.startswith(p) for p in _ROLE_PREFIXES):
        return None
    if words and words[0] in _SENTENCE_STARTERS:
        return None

    # OCR underscore normalisation: _Word_ → (Word), _s → 's, M_ → M., else strip
    normalised = _normalise_underscores(raw)

    # Strip possessives
    clean = normalised.rstrip("'s").rstrip("\u2019").rstrip("s'").strip()

    # Strip trailing junk words
    while True:
        w = clean.split()
        if len(w) <= 1:
            break
        if w[-1].lower() in _TRAILING_JUNK:
            clean = clean[: len(clean) - len(w[-1])].rstrip()
        else:
            break

    return clean if clean else None


def _normalise_underscores(raw: str) -> str:
    # First pass: _Word_ → (Word) for parenthetical aliases
    result = raw
    while True:
        b = result
        match = re.search(r'(?<= )_([^_]+)_(?= |$)', result)
        if match:
            result = result[: match.start()] + f"({match.group(1)})" + result[match.end():]
        if result == b:
            break

    chars = list(result)
    n = len(chars)
    out = []
    i = 0
    while i < n:
        c = chars[i]
        if c != "_":
            out.append(c)
            i += 1
            continue

        # _s → 's
        if i + 1 < n and chars[i + 1] == "s":
            after = i + 2
            if after >= n or not chars[after].isalpha() or chars[after].isupper():
                out.append("'")
                out.append("s")
                i += 2
                continue

        # _ preceded by alpha and followed by space/end/uppercase → .
        prev_alpha = bool(out) and out[-1].isalpha()
        next_ch = chars[i + 1] if i + 1 < n else None
        next_break = next_ch is None or next_ch == " " or (next_ch.isalpha() and next_ch.isupper())
        if prev_alpha and next_break:
            out.append(".")
        i += 1

    return " ".join("".join(out).split())
