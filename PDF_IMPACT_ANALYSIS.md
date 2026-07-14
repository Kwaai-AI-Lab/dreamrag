# PDF Extraction Impact Analysis

## The Opportunity

By enabling PDF extraction with pdfplumber, we can unlock content from 6 currently zero-performing topics that are **entirely composed of PDFs**.

### Zero-Recall Topics (All PDFs)

These 6 topics currently achieve 0% Recall@10 and 0.0 MRR because **all their documents are PDFs that weren't being indexed**:

| Topic | Doc Count | File Types | Questions |
|-------|-----------|-----------|-----------|
| Climate Science | 14 | All PDF | 20 |
| Internet Standard (RFCs) | 13 | All PDF | 20 |
| Legal Documents | 13 | All PDF | 20 |
| NIST AI Security & Governance | 16 | All PDF | 20 |
| Deep Sea Biology | 12 | All PDF | 20 |
| Dream-Based Memory Consolidation | 13 | All PDF | 20 |
| **TOTAL** | **81 PDFs** | | **120 questions** |

## Expected Impact

### Optimistic Scenario (Conservative Estimate)
- **Climate Science**: 0.0 → 0.30 MRR (30% improvement) — domain vocabulary match
- **RFCs**: 0.0 → 0.40 MRR — technical standards have distinctive terminology
- **Legal Documents**: 0.0 → 0.25 MRR — case names, dates are easily matched
- **NIST AI**: 0.0 → 0.35 MRR — governance documents have structured terminology
- **Deep Sea Biology**: 0.0 → 0.20 MRR — scientific papers with specific species/concepts
- **Dream Memory**: 0.0 → 0.25 MRR — neuroscience with specific terms

**Expected Overall Improvement**: 
- Before: 280 questions, 126 hits @ Recall@1 (45%), 280 @ MRR
- After: +60 hits from previously-zero topics
- New Recall@1: 26.8% → ~48% (+82%)
- New MRR: 0.316 → ~0.45 (+42%)

### Conservative Scenario
If PDF extraction is difficult/lossy, we might only improve by:
- 50% of the above gains: MRR 0.316 → ~0.38 (+24%)

### Realistic Scenario
Based on typical PDF extraction challenges:
- Some PDFs are scanned images (OCR would help but we don't have it)
- Some PDFs have complex formatting that confuses extractors
- Most scientific/legal PDFs should extract reasonably well

**Expected outcome**: MRR 0.316 → ~0.42 (+33%)

---

## Document Breakdown

### Currently Indexed (Before) — 85 documents

```
✓ TXT:   38 documents (War & Peace, Python Docs, RFCs text, historical docs)
✓ HTML:  32 documents (Wikipedia, OSM wiki, poetry collections)
✓ VTT:   15 documents (Meeting transcripts)
✓ JSON:   1 document (NIST atlas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   86 total

✗ PDF: 139 documents (SKIPPED — no pdfplumber)
```

### After PDF Extraction (Expected) — 195+ documents

```
✓ TXT:   38 documents
✓ HTML:  32 documents
✓ VTT:   15 documents
✓ JSON:   1 document
✓ PDF:  ~130-139 documents (newly indexed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   216-225 total (96-100% of corpus)
```

---

## Why This Matters for DreamRAG

### 1. **Corpus Completeness**
- Before: Only 38% of documents indexed
- After: 96-100% of documents indexed
- Evaluation becomes more realistic

### 2. **Domain Coverage**
- 6 zero-performing topics (40% of test set) become evaluable
- Exposes where DreamRAG's semantic/graph approaches outperform lexical baselines
- Tests graph refinement on structured (legal, technical) vs. unstructured (literature, science) domains

### 3. **Realistic Baseline**
- BM25 on complete corpus is fairer comparison point for hybrid/semantic approaches
- Currently artificially penalized by missing PDF content
- After: BM25 becomes proper baseline for evaluating improvements from embeddings, graph-based retrieval, etc.

### 4. **Path to Optimization**
- **If MRR improves to 0.42+**: Shows PDFs are valuable but still underperforming → reranking, embeddings, or graph-based approaches could gain 20-40%
- **If MRR stays near 0.32**: Shows vocabulary mismatch is real problem → motivates entity extraction, synonym expansion, dense retrieval as solutions

---

## Technical Challenges Expected

### 1. **Slow PDF Processing**
- 139 PDFs will take 3-10 minutes to extract
- Solution: Limit to first 5 pages per PDF, skip problematic files gracefully

### 2. **Poor Text Extraction from Scanned PDFs**
- Some scientific papers are scanned (image-based)
- OCR needed but not available
- Impact: ~10-15% of PDFs may yield empty/useless text
- Mitigation: Skip gracefully, still index 85-90% of PDFs

### 3. **Complex Formatting Loss**
- Tables, equations, multi-column layouts lose structure
- Trade-off: Lexical retrieval still works on extracted text
- Impact: Minor (maybe 5-10% MRR loss on highly formatted docs)

### 4. **Memory/Performance**
- 225 documents × BM25 indexing = manageable (~200MB RAM)
- Retrieval may slow from ~100ms to ~200-300ms per query
- Acceptable for evaluation purposes

---

## Success Metrics

After re-running with PDF extraction enabled:

| Metric | Before | Expected After | Success Threshold |
|--------|--------|-----------------|-------------------|
| Total Docs Indexed | 85 | 200+ | ✓ |
| Docs Indexed (%) | 38% | 90%+ | ✓ |
| Zero-Recall Topics | 6 | 0-1 | ✓ |
| Recall@1 | 26.8% | 35-50% | ✓ if >35% |
| Recall@10 | 42.5% | 55-70% | ✓ if >50% |
| MRR | 0.316 | 0.40-0.45 | ✓ if >0.38 |

---

## Next: Dense Retrieval + Hybrid Fusion

Once PDFs are indexed, the next step is implementing hybrid retrieval:

1. **Embeddings-based retrieval** (e.g., sentence-transformers)
   - Handles synonym, semantic similarity, paraphrasing
   - Especially helps zero-recall topics (vocabulary mismatch)
   - Expected gain: +10-20% MRR

2. **Reciprocal Rank Fusion (RRF)**
   - Combines BM25 + embeddings rankings
   - No machine learning needed
   - Expected gain: +5-15% over either alone

3. **Cross-Encoder Reranking**
   - Reranks top-100 BM25 results using semantic scoring
   - Expected gain: +5-10% MRR

This would push expected performance to **0.55-0.65 MRR** on the complete corpus.
