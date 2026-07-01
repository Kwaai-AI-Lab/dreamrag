# Security Evaluation — Future Work

## Purpose

This document outlines future evaluation directions for assessing Dream RAG under adversarial or corrupted knowledge conditions.

This evaluation category is outside the current retrieval-quality and graph-refinement scope.

## Motivation

Dream RAG continuously refines graph memory. This creates opportunities to evaluate whether the system can resist, detect, or recover from corrupted knowledge.

## Candidate Evaluation Dimensions

- Knowledge Integrity
- Poison Persistence
- Recovery Rate
- Recovery Latency
- Attack Success Rate

## Potential Research Questions

- Can poisoned entities or relations persist across dream cycles?
- Can dream-cycle refinement reduce or remove corrupted knowledge?
- How does graph poisoning affect retrieval quality?

## Deferred Scope

This work is intentionally deferred because adversarial evaluation introduces a separate security research question.
