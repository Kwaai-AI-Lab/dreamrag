"""
dream_cycle.py — Iterative graph refinement via simulated "sleep-like" consolidation.

Implements periodic refinement cycles that:
1. Update strength for all nodes/edges using forgetting curve
2. Classify into states (long_term, short_term, dormant)
3. Prune weak facts below thresholds
4. Consolidate/deduplicate similar entities
5. Track metrics for graph evolution

This is where the memory.py forgetting curve becomes actionable.
"""

from __future__ import annotations

import sqlite3
import json
import math
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from memory import MemoryParams, strength, classify, elapsed_days, LONG_TERM, SHORT_TERM, DORMANT


@dataclass
class DreamCycleMetrics:
    """Metrics collected during a dream cycle."""
    cycle_num: int
    timestamp: datetime

    # Entity metrics
    total_entities_before: int
    total_entities_after: int
    entities_pruned: int
    entities_consolidated: int

    # Relation metrics
    total_relations_before: int
    total_relations_after: int
    relations_pruned: int

    # Strength distribution
    long_term_count: int      # strength >= 0.66
    short_term_count: int     # 0.33 <= strength < 0.66
    dormant_count: int        # strength < 0.33
    avg_strength: float

    # Stability
    graph_converged: bool
    compression_ratio: float  # (before - after) / before


class DreamCycle:
    """Manages iterative graph refinement with forgetting curve."""

    def __init__(
        self,
        db_path: str | Path,
        memory_params: MemoryParams = None,
    ):
        self.db_path = Path(db_path)
        self.memory_params = memory_params or MemoryParams()
        self.metrics_history: list[DreamCycleMetrics] = []
        self.cycle_count = 0

    def run_cycle(self, now: Optional[datetime] = None) -> DreamCycleMetrics:
        """Execute one dream cycle: strengthen, classify, prune, consolidate."""
        now = now or datetime.now(timezone.utc)
        cycle_num = self.cycle_count + 1

        print(f"\n{'='*80}")
        print(f"DREAM CYCLE {cycle_num}")
        print(f"{'='*80}")
        print(f"Timestamp: {now.isoformat()}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Phase 1: Load graph state
        print(f"\n[Phase 1] Loading graph state...")
        entities_before, relations_before = self._load_graph_counts(cursor)
        print(f"  Entities: {entities_before}")
        print(f"  Relations: {relations_before}")

        # Phase 2: Update strengths for all nodes
        print(f"\n[Phase 2] Updating entity strengths using forgetting curve...")
        self._update_entity_strengths(cursor, now)

        # Phase 3: Classify entities into states
        print(f"\n[Phase 3] Classifying entity states...")
        state_counts = self._classify_entities(cursor)
        print(f"  Long-term (≥0.66): {state_counts['long_term']}")
        print(f"  Short-term (0.33-0.66): {state_counts['short_term']}")
        print(f"  Dormant (<0.33): {state_counts['dormant']}")
        print(f"  Avg strength: {state_counts['avg_strength']:.3f}")

        # Phase 4: Prune weak/dormant entities
        print(f"\n[Phase 4] Pruning dormant entities...")
        pruned_count, relations_pruned = self._prune_dormant(cursor)
        print(f"  Pruned entities: {pruned_count}")
        print(f"  Pruned relations: {relations_pruned}")

        # Phase 5: Consolidate duplicates
        print(f"\n[Phase 5] Consolidating duplicate entities...")
        consolidated_count = self._consolidate_duplicates(cursor)
        print(f"  Consolidated: {consolidated_count}")

        # Phase 6: Commit and measure
        print(f"\n[Phase 6] Committing changes...")
        conn.commit()

        entities_after, relations_after = self._load_graph_counts(cursor)
        conn.close()

        # Compute metrics
        metrics = DreamCycleMetrics(
            cycle_num=cycle_num,
            timestamp=now,
            total_entities_before=entities_before,
            total_entities_after=entities_after,
            entities_pruned=pruned_count,
            entities_consolidated=consolidated_count,
            total_relations_before=relations_before,
            total_relations_after=relations_after,
            relations_pruned=relations_pruned,
            long_term_count=state_counts['long_term'],
            short_term_count=state_counts['short_term'],
            dormant_count=state_counts['dormant'],
            avg_strength=state_counts['avg_strength'],
            graph_converged=self._check_convergence(),
            compression_ratio=(entities_before - entities_after) / entities_before if entities_before > 0 else 0.0,
        )

        self.metrics_history.append(metrics)
        self.cycle_count += 1

        # Print summary
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle_num} SUMMARY")
        print(f"{'='*80}")
        print(f"Entities: {entities_before} → {entities_after} (Δ {entities_after - entities_before:+d}, {metrics.compression_ratio:.1%} compression)")
        print(f"Relations: {relations_before} → {relations_after} (Δ {relations_after - relations_before:+d})")
        print(f"State distribution: Long-term {state_counts['long_term']:3} | Short-term {state_counts['short_term']:3} | Dormant {state_counts['dormant']:3}")
        print(f"Graph converged: {metrics.graph_converged}")

        return metrics

    def _load_graph_counts(self, cursor) -> tuple[int, int]:
        """Get current entity and relation counts."""
        cursor.execute("SELECT COUNT(*) FROM entities;")
        entity_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM relations;")
        relation_count = cursor.fetchone()[0]

        return entity_count, relation_count

    def _update_entity_strengths(self, cursor, now: datetime) -> None:
        """Calculate and store strength for all entities based on forgetting curve."""
        # Get all entities
        cursor.execute("SELECT entity_id, node_json FROM entities;")
        rows = cursor.fetchall()

        updated = 0
        for entity_id, node_json in rows:
            try:
                node = json.loads(node_json)

                # Extract memory fields (with defaults)
                reinforcement = node.get('reinforcement', 1)
                last_seen_str = node.get('last_seen')

                # Parse last_seen timestamp
                if last_seen_str:
                    try:
                        last_seen = datetime.fromisoformat(last_seen_str)
                    except (ValueError, TypeError):
                        last_seen = now
                else:
                    last_seen = now

                # Calculate strength
                elapsed = elapsed_days(last_seen, now)
                strength_val = strength(reinforcement, elapsed, self.memory_params)

                # Update node
                node['strength'] = strength_val
                node['last_updated'] = now.isoformat()

                # Update database
                updated_json = json.dumps(node)
                cursor.execute(
                    "UPDATE entities SET node_json = ? WHERE entity_id = ?",
                    (updated_json, entity_id)
                )
                updated += 1

            except Exception as e:
                print(f"  Error updating entity {entity_id}: {e}")

        print(f"  Updated {updated} entities")

    def _classify_entities(self, cursor) -> dict:
        """Classify all entities into memory states and collect stats."""
        cursor.execute("SELECT entity_id, node_json FROM entities;")
        rows = cursor.fetchall()

        state_counts = {
            'long_term': 0,
            'short_term': 0,
            'dormant': 0,
        }
        strengths = []

        for entity_id, node_json in rows:
            try:
                node = json.loads(node_json)
                strength_val = node.get('strength', 1.0)
                strengths.append(strength_val)

                state = classify(strength_val, self.memory_params)
                state_counts[state] += 1

                # Update node with state
                node['memory_state'] = state
                updated_json = json.dumps(node)
                cursor.execute(
                    "UPDATE entities SET node_json = ? WHERE entity_id = ?",
                    (updated_json, entity_id)
                )

            except Exception as e:
                print(f"  Error classifying entity {entity_id}: {e}")

        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        state_counts['avg_strength'] = avg_strength

        return state_counts

    def _prune_dormant(self, cursor) -> tuple[int, int]:
        """Remove entities in dormant state (strength < 0.33) and their relations."""
        # Find dormant entities
        cursor.execute("SELECT entity_id, node_json FROM entities;")
        dormant_ids = []

        for entity_id, node_json in cursor.fetchall():
            try:
                node = json.loads(node_json)
                state = node.get('memory_state', DORMANT)
                if state == DORMANT:
                    dormant_ids.append(entity_id)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse entity {entity_id}: {e}")
                pass

        # Delete dormant entities
        pruned_count = 0
        for entity_id in dormant_ids:
            cursor.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
            pruned_count += 1

        # Delete relations involving pruned entities
        relations_pruned = 0
        for entity_id in dormant_ids:
            cursor.execute(
                "DELETE FROM relations WHERE src_id = ? OR dst_id = ?",
                (entity_id, entity_id)
            )
            relations_pruned += cursor.rowcount

        return pruned_count, relations_pruned

    def _consolidate_duplicates(self, cursor) -> int:
        """Merge similar entities (e.g., name variations)."""
        # Simple strategy: merge entities with similar names (fuzzy match)
        cursor.execute("SELECT entity_id, node_json FROM entities;")
        entities = {}

        for entity_id, node_json in cursor.fetchall():
            try:
                node = json.loads(node_json)
                entities[entity_id] = node
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse entity {entity_id}: {e}")

        # Find duplicates by fuzzy name matching
        consolidated = 0
        seen_names = {}

        for entity_id, node in entities.items():
            name = node.get('name', '').lower().strip()

            if not name or len(name) < 2:
                continue

            # Check if we've seen a similar name
            for existing_name, existing_id in list(seen_names.items()):
                if self._are_duplicate_names(name, existing_name):
                    # Merge: redirect relations, keep higher-strength entity
                    self._merge_entities(cursor, entity_id, existing_id)
                    consolidated += 1
                    break
            else:
                # New unique name
                seen_names[name] = entity_id

        return consolidated

    def _are_duplicate_names(self, name1: str, name2: str) -> bool:
        """Check if two names likely refer to same entity.

        Uses strict matching: exact match or last-name-only match.
        Avoids false positives from substring matching.
        """
        # Exact match (after normalization)
        if name1 == name2:
            return True

        # Last-name-only match (e.g., "Leslie Groves" vs "Groves")
        # Only if one is a single word and matches the last word of the other
        parts1 = name1.split()
        parts2 = name2.split()

        if len(parts1) > 1 and len(parts2) == 1:
            if parts1[-1] == parts2[0]:
                return True
        elif len(parts2) > 1 and len(parts1) == 1:
            if parts2[-1] == parts1[0]:
                return True

        # Conservative substring match: only if one is significantly contained
        # e.g., "Leslie Groves" vs "L. Groves" (initials), NOT "John" vs "John Smith"
        if len(name1) > 10 and len(name2) > 10:
            # Only match if names share most words (>80% word overlap)
            words1 = set(parts1)
            words2 = set(parts2)
            if words1 and words2:
                overlap = len(words1 & words2) / max(len(words1), len(words2))
                if overlap >= 0.8:
                    return True

        return False

    def _merge_entities(self, cursor, entity_id_weak: int, entity_id_strong: int) -> None:
        """Merge two entities, redirecting all relations to the stronger one."""
        # Get all relations from weak entity
        cursor.execute(
            "SELECT src_id, dst_id, relation_type FROM relations WHERE src_id = ? OR dst_id = ?",
            (entity_id_weak, entity_id_weak)
        )
        relations = cursor.fetchall()

        # Redirect each relation, handling conflicts
        for src_id, dst_id, rel_type in relations:
            new_src = entity_id_strong if src_id == entity_id_weak else src_id
            new_dst = entity_id_strong if dst_id == entity_id_weak else dst_id

            # Skip self-loops
            if new_src == new_dst:
                cursor.execute(
                    "DELETE FROM relations WHERE src_id = ? AND dst_id = ? AND relation_type = ?",
                    (src_id, dst_id, rel_type)
                )
                continue

            # Try to update, delete if conflict (duplicate relation)
            try:
                cursor.execute(
                    "UPDATE relations SET src_id = ?, dst_id = ? WHERE src_id = ? AND dst_id = ? AND relation_type = ?",
                    (new_src, new_dst, src_id, dst_id, rel_type)
                )
            except sqlite3.IntegrityError:
                # Relation already exists, just delete the duplicate
                cursor.execute(
                    "DELETE FROM relations WHERE src_id = ? AND dst_id = ? AND relation_type = ?",
                    (src_id, dst_id, rel_type)
                )

        # Delete weak entity
        cursor.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id_weak,))

    def _check_convergence(self) -> bool:
        """Check if graph has converged (minimal change over last 3 cycles)."""
        if len(self.metrics_history) < 3:
            return False

        # Check if compression ratio < 5% for last 3 cycles
        recent = self.metrics_history[-3:]
        avg_compression = sum(m.compression_ratio for m in recent) / 3

        return avg_compression < 0.05

    def run_until_convergence(self, max_cycles: int = 20) -> list[DreamCycleMetrics]:
        """Run dream cycles until graph converges or max_cycles reached."""
        for i in range(max_cycles):
            metrics = self.run_cycle()

            if metrics.graph_converged:
                print(f"\n✓ Graph converged after {metrics.cycle_num} cycles")
                break

        return self.metrics_history

    def export_metrics(self, output_path: str | Path) -> None:
        """Save metrics history to JSON."""
        output_path = Path(output_path)

        data = {
            'total_cycles': len(self.metrics_history),
            'cycles': [
                {
                    'cycle': m.cycle_num,
                    'timestamp': m.timestamp.isoformat(),
                    'entities_before': m.total_entities_before,
                    'entities_after': m.total_entities_after,
                    'entities_pruned': m.entities_pruned,
                    'entities_consolidated': m.entities_consolidated,
                    'relations_before': m.total_relations_before,
                    'relations_after': m.total_relations_after,
                    'relations_pruned': m.relations_pruned,
                    'long_term': m.long_term_count,
                    'short_term': m.short_term_count,
                    'dormant': m.dormant_count,
                    'avg_strength': m.avg_strength,
                    'graph_converged': m.graph_converged,
                    'compression_ratio': m.compression_ratio,
                }
                for m in self.metrics_history
            ]
        }

        output_path.write_text(json.dumps(data, indent=2))
        print(f"\nMetrics exported to {output_path}")


if __name__ == "__main__":
    # Example usage
    db_path = "/Users/christophermayfield/Documents/Projects/dreamrag/data/store/graph-00000000-0000-4000-8000-000000000001.db"

    if Path(db_path).exists():
        dream = DreamCycle(db_path)

        # Run until convergence
        metrics_history = dream.run_until_convergence(max_cycles=10)

        # Export results
        dream.export_metrics("dream_cycle_metrics.json")

        print(f"\n✓ Completed {len(metrics_history)} cycles")
    else:
        print(f"Database not found: {db_path}")
