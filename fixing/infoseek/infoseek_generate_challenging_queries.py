#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict


DEFAULT_TARGET_TYPES = [
    # Structures / locations
    "building",
    "bridge",
    "facility",
    "park",
    "mountain",
    "lake",
    "island",
    # "place",
    # "location",
    "city",
    "town",
    "village",
    # Transport
    "vehicle",
    "aircraft",
    # Living things
    "plant",
    "bird",
    "animal",
    "insect",
    "fish",
    "fungus",
    # Artifacts
    "food",
    "material",
]

TYPE_DETECT_PATTERNS = (
    r"\bthis\s+{type}\b",
    r"\bthis\s+(?:specific|particular|pictured|shown)\s+{type}\b",
    r"\bthis\s+type\s+of\s+{type}\b",
)

TYPE_FAMILY = {
    # Structures
    "building": "structure",
    "bridge": "structure",
    "facility": "structure",
    # Geography / place-like targets
    "park": "geography",
    "mountain": "geography",
    "lake": "geography",
    "island": "geography",
    "place": "geography",
    "location": "geography",
    "city": "geography",
    "town": "geography",
    "village": "geography",
    # Transport
    "vehicle": "transport",
    "aircraft": "transport",
    # Living
    "plant": "living",
    "bird": "living",
    "animal": "living",
    "insect": "living",
    "fish": "living",
    "fungus": "living",
    # Artifact-like
    "food": "artifact",
    "material": "artifact",
}


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def parse_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def iter_type_pattern_strings(entity_type):
    escaped = re.escape(normalize_text(entity_type).lower())
    for template in TYPE_DETECT_PATTERNS:
        yield template.format(type=escaped)


def detect_entity_type(question, target_types):
    q = normalize_text(question)
    if not q:
        return ""
    q_lower = q.lower()
    best = None
    for t in target_types:
        for pattern_rank, pattern in enumerate(iter_type_pattern_strings(t)):
            m = re.search(pattern, q_lower)
            if not m:
                continue
            pos = m.start()
            candidate = (pos, pattern_rank, -len(t), t)
            if best is None or candidate < best:
                best = candidate
    return best[3] if best is not None else ""


def type_family(entity_type):
    t = normalize_text(entity_type).lower()
    return TYPE_FAMILY.get(t, t)


def parse_force_pairs(raw):
    out = {}
    text = normalize_text(raw)
    if not text:
        return out
    chunks = [c.strip() for c in text.split(";") if c.strip()]
    for chunk in chunks:
        if "=" not in chunk:
            continue
        left, right = chunk.split("=", 1)
        left = normalize_text(left)
        right = normalize_text(right)
        if left and right:
            out[left] = right
    return out


def score_row(row, entity_type):
    score = 0
    q = normalize_text(row.get("question")).lower()
    if any(re.search(pattern, q) for pattern in iter_type_pattern_strings(entity_type)):
        score += 1
    if normalize_text(row.get("question_type")):
        score += 1
    return score


def pick_best_row(rows, entity_type):
    best = None
    best_key = None
    for row in rows:
        key = (
            score_row(row, entity_type),
            normalize_text(row.get("data_id")),
        )
        if best_key is None or key > best_key:
            best = row
            best_key = key
    return best


def build_entity_pool(rows, target_types, min_entities_per_type, dedupe_mode):
    if dedupe_mode not in {"question", "entity"}:
        raise ValueError(f"Unsupported dedupe_mode: {dedupe_mode}")

    filtered = {}
    if dedupe_mode == "entity":
        grouped = defaultdict(list)
        for row in rows:
            entity = normalize_text(row.get("wikipedia_title"))
            question = normalize_text(row.get("question"))
            if not entity or not question:
                continue
            entity_type = detect_entity_type(question, target_types)
            if not entity_type:
                continue
            grouped[(entity_type, entity)].append(row)

        by_type = defaultdict(list)
        for (entity_type, _entity), group_rows in grouped.items():
            chosen = pick_best_row(group_rows, entity_type)
            if chosen is None:
                continue
            row_copy = dict(chosen)
            row_copy["_entity_type"] = entity_type
            by_type[entity_type].append(row_copy)

        for entity_type, items in by_type.items():
            if len(items) >= min_entities_per_type:
                filtered[entity_type] = items
        return filtered

    # dedupe_mode == "question": keep all qualified question rows.
    by_type = defaultdict(list)
    unique_entities_per_type = defaultdict(set)
    for row in rows:
        entity = normalize_text(row.get("wikipedia_title"))
        question = normalize_text(row.get("question"))
        if not entity or not question:
            continue
        entity_type = detect_entity_type(question, target_types)
        if not entity_type:
            continue
        row_copy = dict(row)
        row_copy["_entity_type"] = entity_type
        by_type[entity_type].append(row_copy)
        unique_entities_per_type[entity_type].add(entity)

    for entity_type, items in by_type.items():
        if len(unique_entities_per_type[entity_type]) >= min_entities_per_type:
            filtered[entity_type] = items
    return filtered


def compute_balanced_targets(type_to_count, target_total, rng):
    if target_total <= 0:
        return {}
    types = [t for t, c in type_to_count.items() if c > 0]
    if not types:
        return {}

    # If target size is smaller than number of types, pick random subset of types.
    if target_total < len(types):
        chosen = set(rng.sample(types, target_total))
        return {t: (1 if t in chosen else 0) for t in types}

    targets = {t: 0 for t in types}
    remaining = target_total
    active = set(types)
    while remaining > 0 and active:
        candidates = sorted(
            [t for t in active if targets[t] < type_to_count[t]],
            key=lambda t: (targets[t], -type_to_count[t], t),
        )
        if not candidates:
            break
        chosen = candidates[0]
        targets[chosen] += 1
        remaining -= 1
        if targets[chosen] >= type_to_count[chosen]:
            active.discard(chosen)
    return targets


def rewrite_query_with_side(question, entity_type, side):
    q = normalize_text(question)
    if not q:
        return q
    for pattern in iter_type_pattern_strings(entity_type):
        replaced = re.sub(
            pattern,
            f"the {entity_type} on the {side}",
            q,
            count=1,
            flags=re.IGNORECASE,
        )
        if replaced != q:
            return replaced
    return q.rstrip("?") + f" for the {entity_type} on the {side}?"


def rewrite_query_without_side(question, entity_type):
    q = normalize_text(question)
    if not q:
        return q
    pattern = re.compile(
        r"\bthe\s+" + re.escape(entity_type) + r"\s+on\s+the\s+(left|right)\b",
        re.IGNORECASE,
    )
    restored = pattern.sub(f"this {entity_type}", q, count=1)
    return restored


def choose_side(index, side_mode, rng):
    if side_mode == "random":
        return "left" if rng.random() < 0.5 else "right"
    return "left" if index % 2 == 0 else "right"


def pick_method1_pair(anchor, pool_by_type, force_pairs, rng):
    anchor_entity = normalize_text(anchor.get("wikipedia_title"))
    anchor_type = normalize_text(anchor.get("_entity_type"))
    anchor_question = normalize_text(anchor.get("question"))

    forced_target = force_pairs.get(anchor_entity)
    if forced_target:
        forced_candidates = [
            row
            for row in pool_by_type.get(anchor_type, [])
            if normalize_text(row.get("wikipedia_title")) == forced_target
        ]
        if forced_candidates:
            return rng.choice(forced_candidates), "forced_entity_pair"

    same_type = [
        row
        for row in pool_by_type.get(anchor_type, [])
        if normalize_text(row.get("wikipedia_title")) != anchor_entity
    ]
    if not same_type:
        return None, "no_same_type_candidate"

    same_question = [
        row
        for row in same_type
        if normalize_text(row.get("question")) == anchor_question
    ]
    if same_question:
        return rng.choice(same_question), "same_type+same_question"
    return rng.choice(same_type), "same_type"


def pick_method2_pair(anchor, pool_by_type, method2_type_usage, rng):
    anchor_entity = normalize_text(anchor.get("wikipedia_title"))
    anchor_type = normalize_text(anchor.get("_entity_type"))
    anchor_family = type_family(anchor_type)
    cross_family = []
    all_candidates = []
    for entity_type, rows in pool_by_type.items():
        if entity_type == anchor_type:
            continue
        valid = [r for r in rows if normalize_text(r.get("wikipedia_title")) != anchor_entity]
        if valid:
            candidate = (entity_type, valid)
            all_candidates.append(candidate)
            if type_family(entity_type) != anchor_family:
                cross_family.append(candidate)
    if not all_candidates:
        return None, "no_different_type_candidate"

    # Prefer cross-family distractors so the target in no-position queries is easier to infer.
    candidates = cross_family if cross_family else all_candidates
    basis = "different_entity_type+cross_family" if cross_family else "different_entity_type"

    # Keep distractor types balanced across the generated set.
    candidates.sort(key=lambda x: (method2_type_usage[x[0]], -len(x[1]), x[0]))
    chosen_type, rows = candidates[0]
    chosen_row = rng.choice(rows)
    method2_type_usage[chosen_type] += 1
    return chosen_row, basis


def generate_rows(anchors, pool_by_type, force_pairs, side_mode, rng):
    out = []
    method2_type_usage = Counter()

    for i, anchor in enumerate(anchors):
        side = choose_side(i, side_mode, rng)
        other_side = "right" if side == "left" else "left"
        anchor_entity = normalize_text(anchor.get("wikipedia_title"))
        anchor_type = normalize_text(anchor.get("_entity_type"))
        anchor_question = normalize_text(anchor.get("question"))
        anchor_answer = normalize_text(anchor.get("answer"))
        anchor_url = normalize_text(anchor.get("wikipedia_url"))

        method1, method1_basis = pick_method1_pair(anchor, pool_by_type, force_pairs, rng)
        method2, method2_basis = pick_method2_pair(anchor, pool_by_type, method2_type_usage, rng)

        method1_entity = normalize_text(method1.get("wikipedia_title")) if method1 else ""
        method1_data_id = normalize_text(method1.get("data_id")) if method1 else ""
        method1_type = normalize_text(method1.get("_entity_type")) if method1 else ""
        method1_url = normalize_text(method1.get("wikipedia_url")) if method1 else ""

        method2_entity = normalize_text(method2.get("wikipedia_title")) if method2 else ""
        method2_data_id = normalize_text(method2.get("data_id")) if method2 else ""
        method2_type = normalize_text(method2.get("_entity_type")) if method2 else ""
        method2_url = normalize_text(method2.get("wikipedia_url")) if method2 else ""

        query_with_position = rewrite_query_with_side(anchor_question, anchor_type, side)
        query_without_position = rewrite_query_without_side(anchor_question, anchor_type)

        out.append(
            {
                "anchor_data_id": normalize_text(anchor.get("data_id")),
                "anchor_entity": anchor_entity,
                "anchor_wikipedia_url": anchor_url,
                "anchor_entity_type": anchor_type,
                "anchor_question": anchor_question,
                "anchor_answer": anchor_answer,
                "anchor_question_type": normalize_text(anchor.get("question_type")),
                "target_side": side,
                "method1_pair_entity": method1_entity,
                "method1_pair_data_id": method1_data_id,
                "method1_pair_wikipedia_url": method1_url,
                "method1_pair_type": method1_type,
                "method1_similarity_basis": method1_basis,
                "method1_composite_layout": (
                    f"{anchor_entity}({side}) + {method1_entity}({other_side})"
                    if method1_entity
                    else ""
                ),
                "method1_query": query_with_position,
                "method1_expected_answer": anchor_answer,
                "method2_pair_entity": method2_entity,
                "method2_pair_data_id": method2_data_id,
                "method2_pair_wikipedia_url": method2_url,
                "method2_pair_type": method2_type,
                "method2_similarity_basis": method2_basis,
                "method2_composite_layout": (
                    f"{anchor_entity}({side}) + {method2_entity}({other_side})"
                    if method2_entity
                    else ""
                ),
                "method2_query": query_with_position,
                "method2_query_with_position": query_with_position,
                "method2_query_without_position": query_without_position,
                "method2_expected_answer": anchor_answer,
                "distractor_entity": method2_entity,
                "distractor_wikipedia_url": method2_url,
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate challenging InfoSeek queries with two pairing methods and diverse entity types."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument(
        "--target-types",
        default=",".join(DEFAULT_TARGET_TYPES),
        help="Comma-separated entity types to consider, matched from patterns like 'this <type>' / 'this specific <type>' / 'this type of <type>'.",
    )
    parser.add_argument(
        "--min-entities-per-type",
        type=int,
        default=4,
        help="Drop types with fewer unique entities than this threshold.",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["question", "entity"],
        default="question",
        help=(
            "Pool construction mode: 'question' keeps all qualified question rows; "
            "'entity' keeps one best row per (type, entity)."
        ),
    )
    parser.add_argument(
        "--side-mode",
        choices=["alternating", "random"],
        default="alternating",
        help="How to assign left/right for target entities.",
    )
    parser.add_argument(
        "--force-method1-pairs",
        default="Altice Arena=Allianz Arena;Allianz Arena=Altice Arena",
        help="Optional semicolon-separated mapping: anchor_entity=method1_pair_entity.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_types = [normalize_text(x).lower() for x in args.target_types.split(",") if normalize_text(x)]
    force_pairs = parse_force_pairs(args.force_method1_pairs)

    rows, _ = parse_csv(args.input_csv)
    pool_by_type = build_entity_pool(
        rows=rows,
        target_types=target_types,
        min_entities_per_type=args.min_entities_per_type,
        dedupe_mode=args.dedupe_mode,
    )
    if not pool_by_type:
        raise RuntimeError("No eligible entities found. Try lowering --min-entities-per-type.")

    # Shuffle each type pool before sampling so selection is deterministic but not ordered by source CSV.
    for entity_type in pool_by_type:
        rng.shuffle(pool_by_type[entity_type])

    type_to_count = {t: len(items) for t, items in pool_by_type.items()}
    targets = compute_balanced_targets(type_to_count, args.num_samples, rng)
    selected = []
    selected_per_type = Counter()
    for entity_type, n in sorted(targets.items()):
        if n <= 0:
            continue
        picks = pool_by_type[entity_type][:n]
        selected.extend(picks)
        selected_per_type[entity_type] += len(picks)

    if len(selected) < args.num_samples:
        # Fill shortfall from any remaining capacity while preserving type diversity.
        remaining = args.num_samples - len(selected)
        leftovers = []
        for entity_type, items in pool_by_type.items():
            used = selected_per_type[entity_type]
            leftovers.extend(items[used:])
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    if not selected:
        raise RuntimeError("No anchors selected.")
    if len(selected) > args.num_samples:
        selected = selected[: args.num_samples]
    rng.shuffle(selected)

    out_rows = generate_rows(
        anchors=selected,
        pool_by_type=pool_by_type,
        force_pairs=force_pairs,
        side_mode=args.side_mode,
        rng=rng,
    )
    write_csv(args.output_csv, out_rows)

    anchor_type_counts = Counter(row["anchor_entity_type"] for row in out_rows)
    method2_type_counts = Counter(row["method2_pair_type"] for row in out_rows if row["method2_pair_type"])
    print(f"Wrote: {args.output_csv}")
    print(f"Rows: {len(out_rows)}")
    print("Anchor type counts:")
    for t, c in sorted(anchor_type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {t}: {c}")
    print("Method2 distractor type counts:")
    for t, c in sorted(method2_type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
