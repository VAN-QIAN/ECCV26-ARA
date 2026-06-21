#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict


DEFAULT_TARGET_TYPES = [
    # Broad natural categories
    "plant",
    "bird",
    "animal",
    "insect",
    "fish",
    "reptile",
    "fungus",
    "fungi",
    "mammal",
    "species",
    "tree",
    "moth",
    "butterfly",
    "lizard",
    "beetle",
    "wasp",
    "amphibian",
    # Structures / man-made places
    "building",
    "church",
    "castle",
    "stadium",
    "museum",
    "bridge",
    "mosque",
    "monastery",
    "hotel",
    "lighthouse",
    "cathedral",
    "temple",
    "palace",
    "abbey",
    "fortress",
    "tower",
    "monument",
    "house",
    "synagogue",
    "theatre",
    "gate",
    "market",
    "mine",
    "ballpark",
    "velodrome",
    "pier",
    # Geographic / place-like
    "lake",
    "park",
    "mountain",
    "reservoir",
    "square",
    "river",
    "canal",
    "location",
    "place",
    "city",
    "island",
    "beach",
    "cave",
    "neighborhood",
    # Common two-word targets in EVQA
    "archaeological site",
    "railway station",
]


TYPE_DETECT_PATTERNS = (
    r"\bthis\s+{type}\b",
    r"\bthis\s+(?:specific|particular|pictured|shown)\s+{type}\b",
    r"\bthis\s+type\s+of\s+{type}\b",
    r"\bthese\s+{type}s?\b",
    r"\bthose\s+{type}s?\b",
    r"\bthe\s+{type}\s+in\s+(?:this|the)\s+(?:image|photo|picture)\b",
)


AMBIGUOUS_FALLBACK_WORDS = {
    "thing",
    "item",
    "object",
    "type",
    "kind",
    "sort",
    "part",
    "area",
    "image",
    "photo",
    "picture",
    "entity",
    "one",
    "ones",
    "other",
    "same",
}


TYPE_FAMILY = {
    # Living
    "plant": "living",
    "bird": "living",
    "animal": "living",
    "insect": "living",
    "fish": "living",
    "reptile": "living",
    "fungus": "living",
    "fungi": "living",
    "mammal": "living",
    "species": "living",
    "tree": "living",
    "moth": "living",
    "butterfly": "living",
    "lizard": "living",
    "beetle": "living",
    "wasp": "living",
    "amphibian": "living",
    # Structures / built places
    "building": "structure",
    "church": "structure",
    "castle": "structure",
    "stadium": "structure",
    "museum": "structure",
    "bridge": "structure",
    "mosque": "structure",
    "monastery": "structure",
    "hotel": "structure",
    "lighthouse": "structure",
    "cathedral": "structure",
    "temple": "structure",
    "palace": "structure",
    "abbey": "structure",
    "fortress": "structure",
    "tower": "structure",
    "monument": "structure",
    "house": "structure",
    "synagogue": "structure",
    "theatre": "structure",
    "gate": "structure",
    "market": "structure",
    "mine": "structure",
    "ballpark": "structure",
    "velodrome": "structure",
    "pier": "structure",
    "archaeological site": "structure",
    "railway station": "structure",
    # Geography
    "lake": "geography",
    "park": "geography",
    "mountain": "geography",
    "reservoir": "geography",
    "square": "geography",
    "river": "geography",
    "canal": "geography",
    "location": "geography",
    "place": "geography",
    "city": "geography",
    "island": "geography",
    "beach": "geography",
    "cave": "geography",
    "neighborhood": "geography",
}


AMBIGUOUS_PRONOUN_PATTERN = re.compile(
    r"\b(it|its|they|them|their|these|those)\b", re.IGNORECASE
)


def normalize_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def normalize_dataset_name(dataset_name):
    ds = normalize_text(dataset_name).lower()
    if ds == "landmark":
        return "landmarks"
    return ds


def cleanup_question(text):
    q = normalize_text(text)
    if not q:
        return q
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"\s+\?", "?", q)
    if q and not q.endswith("?"):
        q = q + "?"
    if q and q[0].isalpha() and q[0].islower():
        q = q[0].upper() + q[1:]
    return q


def parse_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def iter_type_pattern_strings(entity_type):
    escaped = re.escape(normalize_text(entity_type).lower())
    for template in TYPE_DETECT_PATTERNS:
        yield template.format(type=escaped)


def detect_entity_type(question, target_types, allow_fallback_this_word):
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
            candidate = (m.start(), pattern_rank, -len(t), t)
            if best is None or candidate < best:
                best = candidate
    if best is not None:
        return best[3]

    if not allow_fallback_this_word:
        return ""

    fallback = re.search(r"\b(?:this|these|those)\s+([a-z][a-z\-]+)\b", q_lower)
    if not fallback:
        return ""
    token = fallback.group(1).strip().lower()
    if token in AMBIGUOUS_FALLBACK_WORDS:
        return ""
    return token


def explicit_type_reference_present(question, entity_type):
    q = normalize_text(question).lower()
    t = normalize_text(entity_type).lower()
    if not q or not t:
        return False
    escaped = re.escape(t)
    patterns = [
        rf"\bthis\s+{escaped}\b",
        rf"\bthis\s+(?:specific|particular|pictured|shown)\s+{escaped}\b",
        rf"\bthis\s+type\s+of\s+{escaped}\b",
        rf"\bthis\s+{escaped}'s\b",
        rf"\bthe\s+{escaped}\s+in\s+(?:this|the)\s+(?:image|photo|picture)\b",
        rf"\bthe\s+{escaped}\s+shown\b",
    ]
    return any(re.search(p, q) for p in patterns)


def ensure_explicit_target_reference(question, entity_type):
    q = normalize_text(question)
    entity_type = normalize_text(entity_type).lower()
    if not q:
        return "", "empty_question"

    q = re.sub(r"\[\s*this\s+([^\]]+?)\s*\]", r"this \1", q, flags=re.IGNORECASE)
    q = cleanup_question(q)
    applied = []

    if explicit_type_reference_present(q, entity_type):
        return q, "already_explicit"

    # Resolve the most common ambiguous anaphora before composing Method2 queries.
    rewrites = [
        (
            rf"\bthese\s+{re.escape(entity_type)}s?\b",
            f"this {entity_type}",
            "replace_these_type",
        ),
        (
            rf"\bthose\s+{re.escape(entity_type)}s?\b",
            f"this {entity_type}",
            "replace_those_type",
        ),
        (
            r"\bits\s+([a-z][a-z0-9_-]*)\b",
            rf"this {entity_type}'s \1",
            "replace_its_np",
        ),
        (r"\bits\b", f"this {entity_type}'s", "replace_its"),
        (r"\bhow do they\b", f"How does this {entity_type}", "rewrite_how_do_they"),
        (r"\bwhere do they\b", f"Where does this {entity_type}", "rewrite_where_do_they"),
        (r"\bwhat do they\b", f"What does this {entity_type}", "rewrite_what_do_they"),
        (r"\bwhen do they\b", f"When does this {entity_type}", "rewrite_when_do_they"),
        (r"\bwhy do they\b", f"Why does this {entity_type}", "rewrite_why_do_they"),
        (r"\bdo they\b", f"Does this {entity_type}", "rewrite_do_they"),
        (r"\bare they\b", f"Is this {entity_type}", "rewrite_are_they"),
        (r"\btheir\b", f"this {entity_type}'s", "replace_their"),
        (r"\bthem\b", f"this {entity_type}", "replace_them"),
        (r"\bthey\b", f"this {entity_type}", "replace_they"),
        (r"\bthese\b", f"this {entity_type}", "replace_these"),
        (r"\bthose\b", f"this {entity_type}", "replace_those"),
        (r"\bit\b", f"this {entity_type}", "replace_it"),
    ]
    for pattern, replacement, label in rewrites:
        new_q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
        if new_q != q:
            q = new_q
            applied.append(label)

    # Fix common agreement issues after pronoun replacement.
    grammar_fixes = [
        (rf"\bhow do this {re.escape(entity_type)}\b", f"how does this {entity_type}"),
        (rf"\bwhat do this {re.escape(entity_type)}\b", f"what does this {entity_type}"),
        (rf"\bwhere do this {re.escape(entity_type)}\b", f"where does this {entity_type}"),
        (rf"\bwhen do this {re.escape(entity_type)}\b", f"when does this {entity_type}"),
        (rf"\bwhy do this {re.escape(entity_type)}\b", f"why does this {entity_type}"),
        (rf"\bdo this {re.escape(entity_type)}\b", f"does this {entity_type}"),
        (rf"\bare this {re.escape(entity_type)}\b", f"is this {entity_type}"),
    ]
    for pattern, replacement in grammar_fixes:
        new_q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
        if new_q != q:
            q = new_q
            applied.append("fix_grammar")

    # If the question still has "this <something>" but not our detected type, normalize once.
    if not explicit_type_reference_present(q, entity_type):
        new_q = re.sub(
            r"\b(?:this|these|those)\s+[a-z][a-z\-]*(?:\s+[a-z][a-z\-]*)?\b",
            f"this {entity_type}",
            q,
            count=1,
            flags=re.IGNORECASE,
        )
        if new_q != q:
            q = new_q
            applied.append("normalize_this_phrase")

    # Hard fallback: append explicit anchor mention.
    if not explicit_type_reference_present(q, entity_type):
        base = q.rstrip().rstrip("?")
        if base:
            q = base + f" about this {entity_type}"
        else:
            q = f"What is asked about this {entity_type}"
        applied.append("append_explicit_anchor")

    q = cleanup_question(q)
    if explicit_type_reference_present(q, entity_type):
        return q, "+".join(applied) if applied else "already_explicit"
    return q, "failed_to_make_explicit"


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


def ambiguous_pronoun_count(question):
    q = normalize_text(question)
    return len(AMBIGUOUS_PRONOUN_PATTERN.findall(q))


def score_row(row, entity_type):
    score = 0
    q = normalize_text(row.get("question"))
    if explicit_type_reference_present(q, entity_type):
        score += 2
    score -= min(2, ambiguous_pronoun_count(q))
    if normalize_text(row.get("question_type")):
        score += 1
    # Prefer deterministic ordering tie-breaks by data_id in pick_best_row.
    return score


def pick_best_row(rows, entity_type):
    best = None
    best_key = None
    for row in rows:
        key = (score_row(row, entity_type), normalize_text(row.get("data_id")))
        if best_key is None or key > best_key:
            best = row
            best_key = key
    return best


def build_entity_pool(
    rows,
    target_types,
    min_entities_per_type,
    allow_fallback_this_word,
    dedupe_mode,
):
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
            entity_type = detect_entity_type(question, target_types, allow_fallback_this_word)
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
        entity_type = detect_entity_type(question, target_types, allow_fallback_this_word)
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


def rewrite_query_with_side(explicit_query, entity_type, side):
    q = cleanup_question(explicit_query)
    t = normalize_text(entity_type).lower()
    if not q or not t:
        return q

    replacements = [
        (rf"\bthis\s+{re.escape(t)}'s\b", f"the {t} on the {side}'s"),
    ]
    for pattern in iter_type_pattern_strings(t):
        replacements.append((pattern, f"the {t} on the {side}"))

    for pattern, replacement in replacements:
        replaced = re.sub(pattern, replacement, q, count=1, flags=re.IGNORECASE)
        if replaced != q:
            return cleanup_question(replaced)

    return cleanup_question(q.rstrip("?") + f" for the {t} on the {side}?")


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
        row for row in same_type if normalize_text(row.get("question")) == anchor_question
    ]
    if same_question:
        return rng.choice(same_question), "same_type+same_question"
    return rng.choice(same_type), "same_type"


def pick_method2_pair(anchor, pool_by_type, method2_type_usage, rng):
    anchor_entity = normalize_text(anchor.get("wikipedia_title"))
    anchor_type = normalize_text(anchor.get("_entity_type"))
    anchor_dataset = normalize_dataset_name(anchor.get("dataset_name"))
    anchor_family = type_family(anchor_type)
    dataset_cross_family = []
    different_dataset = []
    cross_family = []
    all_candidates = []

    for entity_type, rows in pool_by_type.items():
        if entity_type == anchor_type:
            continue
        valid = [r for r in rows if normalize_text(r.get("wikipedia_title")) != anchor_entity]
        if not valid:
            continue

        family_diff = type_family(entity_type) != anchor_family
        valid_different_dataset = []
        if anchor_dataset:
            for r in valid:
                row_dataset = normalize_dataset_name(r.get("dataset_name"))
                if row_dataset and row_dataset != anchor_dataset:
                    valid_different_dataset.append(r)

        candidate_all = (entity_type, valid)
        all_candidates.append(candidate_all)
        if family_diff:
            cross_family.append(candidate_all)
        if valid_different_dataset:
            candidate_diff_dataset = (entity_type, valid_different_dataset)
            different_dataset.append(candidate_diff_dataset)
            if family_diff:
                dataset_cross_family.append(candidate_diff_dataset)

    if not all_candidates:
        return None, "no_different_type_candidate"

    # EVQA strategy priority:
    # 1) different dataset + cross family
    # 2) different dataset
    # 3) cross family
    # 4) any different type
    if dataset_cross_family:
        candidates = dataset_cross_family
        basis = "different_entity_type+different_dataset+cross_family"
    elif different_dataset:
        candidates = different_dataset
        basis = "different_entity_type+different_dataset"
    elif cross_family:
        candidates = cross_family
        basis = "different_entity_type+cross_family"
    else:
        candidates = all_candidates
        basis = "different_entity_type"

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
        anchor_dataset_name = normalize_dataset_name(anchor.get("dataset_name"))

        explicit_query, pronoun_clarity = ensure_explicit_target_reference(
            anchor_question, anchor_type
        )
        query_with_position = rewrite_query_with_side(explicit_query, anchor_type, side)

        method1, method1_basis = pick_method1_pair(anchor, pool_by_type, force_pairs, rng)
        method2, method2_basis = pick_method2_pair(anchor, pool_by_type, method2_type_usage, rng)

        method1_entity = normalize_text(method1.get("wikipedia_title")) if method1 else ""
        method1_data_id = normalize_text(method1.get("data_id")) if method1 else ""
        method1_type = normalize_text(method1.get("_entity_type")) if method1 else ""
        method1_url = normalize_text(method1.get("wikipedia_url")) if method1 else ""
        method1_dataset_name = normalize_dataset_name(method1.get("dataset_name")) if method1 else ""

        method2_entity = normalize_text(method2.get("wikipedia_title")) if method2 else ""
        method2_data_id = normalize_text(method2.get("data_id")) if method2 else ""
        method2_type = normalize_text(method2.get("_entity_type")) if method2 else ""
        method2_url = normalize_text(method2.get("wikipedia_url")) if method2 else ""
        method2_dataset_name = normalize_dataset_name(method2.get("dataset_name")) if method2 else ""

        out.append(
            {
                "anchor_data_id": normalize_text(anchor.get("data_id")),
                "anchor_entity": anchor_entity,
                "anchor_wikipedia_url": anchor_url,
                "anchor_entity_type": anchor_type,
                "anchor_dataset_name": anchor_dataset_name,
                "anchor_question": anchor_question,
                "anchor_question_explicit": explicit_query,
                "anchor_answer": anchor_answer,
                "anchor_question_type": normalize_text(anchor.get("question_type")),
                "target_side": side,
                "method1_pair_entity": method1_entity,
                "method1_pair_data_id": method1_data_id,
                "method1_pair_wikipedia_url": method1_url,
                "method1_pair_type": method1_type,
                "method1_pair_dataset_name": method1_dataset_name,
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
                "method2_pair_dataset_name": method2_dataset_name,
                "method2_similarity_basis": method2_basis,
                "method2_composite_layout": (
                    f"{anchor_entity}({side}) + {method2_entity}({other_side})"
                    if method2_entity
                    else ""
                ),
                "method2_query": query_with_position,
                "method2_query_with_position": query_with_position,
                "method2_query_without_position": explicit_query,
                "method2_pronoun_clarity": pronoun_clarity,
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
        description=(
            "Generate challenging E-VQA paired queries with Method1/Method2 and "
            "explicit Method2 pronoun resolution."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260220)
    parser.add_argument(
        "--target-types",
        default=",".join(DEFAULT_TARGET_TYPES),
        help=(
            "Comma-separated entity types to consider. Matched from patterns like "
            "'this <type>' and 'this specific <type>'."
        ),
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
        default="",
        help="Optional semicolon-separated mapping: anchor_entity=method1_pair_entity.",
    )
    parser.add_argument(
        "--enable-fallback-this-word",
        action="store_true",
        help=(
            "Enable fallback that infers type from the first word after "
            "'this/these/those' when it is not in --target-types. "
            "Disabled by default to avoid noisy types."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_types = [
        normalize_text(x).lower() for x in args.target_types.split(",") if normalize_text(x)
    ]
    force_pairs = parse_force_pairs(args.force_method1_pairs)

    rows, _ = parse_csv(args.input_csv)
    pool_by_type = build_entity_pool(
        rows=rows,
        target_types=target_types,
        min_entities_per_type=args.min_entities_per_type,
        allow_fallback_this_word=args.enable_fallback_this_word,
        dedupe_mode=args.dedupe_mode,
    )
    if not pool_by_type:
        raise RuntimeError("No eligible entities found. Try lowering --min-entities-per-type.")

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
    pronoun_clarity_counts = Counter(row["method2_pronoun_clarity"] for row in out_rows)

    print(f"Wrote: {args.output_csv}")
    print(f"Rows: {len(out_rows)}")
    print("Anchor type counts:")
    for t, c in sorted(anchor_type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {t}: {c}")
    print("Method2 distractor type counts:")
    for t, c in sorted(method2_type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {t}: {c}")
    print("Method2 pronoun clarity:")
    for t, c in sorted(pronoun_clarity_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
