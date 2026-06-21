"""Top-K aware variant of the Qwen VQA pipeline."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.answer_generator import reconstruct_wiki_article, reconstruct_wiki_sections
from model.retriever import WikipediaKnowledgeBaseEntry

from ..context import SectionReranker, extract_title_from_section
from ..io import build_retrieval_record, iter_examples, load_metadata, load_retrieval_results
from ..pipeline import PipelineConfig, QwenVQAPipeline
from ..types import ContextRecord, IdentificationRecord, MetadataRecord, RetrievalRecord


@dataclass
class TopKPipelineConfig(PipelineConfig):
    """Configuration for the entity top-k selection pipeline."""

    identification_select_top: int = 3
    identification_score_top_k: int = 3
    entity_top_k: int = 3
    section_reranker_backend: str = "auto"  # values: auto, bge
    section_score_weight: float = 1.0
    retrieval_similarity_weight: float = 0.0
    identification_probability_weight: float = 0.0
    identification_score_mode: str = "multiply"  # values: multiply, add
    section_score_source: str = "blended"  # values: blended, raw
    section_score_normalization: str = "none"  # values: none, minmax
    top1_identification_only: bool = False
    prepare_timing_summary_path: Optional[str] = None
    answer_timing_summary_path: Optional[str] = None


@dataclass
class _EntityCandidate:
    url: str
    title: str
    entry: WikipediaKnowledgeBaseEntry
    rank: Optional[int]
    retrieval_similarity: Optional[float] = None
    probability: Optional[float] = None


@dataclass
class _SectionCandidate:
    text: str
    entity_url: str
    entity_title: str
    entity_rank: Optional[int]
    section_title: Optional[str]
    reranker_score: Optional[float] = None
    retrieval_similarity: Optional[float] = None
    score: Optional[float] = None
    entity_probability: Optional[float] = None
    blended_score: Optional[float] = None


class TopKQwenPipeline(QwenVQAPipeline):
    """Pipeline that reranks sections across multiple identified entities."""

    config: TopKPipelineConfig

    def __init__(self, config: TopKPipelineConfig) -> None:
        super().__init__(config)
        backend = (config.section_reranker_backend or "auto").lower()
        if backend not in {"auto", "bge"}:
            self.logger.warning(
                "Unknown section_reranker_backend '%s'; falling back to BGE.",
                backend,
            )
            backend = "bge"
        if backend == "auto":
            backend = "bge"
        self.section_backend = backend
        self.section_selector = SectionReranker(config.section_reranker, config.qwen_device)
        mode = str(getattr(self.config, "identification_score_mode", "multiply")).lower()
        if mode not in {"multiply", "add"}:
            self.logger.warning(
                "Unknown identification_score_mode '%s'; defaulting to 'multiply'.",
                mode,
            )
            mode = "multiply"
        self.config.identification_score_mode = mode
        score_source = str(getattr(self.config, "section_score_source", "blended")).lower()
        if score_source not in {"blended", "raw"}:
            self.logger.warning(
                "Unknown section_score_source '%s'; defaulting to 'blended'.",
                score_source,
            )
            score_source = "blended"
        self.config.section_score_source = score_source
        normalization = str(getattr(self.config, "section_score_normalization", "none")).lower()
        if normalization not in {"none", "minmax"}:
            self.logger.warning(
                "Unknown section_score_normalization '%s'; defaulting to 'none'.",
                normalization,
            )
            normalization = "none"
        self.config.section_score_normalization = normalization

        self.logger.info(
            "TopK pipeline ready | entities_top=%d | reranker_backend=%s",
            config.entity_top_k,
            self.section_backend,
        )
        self._last_section_components = None
        self._last_prepare_timing_summary: Optional[Dict[str, Any]] = None
        self._last_answer_timing_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieval_similarity_lookup(self, retrieval: RetrievalRecord) -> Dict[str, float]:
        similarities = retrieval.meta.get("retrieval_similarities") or []
        lookup: Dict[str, float] = {}
        for url, score in zip(retrieval.candidate_urls, similarities):
            if url in lookup:
                continue
            try:
                lookup[url] = float(score)
            except (TypeError, ValueError):
                continue
        return lookup

    def _entity_candidates(
        self,
        retrieval: RetrievalRecord,
        identification: IdentificationRecord,
        retrieval_similarity: Dict[str, float],
    ) -> List[_EntityCandidate]:
        desired = max(1, self.config.entity_top_k)
        ranked_urls = identification.ranked_urls or []
        rank_lookup = {url: idx for idx, url in enumerate(ranked_urls)}
        probability_lookup: Dict[str, float] = {}
        ranked_probabilities = identification.ranked_probabilities or []
        for url, prob in zip(ranked_urls, ranked_probabilities):
            if prob is None:
                continue
            try:
                probability_lookup[url] = float(prob)
            except (TypeError, ValueError):
                continue
        ordered_urls: List[str] = []
        for url in ranked_urls:
            if url not in ordered_urls:
                ordered_urls.append(url)
        if identification.selected_url and identification.selected_url not in ordered_urls:
            ordered_urls.append(identification.selected_url)
        for url in retrieval.reranked_urls:
            if len(ordered_urls) >= desired:
                break
            if url not in ordered_urls:
                ordered_urls.append(url)
        candidates: List[_EntityCandidate] = []
        for url in ordered_urls[:desired]:
            entry = self.kb_by_url.get(url)
            if entry is None:
                continue
            candidates.append(
                _EntityCandidate(
                    url=url,
                    title=entry.title,
                    entry=entry,
                    rank=rank_lookup.get(url),
                    retrieval_similarity=retrieval_similarity.get(url),
                    probability=probability_lookup.get(url),
                )
            )
        return candidates

    def _collect_sections(self, entities: Sequence[_EntityCandidate]) -> List[_SectionCandidate]:
        pool: List[_SectionCandidate] = []
        for entity in entities:
            sections = reconstruct_wiki_sections(entity.entry)
            if not sections:
                article = reconstruct_wiki_article(entity.entry)
                pool.append(
                    _SectionCandidate(
                        text=article,
                        entity_url=entity.url,
                        entity_title=entity.title,
                        entity_rank=entity.rank,
                        section_title=None,
                        retrieval_similarity=entity.retrieval_similarity,
                        entity_probability=entity.probability,
                    )
                )
                continue
            for section in sections:
                pool.append(
                    _SectionCandidate(
                        text=section,
                        entity_url=entity.url,
                        entity_title=entity.title,
                        entity_rank=entity.rank,
                        section_title=extract_title_from_section(section),
                        retrieval_similarity=entity.retrieval_similarity,
                        entity_probability=entity.probability,
                    )
                )
        return pool

    def _order_sections(
        self,
        question: str,
        sections: Sequence[_SectionCandidate],
        image_path: Optional[str],
    ) -> Tuple[List[_SectionCandidate], List[Optional[float]], List[Optional[float]]]:
        if not sections:
            return [], [], []
        texts = [candidate.text for candidate in sections]
        best_idx, raw_scores = self.section_selector.pick_best(question, texts)

        scores: List[float] = list(raw_scores) if raw_scores else []
        score_source = getattr(self.config, "section_score_source", "blended")
        score_normalization = getattr(self.config, "section_score_normalization", "none")

        base_scores_initial: List[Optional[float]] = []
        for idx in range(len(sections)):
            raw_score = scores[idx] if scores and idx < len(scores) else None
            base_scores_initial.append(raw_score)
        if score_normalization == "minmax":
            normalized_base_scores = self._normalize_optional_scores(base_scores_initial)
        else:
            normalized_base_scores = list(base_scores_initial)

        section_weight = float(getattr(self.config, "section_score_weight", 1.0))
        retrieval_weight = float(getattr(self.config, "retrieval_similarity_weight", 0.0))
        identification_weight = float(getattr(self.config, "identification_probability_weight", 0.0))
        identification_mode = str(getattr(self.config, "identification_score_mode", "multiply")).lower()

        base_scores: List[Optional[float]] = []
        section_components: List[Optional[float]] = []
        retrieval_components: List[Optional[float]] = []
        identification_components: List[Optional[float]] = []
        final_scores: List[Optional[float]] = []

        for idx, candidate in enumerate(sections):
            raw_score = scores[idx] if scores and idx < len(scores) else None
            base_score = normalized_base_scores[idx] if idx < len(normalized_base_scores) else None
            candidate.blended_score = base_score
            candidate.reranker_score = raw_score
            identification_value = None
            if candidate.entity_probability is not None:
                try:
                    identification_value = float(candidate.entity_probability)
                except (TypeError, ValueError):
                    identification_value = None
            retrieval_value = None
            if candidate.retrieval_similarity is not None:
                try:
                    retrieval_value = float(candidate.retrieval_similarity)
                except (TypeError, ValueError):
                    retrieval_value = None

            section_component = base_score
            if identification_mode == "multiply" and section_component is not None and identification_value is not None:
                section_component = section_component * identification_value

            weighted_sum = 0.0
            weight_applied = False
            final_score: Optional[float] = None

            if section_component is not None and section_weight != 0.0:
                weighted_sum += section_weight * section_component
                weight_applied = True
            if retrieval_value is not None and retrieval_weight != 0.0:
                weighted_sum += retrieval_weight * retrieval_value
                weight_applied = True
            if identification_value is not None and identification_weight != 0.0:
                weighted_sum += identification_weight * identification_value
                weight_applied = True

            if weight_applied:
                final_score = weighted_sum
            else:
                final_score = section_component

            base_scores.append(base_score)
            section_components.append(section_component)
            retrieval_components.append(retrieval_value)
            identification_components.append(identification_value)
            final_scores.append(final_score)

        def _score_value(values: Sequence[Optional[float]], idx: int) -> float:
            if not values or idx >= len(values):
                return float("-inf")
            value = values[idx]
            return float(value) if value is not None else float("-inf")

        if any(value is not None for value in final_scores):
            order = sorted(range(len(texts)), key=lambda idx: _score_value(final_scores, idx), reverse=True)
        elif any(value is not None for value in base_scores):
            order = sorted(range(len(texts)), key=lambda idx: _score_value(base_scores, idx), reverse=True)
        elif best_idx >= 0:
            order = [best_idx] + [idx for idx in range(len(texts)) if idx != best_idx]
        else:
            order = list(range(len(texts)))
        ordered: List[_SectionCandidate] = []
        ordered_scores: List[Optional[float]] = []
        ordered_raw_scores: List[Optional[float]] = []
        ordered_section_components: List[Optional[float]] = []
        ordered_retrieval_components: List[Optional[float]] = []
        ordered_identification_components: List[Optional[float]] = []
        for idx in order:
            final_score = final_scores[idx] if idx < len(final_scores) else None
            base_score = base_scores[idx] if idx < len(base_scores) else None
            raw_score = scores[idx] if scores and idx < len(scores) else None
            candidate = sections[idx]
            ordered.append(
                _SectionCandidate(
                    text=candidate.text,
                    entity_url=candidate.entity_url,
                    entity_title=candidate.entity_title,
                    entity_rank=candidate.entity_rank,
                    section_title=candidate.section_title,
                    reranker_score=raw_score,
                    retrieval_similarity=candidate.retrieval_similarity,
                    score=final_score,
                    entity_probability=candidate.entity_probability,
                    blended_score=base_score,
                )
            )
            ordered_scores.append(final_score)
            ordered_raw_scores.append(raw_score)
            ordered_section_components.append(section_components[idx])
            ordered_retrieval_components.append(retrieval_components[idx])
            ordered_identification_components.append(identification_components[idx])

        # Attach component traces to the original sequence order for metadata.
        self._last_section_components = {
            "section": ordered_section_components,
            "retrieval": ordered_retrieval_components,
            "identification": ordered_identification_components,
            "weights": {
                "section": section_weight,
                "retrieval": retrieval_weight,
                "identification": identification_weight,
                "mode": identification_mode,
                "score_normalization": score_normalization,
            },
        }

        return ordered, ordered_scores, ordered_raw_scores

    def _pick_best_sections(
        self,
        question: str,
        sections: Sequence[str],
        image_path: Optional[str],
    ) -> Tuple[int, List[float]]:
        return self.section_selector.pick_best(question, list(sections))

    @staticmethod
    def _normalize_optional_scores(values: Sequence[Optional[float]]) -> List[Optional[float]]:
        valid = [
            float(value)
            for value in values
            if value is not None and isinstance(value, (int, float)) and math.isfinite(value)
        ]
        if not valid:
            return [None for _ in values]
        lower = min(valid)
        upper = max(valid)
        if math.isclose(lower, upper):
            return [1.0 if value is not None else None for value in values]
        span = upper - lower
        normalized: List[Optional[float]] = []
        for value in values:
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                normalized.append(None)
                continue
            norm = (float(value) - lower) / span
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            normalized.append(norm)
        return normalized

    @staticmethod
    def _safe_average(total: float, count: int) -> float:
        return total / count if count > 0 else 0.0

    def _write_timing_summary(self, payload: Dict[str, Any], destination: Optional[str]) -> None:
        if not destination:
            return
        output_path = Path(destination)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.info("Wrote runtime summary -> %s", output_path)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def prepare_metadata(self, metadata_path: str) -> List[MetadataRecord]:
        if not self.kb_by_url:
            raise ValueError("Knowledge base must be loaded to prepare metadata.")
        start_time = time.perf_counter()
        examples_seen = 0
        examples_prepared = 0
        skipped_no_retrieval = 0
        skipped_no_image = 0
        skipped_missing_kb = 0
        skipped_no_entities = 0
        skipped_no_sections = 0
        total_example_seconds = 0.0
        identification_seconds = 0.0
        identification_scoring_seconds = 0.0
        section_pool_build_seconds = 0.0
        section_rerank_seconds = 0.0
        io_write_seconds = 0.0

        retrieval_blob = load_retrieval_results(self.config.retrieval_results)
        records: List[MetadataRecord] = []
        metadata_path_obj = Path(metadata_path)
        metadata_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path_obj.open("w", encoding="utf-8") as metadata_out:
            for idx, example in iter_examples(
                self.config.test_file,
                augmented_csv_mode=getattr(self.config, "augmented_csv_mode", "off"),
                augmented_query_variant=getattr(self.config, "augmented_query_variant", "with_position"),
                augmented_ground_truth_target=getattr(self.config, "augmented_ground_truth_target", "anchor"),
                augmented_image_path_root=getattr(self.config, "augmented_image_path_root", None),
            ):
                examples_seen += 1
                example_start = time.perf_counter()
                candidate_ids = self._candidate_data_ids(idx, example)
                matched_id = next((cid for cid in candidate_ids if cid in retrieval_blob), None)
                if matched_id is None:
                    skipped_no_retrieval += 1
                    self.logger.warning(
                        "Skipping example %s: no retrieval results for candidates %s",
                        example.get("data_id", candidate_ids[0] if candidate_ids else idx),
                        candidate_ids,
                    )
                    continue
                image_path = self._resolve_image_path(
                    dataset_name=example.get("dataset_name", ""),
                    image_id=str(example.get("dataset_image_ids", "")),
                    explicit_path=example.get("image_path_override"),
                )
                if image_path is None:
                    skipped_no_image += 1
                    self.logger.warning("Skipping %s: image not found", matched_id)
                    continue
                retrieval_record = build_retrieval_record(
                    example,
                    matched_id,
                    retrieval_blob[matched_id],
                    image_path=image_path,
                    candidate_ids=candidate_ids,
                )
                try:
                    stage_start = time.perf_counter()
                    identification = self._run_identification(retrieval_record)
                    identification_seconds += time.perf_counter() - stage_start
                except ValueError:
                    skipped_missing_kb += 1
                    self.logger.warning("Skipping %s due to missing knowledge base entries", matched_id)
                    continue
                similarity_lookup = self._retrieval_similarity_lookup(retrieval_record)
                top1_only_mode = bool(getattr(self.config, "top1_identification_only", False))
                score_top_k = (
                    0
                    if top1_only_mode
                    else max(0, int(getattr(self.config, "identification_score_top_k", 0)))
                )
                existing_probs = identification.ranked_probabilities or []
                existing_prob_ready = any(
                    prob is not None for prob in existing_probs[:score_top_k]
                ) if score_top_k > 0 else False
                if score_top_k > 0 and not existing_prob_ready:
                    top_titles = identification.ranked_titles[:score_top_k] or []
                    if top_titles:
                        stage_start = time.perf_counter()
                        try:
                            candidate_similarities = [
                                similarity_lookup.get(url)
                                for url in identification.ranked_urls[:score_top_k]
                            ]
                            scores = self.qwen.score_candidates(
                                image_path=retrieval_record.image_path,
                                candidate_titles=top_titles,
                                question=retrieval_record.question,
                                instruction=self._build_augmented_identification_instruction(retrieval_record),
                                candidate_similarities=candidate_similarities,
                                max_new_tokens=self.config.identification_max_new_tokens,
                                temperature=self.config.identification_temperature,
                            )
                        except Exception as exc:
                            self.logger.warning(
                                "Failed to score identification candidates for %s: %s",
                                matched_id,
                                exc,
                            )
                        else:
                            probability_map = {score.title: score.probability for score in scores.candidates}
                            identification.ranked_probabilities = [
                                probability_map.get(title) if title in probability_map else None
                                for title in identification.ranked_titles
                            ]
                            identification.none_probability = scores.none_probability
                            retrieval_record.meta["identification_scores"] = [
                                {
                                    "title": candidate.title,
                                    "probability": candidate.probability,
                                    "ordered_rank": idx,
                                }
                                for idx, candidate in enumerate(scores.candidates)
                            ]
                            retrieval_record.meta["identification_none_probability"] = scores.none_probability
                            retrieval_record.meta["identification_scores_raw_response"] = scores.raw_response
                        finally:
                            identification_scoring_seconds += time.perf_counter() - stage_start
                elif score_top_k > 0 and existing_prob_ready:
                    retrieval_record.meta.setdefault(
                        "identification_scores",
                        [
                            {
                                "title": title,
                                "probability": prob,
                                "ordered_rank": idx,
                            }
                            for idx, (title, prob) in enumerate(
                                zip(
                                    identification.ranked_titles[:score_top_k],
                                    existing_probs[:score_top_k],
                                )
                            )
                        ],
                    )
                    retrieval_record.meta.setdefault(
                        "identification_none_probability",
                        identification.none_probability,
                    )
                ground_truth_url = (
                    example.get("wikipedia_url")
                    or example.get("ground_truth_url")
                    or example.get("gold_wikipedia_url")
                )
                if ground_truth_url:
                    ranked_urls = identification.ranked_urls or []
                    in_topk = ground_truth_url in ranked_urls or ground_truth_url == identification.selected_url
                    identification.ground_truth_url = ground_truth_url
                    identification.ground_truth_in_topk = in_topk
                    retrieval_record.meta.setdefault("ground_truth_url", ground_truth_url)
                    retrieval_record.meta["ground_truth_entity_in_topk_identification"] = in_topk
                retrieval_record.meta[
                    "identification_prompt_includes_similarity"
                ] = self.config.identification_include_similarity
                retrieval_record.meta["top1_identification_only"] = top1_only_mode
                metadata_record: Optional[MetadataRecord] = None
                if top1_only_mode:
                    selected_entry = self.kb_by_url.get(identification.selected_url)
                    if selected_entry is None:
                        skipped_missing_kb += 1
                        self.logger.warning(
                            "Skipping %s: selected entity missing in KB for top1-identification-only mode",
                            matched_id,
                        )
                        continue
                    source_probability: Optional[float] = None
                    if identification.ranked_urls and identification.ranked_probabilities:
                        for ranked_url, ranked_prob in zip(
                            identification.ranked_urls,
                            identification.ranked_probabilities,
                        ):
                            if ranked_url == identification.selected_url:
                                source_probability = ranked_prob
                                break
                    context = ContextRecord(
                        data_id=retrieval_record.data_id,
                        mode="article",
                        text=reconstruct_wiki_article(selected_entry),
                        source_url=identification.selected_url,
                        source_rank=identification.selected_index,
                        source_probability=source_probability,
                    )
                    retrieval_record.reranked_sections = []
                    retrieval_record.meta["section_reranker_backend"] = "disabled_top1_identification_only"
                    self._last_section_components = None
                    self._log_prepare_step(retrieval_record, identification, context)
                    metadata_record = MetadataRecord(retrieval_record, identification, context)
                else:
                    stage_start = time.perf_counter()
                    entities = self._entity_candidates(
                        retrieval_record,
                        identification,
                        similarity_lookup,
                    )
                    sections: List[_SectionCandidate] = []
                    if entities:
                        sections = self._collect_sections(entities)
                    section_pool_build_seconds += time.perf_counter() - stage_start
                    if not entities:
                        skipped_no_entities += 1
                        self.logger.warning("No entity candidates survived for %s", matched_id)
                        continue
                    if not sections:
                        skipped_no_sections += 1
                        self.logger.warning("No sections available after expansion for %s", matched_id)
                        continue
                    stage_start = time.perf_counter()
                    ordered_sections, ordered_scores, ordered_raw_scores = self._order_sections(
                        retrieval_record.question,
                        sections,
                        retrieval_record.image_path,
                    )
                    section_rerank_seconds += time.perf_counter() - stage_start
                    if not ordered_sections:
                        ordered_sections = sections
                        ordered_scores = [candidate.score for candidate in sections]
                        ordered_raw_scores = [candidate.reranker_score for candidate in sections]
                    best_section = ordered_sections[0]
                    mode = "section" if best_section.section_title else "article"
                    retrieval_record.reranked_sections = [candidate.text for candidate in ordered_sections]
                    retrieval_record.meta["section_entity_urls"] = [candidate.entity_url for candidate in ordered_sections]
                    retrieval_record.meta["section_entity_titles"] = [candidate.entity_title for candidate in ordered_sections]
                    retrieval_record.meta["section_entity_ranks"] = [candidate.entity_rank for candidate in ordered_sections]
                    retrieval_record.meta["section_entity_probabilities"] = [
                        candidate.entity_probability for candidate in ordered_sections
                    ]
                    retrieval_record.meta["section_titles"] = [candidate.section_title for candidate in ordered_sections]
                    retrieval_record.meta["section_scores"] = ordered_scores
                    retrieval_record.meta["section_scores_unweighted"] = [
                        candidate.blended_score for candidate in ordered_sections
                    ]
                    components_payload = getattr(self, "_last_section_components", None)
                    if isinstance(components_payload, dict):
                        retrieval_record.meta["section_scores_components"] = {
                            "section": components_payload.get("section"),
                            "retrieval": components_payload.get("retrieval"),
                            "identification": components_payload.get("identification"),
                            "weights": components_payload.get("weights"),
                        }
                        retrieval_record.meta["section_scores_weighting"] = components_payload.get("weights")
                    if ordered_raw_scores:
                        retrieval_record.meta["section_scores_raw"] = ordered_raw_scores
                    retrieval_record.meta["section_scores_source"] = getattr(self.config, "section_score_source", "blended")
                    retrieval_record.meta["section_scores_normalization"] = getattr(
                        self.config,
                        "section_score_normalization",
                        "none",
                    )
                    # Keep retrieval similarities for all reranker backends so downstream
                    # reranking utilities can always fuse section/retrieval/identification.
                    retrieval_record.meta["section_retrieval_similarities"] = [
                        candidate.retrieval_similarity for candidate in ordered_sections
                    ]
                    retrieval_record.meta["section_reranker_backend"] = self.section_backend
                    context = ContextRecord(
                        data_id=retrieval_record.data_id,
                        mode=mode,
                        text=best_section.text,
                        source_url=best_section.entity_url,
                        section_title=best_section.section_title,
                        source_rank=best_section.entity_rank,
                        section_score=best_section.score,
                        source_probability=best_section.entity_probability,
                    )
                    self._last_section_components = None
                    self._log_prepare_step(retrieval_record, identification, context)
                    metadata_record = MetadataRecord(retrieval_record, identification, context)

                records.append(metadata_record)
                io_start = time.perf_counter()
                metadata_out.write(json.dumps(metadata_record.to_dict(), ensure_ascii=False) + "\n")
                metadata_out.flush()
                io_write_seconds += time.perf_counter() - io_start
                examples_prepared += 1
                total_example_seconds += time.perf_counter() - example_start
        total_seconds = time.perf_counter() - start_time
        avg_example_seconds = self._safe_average(total_example_seconds, examples_prepared)
        prepare_summary: Dict[str, Any] = {
            "phase": "prepare",
            "section_reranker_backend": self.section_backend,
            "top1_identification_only": bool(getattr(self.config, "top1_identification_only", False)),
            "test_file": self.config.test_file,
            "retrieval_results": self.config.retrieval_results,
            "metadata_path": metadata_path,
            "examples_seen": examples_seen,
            "examples_prepared": examples_prepared,
            "skipped": {
                "no_retrieval": skipped_no_retrieval,
                "no_image": skipped_no_image,
                "missing_kb": skipped_missing_kb,
                "no_entities": skipped_no_entities,
                "no_sections": skipped_no_sections,
            },
            "timing_seconds": {
                "total": total_seconds,
                "examples_total": total_example_seconds,
                "identification": identification_seconds,
                "identification_scoring": identification_scoring_seconds,
                "section_pool_build": section_pool_build_seconds,
                "section_rerank": section_rerank_seconds,
                "io_write": io_write_seconds,
            },
            "avg_seconds": {
                "per_prepared_example": avg_example_seconds,
                "identification_per_prepared_example": self._safe_average(
                    identification_seconds,
                    examples_prepared,
                ),
                "section_rerank_per_prepared_example": self._safe_average(
                    section_rerank_seconds,
                    examples_prepared,
                ),
            },
            "throughput_examples_per_sec": (
                float(examples_prepared) / total_seconds if total_seconds > 0 and examples_prepared > 0 else 0.0
            ),
        }
        if hasattr(self, "qwen") and hasattr(self.qwen, "image_cache_stats"):
            try:
                prepare_summary["vllm_image_cache"] = self.qwen.image_cache_stats()
            except Exception:
                pass
        self._last_prepare_timing_summary = prepare_summary
        self.logger.info(
            "Prepare timing | seen=%d prepared=%d total=%.3fs avg/prepared=%.3fs id=%.3fs score=%.3fs rerank=%.3fs io=%.3fs",
            examples_seen,
            examples_prepared,
            total_seconds,
            avg_example_seconds,
            identification_seconds,
            identification_scoring_seconds,
            section_rerank_seconds,
            io_write_seconds,
        )
        self._write_timing_summary(
            prepare_summary,
            getattr(self.config, "prepare_timing_summary_path", None),
        )
        self.logger.info("Prepared metadata for %d examples -> %s", len(records), metadata_path)
        return records

    def answer_from_metadata(
        self,
        metadata_path: str,
        output_path: str,
        use_image: bool = True,
        dataset_name: Optional[str] = None,
    ) -> None:
        start_time = time.perf_counter()
        rows = load_metadata(metadata_path)
        rows_seen = 0
        rows_answered = 0
        total_example_seconds = 0.0
        section_rerank_seconds = 0.0
        answer_generation_seconds = 0.0
        io_write_seconds = 0.0
        output_file_path = Path(output_path)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with output_file_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                rows_seen += 1
                example_start = time.perf_counter()
                question = row["question"]
                context_text = row.get("context_text")
                effective_row: Dict[str, object] = dict(row)
                effective_row["question"] = question
                if dataset_name is not None:
                    effective_row["dataset_name"] = dataset_name
                if context_text is not None:
                    effective_row["context_text"] = context_text
                section_title = row.get("context_section_title")
                section_index: Optional[int] = None
                section_scores: Optional[List[float]] = None
                section_source = "metadata"
                if row.get("context_mode") == "section":
                    sections: List[str] = row.get("reranked_sections", []) or []
                    if (
                        self.config.answer_rerank_sections
                        and sections
                        and getattr(self.section_selector, "model", None) is not None
                    ):
                        stage_start = time.perf_counter()
                        best_index, scores = self._pick_best_sections(
                            question or "",
                            sections,
                            row.get("image_path") if use_image else None,
                        )
                        section_rerank_seconds += time.perf_counter() - stage_start
                        if 0 <= best_index < len(sections):
                            context_text = sections[best_index]
                            section_title = extract_title_from_section(context_text)
                            section_index = best_index
                            section_scores = scores
                            section_source = "answer_reranker"
                    else:
                        if sections and context_text in sections:
                            section_index = sections.index(context_text)
                    if context_text is not None:
                        effective_row["context_text"] = context_text
                    if section_index is not None:
                        effective_row["selected_section_index"] = section_index
                    if section_title is not None:
                        effective_row["context_section_title"] = section_title
                stage_start = time.perf_counter()
                result = self._answer_single(
                    question=question,
                    context_text=context_text,
                    image_path=row.get("image_path") if use_image else None,
                    use_image=use_image,
                    metadata_row=effective_row,
                )
                answer_generation_seconds += time.perf_counter() - stage_start
                output_row = {
                    "data_id": row["data_id"],
                    "prediction": result.answer,
                    "raw_response": result.raw_response,
                    "context_mode": row.get("context_mode"),
                    "selected_url": row.get("selected_url"),
                    "selected_title": row.get("selected_title"),
                    "use_image": use_image,
                }
                if row.get("context_mode") == "section":
                    output_row.update(
                        {
                            "selected_section_text": context_text,
                            "selected_section_title": section_title,
                            "selected_section_index": section_index,
                            "selected_section_source": section_source,
                        }
                    )
                    if section_scores is not None:
                        output_row["section_scores"] = section_scores
                context_ref = section_title or row.get("context_source_url") or row.get("selected_url")
                self._log_answer_step(
                    data_id=row["data_id"],
                    question=question,
                    selected_entity=row.get("selected_title"),
                    context_mode=row.get("context_mode"),
                    context_ref=context_ref,
                    context_text=context_text,
                    answer=result.answer,
                    raw_response=result.raw_response,
                    section_source=section_source if row.get("context_mode") == "section" else None,
                    section_index=section_index,
                    section_scores=section_scores,
                )
                io_start = time.perf_counter()
                json.dump(output_row, output_file, ensure_ascii=False)
                output_file.write("\n")
                output_file.flush()
                io_write_seconds += time.perf_counter() - io_start
                rows_answered += 1
                total_example_seconds += time.perf_counter() - example_start
        total_seconds = time.perf_counter() - start_time
        avg_example_seconds = self._safe_average(total_example_seconds, rows_answered)
        answer_summary: Dict[str, Any] = {
            "phase": "answer",
            "metadata_path": metadata_path,
            "output_path": output_path,
            "use_image": use_image,
            "rows_seen": rows_seen,
            "rows_answered": rows_answered,
            "timing_seconds": {
                "total": total_seconds,
                "examples_total": total_example_seconds,
                "section_rerank": section_rerank_seconds,
                "answer_generation": answer_generation_seconds,
                "io_write": io_write_seconds,
            },
            "avg_seconds": {
                "per_answered_row": avg_example_seconds,
                "answer_generation_per_row": self._safe_average(answer_generation_seconds, rows_answered),
                "section_rerank_per_row": self._safe_average(section_rerank_seconds, rows_answered),
            },
            "throughput_rows_per_sec": (
                float(rows_answered) / total_seconds if total_seconds > 0 and rows_answered > 0 else 0.0
            ),
        }
        if hasattr(self, "qwen") and hasattr(self.qwen, "image_cache_stats"):
            try:
                answer_summary["vllm_image_cache"] = self.qwen.image_cache_stats()
            except Exception:
                pass
        self._last_answer_timing_summary = answer_summary
        self.logger.info(
            "Answer timing | rows=%d total=%.3fs avg/row=%.3fs gen=%.3fs rerank=%.3fs io=%.3fs",
            rows_answered,
            total_seconds,
            avg_example_seconds,
            answer_generation_seconds,
            section_rerank_seconds,
            io_write_seconds,
        )
        self._write_timing_summary(
            answer_summary,
            getattr(self.config, "answer_timing_summary_path", None),
        )
        self.logger.info("Generated answers for %d examples -> %s", rows_answered, output_path)


__all__ = ["TopKPipelineConfig", "TopKQwenPipeline"]
