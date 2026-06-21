"""Helpers for mapping Qwen metadata rows to EchoSight generator inputs."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

from model.retriever import WikipediaKnowledgeBaseEntry

from .types import GeneratorInput


KnowledgeBaseIndex = Mapping[str, WikipediaKnowledgeBaseEntry]


def adapt_metadata_row(
    row: MutableMapping[str, object],
    kb_by_url: Optional[KnowledgeBaseIndex] = None,
) -> GeneratorInput:
    """Convert a raw metadata row into a :class:`GeneratorInput`.

    The adapter looks up the selected Wikipedia entry (if provided) and
    builds the most specific context we can pass to legacy EchoSight answer
    generators. When the knowledge base lookup fails we fall back to the
    literal context snippet stored in the metadata.
    """

    selected_url = _safe_str(row.get("selected_url"))
    context_source_url = _safe_str(row.get("context_source_url"))
    dataset_name = _safe_str(row.get("dataset_name"))
    entry: Optional[WikipediaKnowledgeBaseEntry] = None
    if kb_by_url:
        lookup_urls = []
        context_mode = _safe_str(row.get("context_mode"))
        # When answering with a specific section we want to preserve the exact
        # snippet stored in metadata; only fall back to KB lookups if we have no
        # usable context text.
        if context_mode == "section":
            context_text = row.get("context_text")
            if not _safe_str(context_text):
                lookup_urls.extend(url for url in (context_source_url, selected_url) if url)
        else:
            lookup_urls.extend(url for url in (context_source_url, selected_url) if url)

        for url in lookup_urls:
            entry = kb_by_url.get(url)
            if entry is not None:
                break

    context_text = _safe_str(row.get("context_text"))
    context_mode = _safe_str(row.get("context_mode"))

    raw_sections = row.get("reranked_sections")
    if isinstance(raw_sections, list):
        reranked_sections = raw_sections
    elif raw_sections is None:
        reranked_sections = []
    else:
        reranked_sections = [raw_sections]
    image_path = _safe_str(row.get("image_path"))

    if reranked_sections and not isinstance(reranked_sections, list):
        reranked_sections = [str(reranked_sections)]

    entry_section: Optional[str] = None
    if context_mode == "section" and context_text:
        entry_section = context_text
        # Ensure the downstream generator consumes the stored snippet instead of
        # expanding to a full article.
        entry = None
    elif entry is None and context_text:
        entry_section = context_text

    return GeneratorInput(
        question=_safe_str(row.get("question")) or "",
        dataset_name=dataset_name,
        entry=entry,
        entry_section=entry_section,
        context_mode=context_mode,
        context_text=context_text,
        reranked_sections=[str(section) for section in reranked_sections],
        image_path=image_path,
        raw=dict(row),
    )


def _safe_str(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    return str(value)
