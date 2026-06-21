def reconstruct_wiki_article(knowledge_entry):
    """Reconstruct a compact wiki article string from a KB entry."""
    title = getattr(knowledge_entry, "title", "")
    section_titles = list(getattr(knowledge_entry, "section_titles", []) or [])
    section_texts = list(getattr(knowledge_entry, "section_texts", []) or [])

    article = "# Wiki Article: " + str(title) + "\n"
    for section_title, section_text in zip(section_titles, section_texts):
        section_title = str(section_title)
        if "external link" in section_title.lower() or "reference" in section_title.lower():
            continue
        article += "\n## Section Title: " + section_title + "\n" + str(section_text)
    return article


def reconstruct_wiki_sections(knowledge_entry, section_index=-1):
    """Return KB section texts; kept for compatibility with older scripts."""
    section_titles = list(getattr(knowledge_entry, "section_titles", []) or [])
    section_texts = list(getattr(knowledge_entry, "section_texts", []) or [])

    kept_sections = []
    evidence_section = ""
    for idx, (section_title, section_text) in enumerate(zip(section_titles, section_texts)):
        section_title = str(section_title)
        section_text = str(section_text)
        if idx == int(section_index):
            evidence_section = section_text
            continue
        if "external links" in section_title.lower() or "references" in section_title.lower():
            continue
        kept_sections.append(section_text)

    if section_index != -1:
        return evidence_section, kept_sections
    return kept_sections
