"""  Serves as the retriever for the EchoSight.
"""

import os
import torch
import tqdm
import pickle
import json
from typing import List, Optional
from transformers import AutoModel, AutoProcessor, CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
import faiss
import numpy as np
from faiss import write_index, read_index
import faiss.contrib.torch_utils
try:
    from spacy.lang.en import English  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    English = None  # type: ignore


DEFAULT_SECTION_TOKEN_BUDGET = 300
_SPACY_SENTENCIZER = None


def _get_spacy_sentencizer():
    global _SPACY_SENTENCIZER
    if _SPACY_SENTENCIZER is None and English is not None:
        nlp = English()
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        _SPACY_SENTENCIZER = nlp
    return _SPACY_SENTENCIZER


def _chunk_wikipedia_content(text: str, max_tokens: Optional[int] = None) -> List[str]:
    """Split raw wikipedia content into passages roughly matching ReflectiVA chunking."""
    if not text:
        return []
    if max_tokens is None:
        max_tokens = DEFAULT_SECTION_TOKEN_BUDGET
    sentencizer = _get_spacy_sentencizer()
    chunks: List[str] = []
    if sentencizer is not None:
        doc = sentencizer(text)
        current_sentences: List[str] = []
        token_budget = 0
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sent_len = len(sent)
            if token_budget + sent_len > max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sent_text]
                token_budget = sent_len
            else:
                current_sentences.append(sent_text)
                token_budget += sent_len
        if current_sentences:
            chunks.append(" ".join(current_sentences))
        if chunks:
            return chunks

    # Fallback when spaCy is unavailable
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        chunks.extend(paragraphs)
    else:
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        current_sentences = []
        token_budget = 0
        for sent in sentences:
            sent_len = len(sent.split())
            if token_budget + sent_len > max_tokens and current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sent]
                token_budget = sent_len
            else:
                current_sentences.append(sent)
                token_budget += sent_len
        if current_sentences:
            chunks.append(" ".join(current_sentences))
    return chunks or [text.strip()]


class KnowledgeBase:
    """Knowledge base for EchoSight system.

    Returns:
        KnowledgeBase
    """

    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)

    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None

    def load_knowledge_base(self):
        """Load the knowledge base."""
        raise NotImplementedError


class WikipediaKnowledgeBase(KnowledgeBase):
    """Knowledge base for EchoSight."""

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        super().__init__(knowledge_base_path)
        self.knowledge_base = []

    def load_knowledge_base_full(
        self, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base from multiple score files.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The parent folder path to the vision similarity scores to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None

        if visual_attr is not None:
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if scores_path is not None:
            # get the image scores for each entry
            # get all the *.pkl files in the scores_path
            print("Loading knowledge base score from {}.".format(scores_path))
            import glob

            score_files = glob.glob(scores_path + "/*.pkl")
            image_scores = {}
            for score_file in tqdm.tqdm(score_files):
                try:
                    with open(score_file, "rb") as f:
                        image_scores.update(pickle.load(f))
                except:
                    raise FileNotFoundError(
                        "Image scores not found, which should be a url or path to a pickle file."
                    )
            print("Loaded {} image scores.".format(len(image_scores)))
            for wiki_url, entry in knowledge_base_dict.items():
                entry = dict(entry)
                entry.setdefault("url", wiki_url)
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                entry = dict(entry)
                entry.setdefault("url", wiki_url)
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base

    def load_knowledge_base(self, image_dict=None, scores_path=None, visual_attr=None):
        """Load the knowledge base.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None
        if visual_attr is not None:
            # raise NotImplementedError
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if (
            scores_path is not None
        ):  # TODO: fix the knowledge base and visual_attr is None:
            # get the image scores for each entry
            print("Loading knowledge base score from {}.".format(scores_path))
            try:
                with open(scores_path, "rb") as f:
                    image_scores = pickle.load(f)
            except:
                raise FileNotFoundError(
                    "Image scores not found, which should be a url or path to a pickle file."
                )
            for wiki_url, entry in knowledge_base_dict.items():
                entry = dict(entry)
                entry.setdefault("url", wiki_url)
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                entry = dict(entry)
                entry.setdefault("url", wiki_url)
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base


class WikipediaKnowledgeBaseEntry:
    """Knowledge base entry for EchoSight.

    Returns:
    """

    def __init__(self, entry_dict, visual_attr=None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            entry_dict: The dictionary containing the knowledge base entry.
            visual_attr: The visual attribute. Deprecated in the current version.

        Returns:
            KnowledgeBaseEntry
        """
        self.title = (
            entry_dict.get("title")
            or entry_dict.get("wikipedia_title")
            or entry_dict.get("page_title")
            or ""
        )
        url = entry_dict.get("url") or entry_dict.get("wikipedia_url")
        if not url and self.title:
            canonical_title = self.title.replace(" ", "_")
            url = f"https://en.wikipedia.org/wiki/{canonical_title}"
        self.url = url or ""
        self.wikidata_id = entry_dict.get("wikidata_id")

        content = (
            entry_dict.get("wikipedia_content")
            or entry_dict.get("content")
            or entry_dict.get("wikipedia_summary")
            or entry_dict.get("summary")
            or ""
        )

        section_titles = entry_dict.get("section_titles") or []
        section_texts = entry_dict.get("section_texts") or []

        if not section_titles or not section_texts:
            chunked_sections = _chunk_wikipedia_content(content)
            if chunked_sections:
                section_texts = chunked_sections
                section_titles = (
                    ["Summary"]
                    if len(chunked_sections) == 1
                    else [f"Section {idx + 1}" for idx in range(len(chunked_sections))]
                )

        self.section_titles = list(section_titles)
        self.section_texts = list(section_texts)

        def _ensure_list(value):
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value]

        image_urls = entry_dict.get("image_urls")
        if image_urls is None:
            image_urls = entry_dict.get("wikipedia_image_url")
        image_urls = [url for url in _ensure_list(image_urls) if url]

        if not self.section_titles and image_urls:
            self.section_titles = ["Summary"]
            self.section_texts = [content]

        image_descriptions = entry_dict.get("image_reference_descriptions")
        image_descriptions = _ensure_list(image_descriptions)

        image_section_indices = entry_dict.get("image_section_indices")
        image_section_indices = _ensure_list(image_section_indices)

        if len(image_descriptions) < len(image_urls):
            image_descriptions.extend([""] * (len(image_urls) - len(image_descriptions)))
        if len(image_section_indices) < len(image_urls):
            default_index = 0
            image_section_indices.extend(
                [default_index] * (len(image_urls) - len(image_section_indices))
            )

        self.image_urls = image_urls
        self.image_reference_descriptions = image_descriptions
        self.image_section_indices = image_section_indices
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr


class Retriever:
    """Retriever parent class for EchoSight."""

    def __init__(self, model=None):
        """Initialize the Retriever class.

        Args:
            model: The model to use for retrieval.
        """
        self.model = model

    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        raise NotImplementedError

    def retrieve_image(self, image):
        """Retrieve the image.

        Args:
            image: The image to retrieve.
        """
        raise NotImplementedError


class ClipRetriever(Retriever):
    """Image Retriever with CLIP-based VIT."""

    def __init__(self, model="clip", device="cpu"):
        """Initialize the ClipRetriever class.

        Args:
            model: The model to use for retrieval. Should be 'clip' or 'eva-clip'.
            device: The device to use for retrieval.
        """
        super().__init__(model)
        self.model_type = model
        requested_device = str(device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to(requested_device).eval()
            self.processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip":
            self.model = AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            self.model.to(requested_device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        self.device = requested_device
        self.knowledge_base = None

    def load_knowledge_base(
        self, knowledge_base_path, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        self.knowledge_base.load_knowledge_base(
            image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
        )
        # if scores_path is a folder, then load all the scores in the folder, otherwise, load the single score file

    def save_knowledge_base_faiss(
        self,
        knowledge_base_path,
        image_dict=None,
        scores_path=None,
        visual_attr=None,
        save_path=None,
    ):
        """Save the knowledge base with faiss index.

        Args:
            knowledge_base_path: The knowledge base to load.
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
            save_path: The path to save the faiss index.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        if scores_path[-4:] == ".pkl":
            print("Loading knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        else:
            print("Loading full knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base_full(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        self.prepare_faiss_index()
        self.save_faiss_index(save_path)

    def retrieve_image(
        self, image, top_k=100, pool_method="max", return_entry_list=False
    ):
        raise NotImplementedError("Pleas use retrieve_image_faiss or retrieve_image_faiss_batch.")
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs.to(self.device)
        outputs = self.model(**inputs)
        image_score = outputs.pooler_output
        # get the top k images in kb by cosine similarity
        kb_image_similarities = {}
        for i in range(len(self.knowledge_base)):
            kb_image_similarity = []
            wiki_url = self.knowledge_base[i].url
            image_urls = list(self.knowledge_base[i].score.keys())
            scores = [
                torch.tensor(self.knowledge_base[i].score[url]).to(self.device)
                for url in image_urls
            ]
            if len(scores) == 0:
                continue
            scores_matrix = torch.stack(scores, dim=0)
            kb_image_similarity = torch.cosine_similarity(
                image_score.unsqueeze(0), scores_matrix, dim=-1
            ).squeeze(0)
            if pool_method == "max":
                # get the max score
                # kb_image_similarity = torch.max(kb_image_similarity, dim=0)[0]
                # get the max score's index in the url list
                max_similarity_index = torch.argmax(kb_image_similarity, dim=0)
                max_similarity = kb_image_similarity[max_similarity_index]
                max_similarity_url = image_urls[max_similarity_index]
            else:
                raise NotImplementedError("Only max pooling is implemented.")
            # add key to the dict
            if wiki_url not in kb_image_similarities:
                kb_image_similarities[wiki_url] = {}
            kb_image_similarities[wiki_url].update(
                {"similarity": max_similarity.item()}
            )
            kb_image_similarities[wiki_url].update({"knowledge_base_index": i})
            kb_image_similarities[wiki_url].update(
                {"image_url": max_similarity_url}
            )  # TODO bug to fix, if multiple images of same entry are hit

        ranked_list = sorted(
            kb_image_similarities.items(),
            key=lambda x: x[1]["similarity"],
            reverse=True,
        )
        # get the top k images' urls
        top_k_entries = []
        if return_entry_list:
            for i in range(top_k):
                top_k_entries.append(
                    self.knowledge_base[ranked_list[i][1]["knowledge_base_index"]]
                )
            return top_k_entries
        for i in range(top_k):
            top_k_entries.append(
                {
                    "url": ranked_list[i][0],
                    "knowledge_base_index": ranked_list[i][1]["knowledge_base_index"],
                    "image_url": ranked_list[i][1]["image_url"],
                    "similarity": ranked_list[i][1]["similarity"],
                    "kb_entry": self.knowledge_base[
                        ranked_list[i][1]["knowledge_base_index"]
                    ],
                }
            )

        return top_k_entries

    def save_faiss_index(self, save_index_path):
        """Save the faiss index.
        
        Args:
            save_index_path: The path to save the faiss index.
        """
        if save_index_path is not None:
            write_index(self.faiss_index, os.path.join(save_index_path, "kb_index.faiss"))
            with open(os.path.join(save_index_path, "kb_index_ids.pkl"), "wb") as f:
                pickle.dump(self.faiss_index_ids, f)

    @staticmethod
    def _url_aliases(url):
        aliases = {url}
        if url.startswith("https://en.m.wikipedia.org/wiki/"):
            aliases.add(url.replace("https://en.m.wikipedia.org/wiki/", "https://en.wikipedia.org/wiki/", 1))
        if url.startswith("http://en.m.wikipedia.org/wiki/"):
            aliases.add(url.replace("http://en.m.wikipedia.org/wiki/", "https://en.wikipedia.org/wiki/", 1))
        if url.startswith("http://en.wikipedia.org/wiki/"):
            aliases.add(url.replace("http://en.wikipedia.org/wiki/", "https://en.wikipedia.org/wiki/", 1))
        return aliases

    def _load_knn_ids_from_urls(self, knn_json_path):
        url_to_kb_index = {}
        for kb_idx, entry in enumerate(self.knowledge_base):
            for alias in self._url_aliases(entry.url):
                url_to_kb_index.setdefault(alias, kb_idx)

        with open(knn_json_path, "r") as f:
            knn_values = json.load(f)

        missing = 0
        faiss_index_ids = []
        for value in knn_values:
            url = value[0] if isinstance(value, list) else value.get("url")
            kb_idx = None
            if url:
                for alias in self._url_aliases(url):
                    if alias in url_to_kb_index:
                        kb_idx = url_to_kb_index[alias]
                        break
            if kb_idx is None:
                missing += 1
                kb_idx = -1
            faiss_index_ids.append(kb_idx)

        if missing:
            print(f"Warning: {missing} FAISS entries from knn.json were not found in the loaded KB.")
        return faiss_index_ids

    def _move_index_to_gpu_if_available(self):
        if not self.device.startswith("cuda") or not hasattr(faiss, "StandardGpuResources"):
            return
        gpu_id = 0
        if ":" in self.device:
            gpu_id = int(self.device.split(":", 1)[1])
        try:
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, self.faiss_index)
        except Exception as exc:
            print(f"Could not move FAISS index to GPU {gpu_id}: {exc}. Using CPU index.")

    def load_faiss_index(self, load_index_path):
        """Load the faiss index.
        
        Args:
            load_index_path: The path to load the faiss index.
        """
        if load_index_path is not None:
            kb_index_path = os.path.join(load_index_path, "kb_index.faiss")
            kb_ids_path = os.path.join(load_index_path, "kb_index_ids.pkl")
            knn_index_path = os.path.join(load_index_path, "knn.index")
            knn_json_path = os.path.join(load_index_path, "knn.json")

            if os.path.exists(kb_index_path) and os.path.exists(kb_ids_path):
                self.faiss_index = read_index(kb_index_path)
                with open(kb_ids_path, "rb") as f:
                    self.faiss_index_ids = pickle.load(f)
            elif os.path.exists(knn_index_path) and os.path.exists(knn_json_path):
                self.faiss_index = read_index(knn_index_path)
                self.faiss_index_ids = self._load_knn_ids_from_urls(knn_json_path)
            else:
                raise FileNotFoundError(
                    f"No supported FAISS index found in {load_index_path}. "
                    "Expected kb_index.faiss/kb_index_ids.pkl or knn.index/knn.json."
                )

            self._move_index_to_gpu_if_available()

            print("Faiss index loaded with {} entries.".format(self.faiss_index.ntotal))
        return

    def prepare_faiss_index(self):
        """Prepare the faiss index from scores in the knowledge base."""
        # use the knowledge base's score element to build the index
        # get the image scores for each entry
        scores = [
            score for entry in self.knowledge_base for score in entry.score.values()
        ]
        score_ids = [
            i
            for i in range(len(self.knowledge_base))
            for j in range(len(self.knowledge_base[i].score))
        ]
        # import ipdb; ipdb.set_trace()
        index = faiss.IndexFlatIP(scores[0].shape[0])
        # res = faiss.StandardGpuResources()
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        np_scores = np.array(scores)
        np_scores = np_scores.astype(np.float32)
        faiss.normalize_L2(np_scores)
        index.add(np_scores)
        self.faiss_index = index
        self.faiss_index_ids = score_ids
        print("Faiss index built with {} entries.".format(index.ntotal))

        return


    def built_text_embedding(self, text_faiss_path):
        """Build the text mathcing faiss index from the knowledge base.
        
        Score is calculated by cosine similarity between the image and article text embeddings.
        
        Args:
            text_faiss_path: The path to save the text faiss index.
        """
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        kb_text = []
        for entry in self.knowledge_base:
            text = entry.title 
            for section in entry.section_texts:
                text += "\n" + section 
                break# only use the first section
            kb_text.append(text)
        inputs = tokenizer(kb_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        batch_size = 512
        outputs = []
        for i in range(0, len(kb_text), batch_size):
            text_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
            output = self.model.get_text_features(**text_inputs)
            outputs.extend(output.cpu().detach().numpy())
        # build the faiss index
        index = faiss.IndexFlatIP(outputs[0].shape[0])
        np_outputs = np.array(outputs)
        np_outputs = np_outputs.astype(np.float32)
        faiss.normalize_L2(np_outputs)
        index.add(np_outputs)
        self.faiss_index = index
        self.faiss_index_ids = [i for i in range(len(kb_text))]
        self.save_faiss_index(text_faiss_path)
        return
    
    @torch.no_grad()
    def retrieve_image_faiss(
        self, image, top_k=100, pool_method="max", return_entry_list=False
    ):
        """Retrieve the top K similar images from the knowledge base using faiss.

        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        if self.model_type == "clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.get_image_features(inputs)
        elif self.model_type == "eva-clip":
            # EVA-CLIP Process the input image
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.encode_image(inputs)
        assert self.faiss_index and self.faiss_index_ids is not None
        query = image_score.float()
        query = torch.nn.functional.normalize(query, dim=-1)
        search_k = min(top_k, self.faiss_index.ntotal)
        if search_k <= 0:
            return []

        # NOTE:
        # In some FAISS+torch builds, querying GPU index with torch tensors can return
        # corrupted indices (very large int64 values). Converting to numpy avoids that path.
        query_np = query.detach().cpu().numpy().astype(np.float32, copy=False)
        D, I = self.faiss_index.search(query_np, search_k)
        top_k_entries = []
        for i in range(search_k):  # for each image in the top k
            faiss_pos = int(I[0][i])
            if faiss_pos < 0 or faiss_pos >= len(self.faiss_index_ids):
                continue
            kb_idx = int(self.faiss_index_ids[faiss_pos])
            if kb_idx < 0 or kb_idx >= len(self.knowledge_base):
                continue
            if return_entry_list:
                print("Retrieving entry for index {} with similarity {}".format(faiss_pos, D[0][i]))
                top_k_entries.append(self.knowledge_base[kb_idx])
            else:
                # find the first knowledge base entry that contains the image
                index_id = kb_idx
                # return the index of the first element in faiss_index_ids that is equal to index_id
                start_id = self.faiss_index_ids.index(index_id)
                offset = faiss_pos - start_id
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[kb_idx].url,
                        "knowledge_base_index": kb_idx,
                        "image_url": self.knowledge_base[kb_idx].image_urls[offset],
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[kb_idx],
                    }
                )
        return top_k_entries

    @torch.no_grad()
    def retrieve_image_faiss_batch(self, images, top_k=100, return_entry_list=False):
        """Retrieve the top K similar images from the knowledge base using faiss in batch.

        Args:
            images: The images to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.

        Returns:
            list: Top k entries, every entry is a dict of (url, kb_index, similarity)
        """
        # Process the input image
        if self.model_type == "clip":
            # CLIP Process the input image
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs.to(self.device)
            outputs = self.model(**inputs)
            image_scores = outputs.pooler_output
        elif self.model_type == "eva-clip":
            # EVA-CLIP Process the input image
            inputs = (
                self.processor(images=images, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_scores = self.model.encode_image(inputs)
        assert self.faiss_index and self.faiss_index_ids is not None
        query = image_scores.float()
        query = torch.nn.functional.normalize(query, dim=-1)
        search_k = min(top_k, self.faiss_index.ntotal)
        if search_k <= 0:
            return []
        query_np = query.detach().cpu().numpy().astype(np.float32, copy=False)
        Ds, Is = self.faiss_index.search(query_np, search_k)
        top_k_list = []
        for D, I in zip(Ds, Is):
            top_k_entries = []
            for i in range(search_k):
                faiss_pos = int(I[i])
                if faiss_pos < 0 or faiss_pos >= len(self.faiss_index_ids):
                    continue
                kb_idx = int(self.faiss_index_ids[faiss_pos])
                if kb_idx < 0 or kb_idx >= len(self.knowledge_base):
                    continue
                if return_entry_list:
                    top_k_entries.append(self.knowledge_base[kb_idx])
                else:
                    top_k_entries.append(
                        {
                            "url": self.knowledge_base[kb_idx].url,
                            "knowledge_base_index": kb_idx,
                            "image_urls": self.knowledge_base[kb_idx].image_urls,
                            "similarity": D[i],
                            "kb_entry": self.knowledge_base[kb_idx],
                        }
                    )
            top_k_list.append(top_k_entries)

        return top_k_list
