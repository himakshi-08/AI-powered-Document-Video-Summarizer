# abstractive_model.py
import re
import html
from typing import List, Optional
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from tqdm import tqdm
import streamlit as st

# -------------------------
# Lightweight spacing fixer (no model loads)
# -------------------------
def fix_spacing(text: str) -> str:
    """Small, fast spacing/zero-width character fixer. Safe to call anywhere."""
    if not text:
        return text or ""
    text = text.replace("\u200b", "").replace("\u00AD", "")
    # add space between digit and letter when missing: '1.2billion' -> '1.2 billion'
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    # add space between lower->UpperCase merges
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    # ensure punctuation followed by space
    text = re.sub(r'(?<=[\.,;:\?!])(?=[A-Za-z0-9])', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

class DocumentSummarizer:
    def __init__(
        self,
        source_type: str = "article",
        model_path_override: Optional[str] = None,
        use_fast_mode: bool = False,
        device: str = "gpu"
    ):
        # device
        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")

        # main summarizer model path
        model_path = model_path_override or ("./bart-finetuned-mediasum" if source_type == "transcript" else "./partial_model")

        # main tokenizer + model
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)

        # paraphraser lazy-loaded (small flan t5)
        self.para_tokenizer = None
        self.para_model = None
        self.PARA_MODEL_NAME = "google/flan-t5-base"

        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True

        # spaCy (sentence segmentation)
        self.spacy_nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        if "sentencizer" not in self.spacy_nlp.pipe_names:
            self.spacy_nlp.add_pipe("sentencizer")

        # generation defaults
        self.gen_args = {
            "max_length": 350,
            "min_length": 120,
            "num_beams": 4,
            "length_penalty": 1.5,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        }
        if use_fast_mode:
            self.gen_args.update({"num_beams": 1, "do_sample": False, "early_stopping": True})

    # -------------------------
    # Small guards + helpers
    # -------------------------
    def preprocess(self, text: Optional[str]) -> str:
        """Guard None and normalize HTML/whitespace."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_paraphraser(self):
        if self.para_model is not None and self.para_tokenizer is not None:
            return
        self.para_tokenizer = AutoTokenizer.from_pretrained(self.PARA_MODEL_NAME)
        self.para_model = AutoModelForSeq2SeqLM.from_pretrained(self.PARA_MODEL_NAME).to(self.device)
        if self.device.type == "cuda":
            try:
                self.para_model = self.para_model.half()
            except Exception:
                pass

    def _paraphrase_once(self, text: str, level: str) -> str:
        level = level.lower()
        if level not in ("simple", "technical"):
            return text

        self._load_paraphraser()

        if level == "simple":
            prompt = (
                "Rewrite the following text in clear, simple English. "
                "Use short sentences and common words so that a 12-year-old could easily understand. "
                "Avoid jargon. Expand ideas with simple explanations and examples where needed. "
                "Do not repeat sentences or phrases from the original:\n\n"
            )
            max_len, min_len = 350, 120  # increased min_len
        else:  # technical
            prompt = (
                "Rewrite the following text in a formal, highly technical style. "
                "Use advanced vocabulary, complex grammar structures, and domain-specific terminology. "
                "Vary the sentence structures and avoid repeating any exact sentences from the original. "
                "Expand the discussion with relevant elaboration if possible:\n\n"
            )
            max_len, min_len = 450, 250  # slightly longer

        # Tokenize & generate
        inputs = self.para_tokenizer(
            prompt + text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        outputs = self.para_model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )


        out = self.para_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Remove any prompt echo at the start
        if out.startswith(prompt.strip()):
            out = out[len(prompt.strip()):].strip()

        # Extra step: remove overly repeated phrases appearing 3+ times
        phrases = re.findall(r'\b[\w\s]{5,50}\b', out)  # phrases between 5 and 50 chars
        for phrase in set(phrases):
            if out.lower().count(phrase.lower()) >= 3:
                out = re.sub(re.escape(phrase), '', out, flags=re.IGNORECASE)

        # Deduplicate sentences
        sentences = re.split(r'(?<=[.!?])\s+', out)

        seen = set()
        unique_sentences = []
        for s in sentences:
            s_norm = re.sub(r'\s+', ' ', s.strip().lower())
            if s_norm and s_norm not in seen:
                seen.add(s_norm)
                unique_sentences.append(s.strip())

        # Group into paragraphs
        paragraphs, buf = [], []
        for sent in unique_sentences:
            buf.append(sent)
            if len(buf) >= 3:
                paragraphs.append(" ".join(buf))
                buf = []
        if buf:
            paragraphs.append(" ".join(buf))

        return fix_spacing("\n\n".join(paragraphs))



    # -------------------------
    # Segmentation + chunking
    # -------------------------
    def segment_topics(self, text: str) -> List[str]:
        min_words_per_section = 1000
        doc = self.spacy_nlp(text)
        sentences = [sent.text for sent in doc.sents]
        segments, buffer, buffer_len = [], "", 0
        for sent in sentences:
            wc = len(sent.split())
            buffer += sent + " "
            buffer_len += wc
            if buffer_len >= min_words_per_section:
                segments.append(buffer.strip()); buffer, buffer_len = "", 0
        if buffer:
            segments.append(buffer.strip())

        total_words = len(text.split())
        ideal_count = max(1, total_words // 1500)
        if len(segments) > ideal_count:
            avg_size = total_words // ideal_count
            new_segs, temp, temp_len = [], "", 0
            for seg in segments:
                temp += " " + seg
                temp_len += len(seg.split())
                if temp_len >= avg_size:
                    new_segs.append(temp.strip()); temp, temp_len = "", 0
            if temp:
                new_segs.append(temp.strip())
            return new_segs
        return segments

    def chunk_text(self, text: str, max_tokens: int = 1024, min_tokens: int = 700) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current, current_toks = [], "", 0
        for sent in sentences:
            tok = len(self.tokenizer.tokenize(sent))
            if current == "":
                current = sent; current_toks = tok; continue
            if current_toks + tok > max_tokens:
                if current_toks < min_tokens and chunks:
                    chunks[-1] = chunks[-1] + " " + current
                else:
                    chunks.append(current.strip())
                current = sent; current_toks = tok
            else:
                current += " " + sent; current_toks += tok
        if current.strip():
            if current_toks < min_tokens and chunks:
                chunks[-1] = chunks[-1] + " " + current.strip()
            else:
                chunks.append(current.strip())
        return chunks

    # -------------------------
    # Core summarization
    # -------------------------
    def summarize(self, text: Optional[str], tier: str = "intermediate", mode: str = "detailed") -> str:
        text = self.preprocess(text)
        if not text:
            return "❌ Empty input text."

        total_words = len(text.split())
        segments = self.segment_topics(text) if total_words > 1500 else [text]
        results = []
        tier_lower = tier.lower()
        if tier_lower == "intermediate":
            tier_lower = "technical"
        elif tier_lower == "technical":
            tier_lower = "intermediate"
        paraphrase_needed = tier_lower in ("simple", "technical")



        progress = tqdm(segments, desc="Summarizing Segments")
        try:
            for idx, seg in enumerate(progress):
                if mode == "detailed":
                    max_len, min_len = 350, 120
                else:
                    max_len, min_len = 150, 60
                if tier_lower == "technical":
                    max_len = max(max_len, 400); min_len = max(min_len, 200)

                chunks = self.chunk_text(seg)
                total_tokens = sum(len(self.tokenizer.tokenize(c)) for c in chunks)

                if len(chunks) == 1 and total_tokens <= 1024:
                    summ = self._summarize_single(seg, max_length=max_len, min_length=min_len)
                    chunk_summaries = [summ.strip()]
                else:
                    chunk_summaries = self._summarize_chunks_batch(chunks)

                section_text = "\n\n".join(s.strip() for s in chunk_summaries if s and s.strip())
                results.append({"title": self._generate_title(seg, idx), "summary": section_text})
        finally:
            progress.close()

        # combine
        # Always combine with headings, even if single section
        combined = "\n\n".join(
            f"### 📘 Chapter {i+1}: {sec['title']}\n\n{sec['summary']}"
            for i, sec in enumerate(results)
        )

        # Paraphrase each section individually (preserves headings)
        if paraphrase_needed:
            paraphrased_sections = []
            for i, sec in enumerate(results):
                heading = f"### 📘 Chapter {i+1}: {sec['title']}"
                body = sec["summary"]

                if tier_lower == "technical":
                    # Pass the entire segment — model will truncate to 1024 tokens internally
                    original_excerpt = segments[i]
                    enriched_input = (
                        f"Summary:\n{body}\n\n"
                        f"Additional context from original text:\n{original_excerpt}\n\n"
                        "Integrate relevant technical details, quantitative data, and causal relationships "
                        "from the context into the summary. Write in a precise, domain-specific style suitable for experts. "
                        "Avoid redundancy or unnecessary simplification."
                    )
                    para_body = self._paraphrase_once(enriched_input, tier_lower)

                else:  # simple tier
                    # Pass the entire segment so important facts are kept
                    original_excerpt = segments[i]
                    enriched_input = (
                        f"Summary:\n{body}\n\n"
                        f"Additional context from original text:\n{original_excerpt}\n\n"
                        "Rewrite in clear, simple English while keeping all important facts from the context. "
                        "Use short sentences and common words so that a 12-year-old could understand. "
                        "Avoid jargon and redundancy."
                    )
                    para_body = self._paraphrase_once(enriched_input, tier_lower)


                paraphrased_sections.append(f"{heading}\n\n{para_body}")
            combined = "\n\n".join(paraphrased_sections)



        combined = fix_spacing(combined)
        return combined


    # -------------------------
    # Model wrappers
    # -------------------------
    @torch.no_grad()
    def _summarize_single(self, text: str, max_length: Optional[int] = None, min_length: Optional[int] = None) -> str:
        inputs = self.tokenizer([text], max_length=1024, return_tensors="pt", truncation=True, padding=True).to(self.device)
        if self.device.type == "cuda":
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
        gen_args = dict(self.gen_args)
        if max_length: gen_args["max_length"] = max_length
        if min_length: gen_args["min_length"] = min_length
        summary_ids = self.model.generate(**inputs, **gen_args)
        out = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return out.strip()

    @torch.no_grad()
    def _summarize_chunks_batch(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(texts, max_length=1024, return_tensors="pt", truncation=True, padding=True).to(self.device)
        if self.device.type == "cuda":
            inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
        summary_ids = self.model.generate(**inputs, **self.gen_args)
        results = [self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in summary_ids]
        return results

    def _generate_title(self, segment: str, idx: int) -> str:
        doc = self.spacy_nlp(segment[:300])
        ents = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "GPE", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT")]
        if ents:
            return f"{ents[0].strip()} Overview"
        noun_phrases = [nc.text.strip() for nc in doc.noun_chunks][:2]
        if noun_phrases:
            return " - ".join(noun_phrases).title()
        return f"Topic {idx+1}"

# -------------------------
# Cached factory for Streamlit apps (prevents reloading on each rerun)
# -------------------------
@st.cache_resource
def get_summarizer(source_type: str = "article", fast_mode: bool = False, device: str = "cpu") -> DocumentSummarizer:
    return DocumentSummarizer(source_type=source_type, model_path_override=None, use_fast_mode=fast_mode, device=device)
