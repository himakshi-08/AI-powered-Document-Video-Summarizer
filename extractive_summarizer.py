# extractive_summarizer.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load spaCy for sentence splitting
nlp = spacy.load("en_core_web_sm")

def load_classifier():
    with open("sentence_scorer_xgb.pkl", "rb") as f:
        return pickle.load(f)

def get_extractive_model(device: str = "cpu"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if device == "cpu":
        model._target_device = torch.device("cpu")
    else:
        model._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model


# === SPLIT TEXT INTO CHAPTER-LIKE SEGMENTS ===
def segment_text_into_sections(text: str, min_words_per_section: int = 1000) -> list:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    sections, buffer, buffer_len = [], "", 0
    for sentence in sentences:
        buffer += sentence + " "
        buffer_len += len(sentence.split())
        if buffer_len >= min_words_per_section:
            sections.append(buffer.strip())
            buffer, buffer_len = "", 0
    if buffer:
        sections.append(buffer.strip())

    # Merge short sections if they exceed ideal segment count
    total_words = len(text.split())
    ideal_count = max(1, total_words // 1500)
    if len(sections) > ideal_count:
        merged, temp, temp_len = [], "", 0
        avg_size = total_words // ideal_count
        for section in sections:
            temp += " " + section
            temp_len += len(section.split())
            if temp_len >= avg_size:
                merged.append(temp.strip())
                temp, temp_len = "", 0
        if temp:
            merged.append(temp.strip())
        return merged

    return sections


# === Split into sentences ===
def split_into_sentences(text):
    return [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip()) > 20]


# === Extractive summarization for a single chunk ===
def summarize_text(text, model, clf, max_sents=6, threshold=0.5):
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    embeddings = model.encode(sentences)

    try:
        probs = clf.predict_proba(embeddings)[:, 1]
    except AttributeError:
        probs = clf.predict(embeddings)

    ranked = sorted(zip(sentences, probs, embeddings), key=lambda x: x[1], reverse=True)

    selected = []
    selected_vecs = []
    for sent, score, vec in ranked:
        if len(selected) >= max_sents:
            break
        if selected_vecs:
            sims = cosine_similarity([vec], selected_vecs)[0]
            if max(sims) > 0.8:
                continue
        selected.append((sent, score))
        selected_vecs.append(vec)

    # Restore original order for coherence
    original_order = {s: i for i, s in enumerate(sentences)}
    selected.sort(key=lambda x: original_order.get(x[0], float('inf')))

    return [s for s, _ in selected]


# === New: Generate full document extractive summary with subsections ===
def summarize_extractively_with_sections(text, model, clf, max_sents=6, threshold=0.5) -> str:
    word_count = len(text.split())
    sections = segment_text_into_sections(text) if word_count > 1500 else [text]

    summaries = []

    for idx, section in enumerate(sections):
        summary_sents = summarize_text(section, model, clf, max_sents=max_sents, threshold=threshold)

        if not summary_sents:
            continue

        title = f"Section {idx+1}"  # Basic title (no title generation here)
        summary = "\n".join(f"• {s}" for s in summary_sents)

        summaries.append(f"### 📘 {title}\n\n{summary}\n")

    if not summaries:
        return "⚠️ No key points identified."
    
    return "\n\n".join(summaries).strip()
