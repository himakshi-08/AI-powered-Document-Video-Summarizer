#main_app.py

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
import time
import sys
import importlib.util
import re
import html
import torch



from abstractive_model import get_summarizer
from core_scraper import scrape_article
from core_transcriber import transcribe_video
from extractive_summarizer import load_classifier, summarize_text, get_extractive_model
from extractive_summarizer import summarize_extractively_with_sections
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Text Summarization Suite",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui_styles import inject_custom_css
inject_custom_css()

def import_module_from_path(module_name, file_path):
    """Dynamically import a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    current_dir = Path(__file__).parent
    text_extraction = import_module_from_path("text_extraction", current_dir / "text_extraction.py")
except Exception as e:
    st.error(f"Failed to load modules: {str(e)}")
    st.stop()


def styled_header(title: str, divider_color: str = "rainbow"):
    """Custom styled header with divider"""
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)
    st.divider()

def progress_with_message(message: str):
    """Animated progress with message"""
    with st.spinner(message):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        progress_bar.empty()

def remove_unnecessary_backslashes(text: str) -> str:
    """Remove Markdown-style backslashes before punctuation like '-' or '.'"""
    return re.sub(r'\\([\-\.])', r'\1', text)

# Calling and cacheing extractive model
@st.cache_resource(show_spinner="Loading extractive model...")
def get_extractive_components(device="cpu"):
    clf = load_classifier()
    model = get_extractive_model(device)
    model.device_str = model._target_device.type
    return clf, model

def summarize_extractive(text, threshold=0.5, max_sents=5, clf=None, model=None):
    if clf is None or model is None:
        device = st.session_state.get("selected_device", "cpu").lower()
        clf, model = get_extractive_components(device=device)


    summary_sents = summarize_text(text, model, clf, threshold=threshold, max_sents=max_sents)

    if not summary_sents:
        return "⚠️ No key sentences selected."
    
    return "\n".join(f"• {s}" for s in summary_sents)


# Processing module
class DocumentProcessor:
    """Wrapper for document processing using text_extraction.py"""

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Use the function from text_extraction.py"""
        return text_extraction.extract_text_from_pdf(file_path)

    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Use the function from text_extraction.py"""
        return text_extraction.extract_text_from_docx(file_path)



class TextSummarizationApp:
    def __init__(self):
        self.input_method = None
        self.processed_content = None

    def _save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temp location"""
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name

    def process_input(self) -> Optional[Dict[str, Any]]:
        """Main processing pipeline based on input method"""
        try:
            if self.input_method == "text":
                return {
                    "text": st.session_state.direct_text,
                    "metadata": {"source": "direct_input"}
                }

            elif self.input_method == "url":
                with st.spinner("Scraping website content..."):
                    scraped = scrape_article(st.session_state.url_input)
                    return {
                        "text": scraped["text"],
                        "metadata": {
                            "title": scraped["title"],
                            "authors": scraped["authors"],
                            "date": scraped["publish_date"],
                            "source": "web"
                        }
                    }

            elif self.input_method == "document":
                file_path = self._save_uploaded_file(st.session_state.doc_upload)
                try:
                    if file_path.lower().endswith('.pdf'):
                        text = DocumentProcessor.extract_text_from_pdf(file_path)
                    elif file_path.lower().endswith('.docx'):
                        text = DocumentProcessor.extract_text_from_docx(file_path)
                    else:
                        raise ValueError("Unsupported file type")

                    return {
                        "text": text,
                        "metadata": {
                            "filename": st.session_state.doc_upload.name,
                            "source": "document"
                        }
                    }
                finally:
                    try:
                        if file_path and os.path.exists(file_path):
                            os.unlink(file_path)
                    except Exception:
                        pass


            # In process_input(), update the audio/video section:
            elif self.input_method == "audio_video":
                if not st.session_state.get("audio_video_upload"):
                    st.error("No file uploaded!")
                    return None
        
                file_path = self._save_uploaded_file(st.session_state.audio_video_upload)
                try:
                    with st.spinner("Processing video/audio file..."):
                        text = transcribe_video(file_path)
                        return {
                            "text": text,
                            "metadata": {
                                "filename": st.session_state.audio_video_upload.name,
                                "source": "audio_video"
                            }
                        }
                except Exception as e:
                    st.error(f"Video processing failed: {str(e)}")
                    return None
                finally:
                    try:
                        if file_path and os.path.exists(file_path):
                            os.unlink(file_path)
                    except Exception:
                        pass


        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return None

    def display_summary_options(self):
        summary_type = st.radio("Choose summary type:", ["Abstractive", "Extractive"], index=0)
        summary_mode = None
        length = 3  # default

        if summary_type == "Abstractive":
            fast_mode = st.checkbox("⚡ Enable Fast Mode ", value=False)
            summary_mode = st.radio(
                "Summary mode:",
                ["Detailed", "Concise"],
                index=0,
                help="Detailed: long summary with full context\nConcise: short overview"
            )
        else:
           # Show slider only for extractive
            summary_len = st.select_slider("Summary Length", options=["Short", "Medium", "Long"], value="Medium")
            length_map = {"Short": 3, "Medium": 6, "Long": 10}
            length = length_map.get(summary_len, 6)
            fast_mode = False

        return {
            "type": summary_type.lower(),
            "mode": summary_mode.lower() if summary_mode else None,
            "length": length,
            "fast_mode": fast_mode
        }


    def generate_summary(self, text, options):
        source_type = st.session_state.processed_content["metadata"].get("source", "article")

        if options["type"] == "abstractive":
            device = st.session_state.get("selected_device", "cpu").lower()
            summarizer_instance = get_summarizer(
                source_type="transcript" if source_type in ["audio_video", "transcript"] else "article",
                fast_mode=options.get("fast_mode", False),
                device=device
            )


            st.sidebar.markdown(f"🖥 Model is using: `{summarizer_instance.device}`")
            selected_tier = st.session_state.get("selected_tier", "intermediate").lower()
            selected_mode = options["mode"]

            # Get the tier selected by the user
            selected_tier = st.session_state.get("selected_tier", "intermediate").lower()

            if not isinstance(text, str) or not text.strip():
                st.error("⚠️ No valid text to summarize.")
                return "⚠️ No text available."


            # Generate summary directly in the selected tier
            base_summary = summarizer_instance.summarize(
                text,
                tier=selected_tier,
                mode=selected_mode
            )

            # Store the generated tier version in session state
            if "paraphrased_summaries" not in st.session_state:
                st.session_state["paraphrased_summaries"] = {}

            st.session_state["paraphrased_summaries"][selected_tier] = base_summary
            st.session_state["current_tier"] = selected_tier

            return base_summary



        elif options["type"] == "extractive":
            device = st.session_state.get("selected_device", "cpu").lower()
            clf, model = get_extractive_components(device=device)

            summary_text = summarize_extractively_with_sections(
                text,
                model=model,
                clf=clf,
                max_sents=options["length"],
                threshold=0.5
            )
            st.sidebar.markdown(f"🧠 Extractive model running on: `{model._target_device.type}`")

            return summary_text





    def render_input_selector(self):
        st.markdown(
            """
            <h1 style='text-align: center; color: #1f4e79; font-size: 48px; font-family: "Segoe UI", sans-serif;'>
                AISum
            </h1>
            """,
            unsafe_allow_html=True
        )
        #Caption
        st.markdown(
            "<p style='text-align: center; color: #444; font-size: 18px;'>"
            "Condense your long content into short, meaningful summaries and be more productive."
            "</p>",
            unsafe_allow_html=True
        )
        input_method = st.radio(
            "Choose your input method:",
            ["Text", "URL", "Document", "Audio/Video"],
            horizontal=True,
            key="input_method_radio"
        ).lower().replace('/', '_').replace(' ', '_')

        if input_method == "text":
            st.text_area(
                "Enter your text directly:",
                key="direct_text",
                height=200,
                placeholder="Paste or type your content here..."
            )

        elif input_method == "url":
            st.text_input(
                "Enter article URL:",
                key="url_input",
                placeholder="https://example.com/article"
            )

        elif input_method == "document":
            st.file_uploader(
                "Upload document (PDF or DOCX):",
                type=["pdf", "docx"],
                key="doc_upload"
            )

        # Change this in render_input_selector():
        elif input_method == "audio_video": 
            st.file_uploader(
                "Upload audio/video file:",
                type=["mp3", "wav", "mp4", "mkv", "avi", "mov", "webm"],
                key="audio_video_upload"  
            )

        return input_method
    
    def preprocess(self, text: str) -> str:
        """Remove HTML tags, decode entities, normalize whitespace."""
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = html.unescape(text)  # Decode HTML entities
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
   

    def render_results(self, content: Dict[str, Any], summary: str, summary_options: Dict[str, Any]):
        """Render the processing results"""
        styled_header("Results")
        def normalize_summary_text(text: str) -> str:
            text = text.replace("\r", "")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.replace("\u200b", "").replace("\u00AD", "")

            from abstractive_model import fix_spacing  # use the global fix_spacing
            text = fix_spacing(text)  # spacing cleanup
            text = remove_unnecessary_backslashes(text)  # <<< NEW: strip stray '\'

            return text.strip()




        
        def escape_markdown(text: str) -> str:
            escape_chars = r"\`*_{}[]()#+-.!|>~^"
            return "".join("\\" + c if c in escape_chars else c for c in text)


        tab1, tab2, tab3 = st.tabs(["Original Content", "Processed Text", "Summary"])

        with tab1:
            st.subheader("Original Content")
            if content["metadata"]["source"] == "web":
                st.markdown(f"**Title:** {content['metadata']['title']}")
                st.markdown(f"**Authors:** {', '.join(content['metadata']['authors']) if content['metadata']['authors'] else 'Unknown'}")
                st.markdown(f"**Date:** {content['metadata']['date'] or 'Unknown'}")
            elif content["metadata"]["source"] in ["document", "audio"]:
                st.markdown(f"**Filename:** {content['metadata']['filename']}")

            st.text_area(
                "Content",
                content["text"],
                height=300,
                label_visibility="collapsed"
            )

        with tab2:
            st.subheader("Cleaned Text")
            cleaned = self.preprocess(content["text"])
            st.text_area(
                "Cleaned",
                cleaned,
                height=300,
                label_visibility="collapsed"
            )

        with tab3:
            tier_display = st.session_state.get("current_tier", "intermediate").capitalize()
            mode_value = summary_options.get("mode") or "detailed"
            mode_display = mode_value.capitalize()

            st.subheader(f"Summary ({tier_display}, {mode_display})")

            chapter_prefix = "### 📘"
            
            # ✅ Determine final_summary based on summary type
            if summary_options.get("type") == "abstractive":
                if "paraphrased_summaries" not in st.session_state:
                    st.session_state["paraphrased_summaries"] = {}

                if "current_tier" not in st.session_state:
                    st.session_state["current_tier"] = "intermediate"

                final_summary = st.session_state["paraphrased_summaries"].get(
                    st.session_state["current_tier"], summary
                )
            else:
                # Extractive summaries don’t use paraphrasing
                final_summary = summary

            # ✅ Show dropdown summary if structured
            if len(content["text"].split()) > 500 and chapter_prefix in final_summary:
                sections = final_summary.strip().split(chapter_prefix)
                for sec in sections:
                    if not sec.strip():
                        continue
                    lines = sec.strip().split("\n", 1)
                    title = lines[0].strip()
                    body = lines[1].strip() if len(lines) > 1 else ""
                    with st.expander(f"{chapter_prefix} {title}"):
                        # Markdown: escape to prevent accidental formatting
                        safe_text = escape_markdown(normalize_summary_text(body))
                        st.markdown(safe_text, unsafe_allow_html=False)
            else:
                # Text area: DO NOT escape markdown
                safe_text = normalize_summary_text(final_summary)
                st.text_area(
                    "Summary",
                    safe_text,
                    height=300,
                    label_visibility="collapsed"
                )



            st.download_button("📥 Download Summary", data=final_summary, file_name="summary.txt", mime="text/plain")




    
    def run(self):
        """Main application flow"""
        # Sidebar with app info
        with st.sidebar:
            st.markdown("## 📝 Summarization Suite")

            st.markdown("""
                Transform your content into concise summaries with AI-powered processing.

                Supported Inputs:
                - Direct text input
                - Web articles (URL)
                - Documents (PDF, DOCX)
                - Audio/Video files (MP3, WAV, MP4, etc.)
            """)
            st.divider()
            # In render_input_selector() or run(), inside sidebar block
            st.sidebar.markdown("### Inference Device")
            st.session_state.selected_device = st.sidebar.radio("Select device:", ["CPU", "GPU"], index=1)
            st.divider()
            st.markdown("### How to use:")
            st.markdown("""
                1. Select your input method
                2. Provide your content
                3. Customize summary options
                4. Generate and view results
            """)

        
        # Main content area
        self.input_method = self.render_input_selector()

        # Use session state to track processing stage
        if 'processed_content' not in st.session_state:
            st.session_state.processed_content = None
        if 'show_summary_options' not in st.session_state:
            st.session_state.show_summary_options = False

        if st.button("Process Content", type="primary"):
            if ((self.input_method == "text" and not st.session_state.get("direct_text")) or \
               (self.input_method == "url" and not st.session_state.get("url_input")) or \
               (self.input_method == "document" and not st.session_state.get("doc_upload")) or \
               (self.input_method == "audio_video" and not st.session_state.get("audio_video_upload"))):
                st.warning("Please provide input content first!")
                return

            with st.spinner("Processing your content..."):
                st.session_state.processed_content = self.process_input()
                st.session_state.show_summary_options = True

        # Show summary options if processing is complete
        if st.session_state.show_summary_options and st.session_state.processed_content:
            summary_options = self.display_summary_options()
            if summary_options["type"] == "abstractive":
                st.session_state.selected_tier = st.radio(
                    "Select Summary Complexity Tier:",
                    ["Simple", "Intermediate", "Technical"],
                    index=1,
                    horizontal=True,
                    key="tier_radio"
                )


            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    try:
                        # Before calling generate_summary()
                        if (
                            not isinstance(st.session_state.processed_content, dict)
                            or "text" not in st.session_state.processed_content
                            or not isinstance(st.session_state.processed_content["text"], str)
                            or not st.session_state.processed_content["text"].strip()
                        ):
                            st.error("⚠️ No valid content to summarize. Please process content first.")
                            return

                        summary = self.generate_summary(
                            st.session_state.processed_content["text"],
                            summary_options
                        )
                        self.render_results(st.session_state.processed_content, summary, summary_options)
                        # Reset the flags after showing results
                        st.session_state.show_summary_options = False
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")

# Run 
if __name__ == "__main__":
    app = TextSummarizationApp()
    app.run()
