# 🎯 AI-Powered Document & Video Summarizer

A comprehensive, intelligent summarization suite that leverages cutting-edge AI models to extract, condense, and summarize content from multiple sources including articles, videos, PDFs, and documents.

## 📋 Overview

This project combines **abstractive** and **extractive** summarization techniques powered by fine-tuned BART models and transformer-based neural networks to provide accurate, coherent summaries of diverse content types. Whether you're processing news articles, YouTube videos, academic papers, or business documents, this tool intelligently condenses information while preserving key insights.

### Key Features

✨ **Multi-Source Summarization**
- 🌐 Web articles (URL-based scraping)
- 📹 YouTube videos and video files
- 📄 PDF documents
- 📝 Word documents (.docx)
- 💬 Direct text input

🧠 **Dual Summarization Approaches**
- **Abstractive Summarization**: Uses fine-tuned BART models to generate natural, human-like summaries
- **Extractive Summarization**: Intelligently selects and highlights the most important sentences

🎨 **User-Friendly Interface**
- Built with Streamlit for seamless interaction
- Custom styling and responsive design
- Real-time processing with progress tracking

⚡ **Advanced Processing**
- Automatic text preprocessing and cleaning
- Intelligent text segmentation for long documents
- Multi-section summarization
- GPU acceleration support
- ROUGE evaluation metrics for quality assessment

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- CUDA 11.8+ (optional, for GPU acceleration)
- Miniconda or Anaconda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/himakshi-08/AI-powered-Document-Video-Summarizer.git
cd AI-powered-Document-Video-Summarizer
```

2. **Create and activate conda environment**
```bash
conda create -n bart-env python=3.10
conda activate bart-env
```

3. **Install core NLP packages**
```bash
conda install -c conda-forge -y transformers datasets sentencepiece accelerate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard
```

4. **Install preprocessing tools**
```bash
conda install -c conda-forge -y spacy beautifulsoup4 nltk scikit-learn
python -m spacy download en_core_web_sm
```

5. **Install additional dependencies**
```bash
conda install -c conda-forge -y tqdm pandas numpy
pip install streamlit>=1.25.0 torch requests yt-dlp ffmpeg-python
pip install openai-whisper rouge-score pdfplumber pymupdf python-docx
pip install newspaper3k requests selenium lxml_html_clean tokenizers evaluate
pip install sentence-transformers
```

6. **Install FFmpeg**
   - **Windows**: `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

### Quick Start

```bash
# Activate the conda environment
conda activate bart-env

# Run the Streamlit application
streamlit run main_app.py
```

The application will open at `http://localhost:8501`

## 📖 Usage Guide

### Abstractive Summarization
1. Select "Abstractive Summarizer" from the sidebar
2. Choose your content source:
   - Paste article URL
   - Upload video file or YouTube link
   - Upload PDF or Word document
   - Paste text directly
3. Adjust summarization parameters:
   - Summary length (min/max tokens)
   - Temperature (creativity level)
4. Click "Summarize" and receive AI-generated summary

### Extractive Summarization
1. Select "Extractive Summarizer" from the sidebar
2. Input text content
3. Choose summarization ratio
4. View highlighted key sentences
5. Optional: Generate per-section summaries for long documents

### Evaluate Quality
1. Use the "Evaluation" tab
2. Compare generated summary with reference text
3. View ROUGE metrics for assessment

## 🔧 Core Modules

### `abstractive_model.py`
Handles abstractive summarization using fine-tuned BART models. Features:
- Multiple source type support (article, transcript, document)
- Intelligent text chunking for long documents
- Fast mode for quick processing
- GPU/CPU device flexibility
- Advanced text cleaning and spacing fixes

### `extractive_summarizer.py`
Implements extractive summarization using:
- Sentence transformers for semantic understanding
- XGBoost classifier for sentence scoring
- cosine similarity for relevance detection
- Section-based summarization for structured content

### `core_scraper.py`
Web scraping functionality:
- URL validation and article extraction
- Metadata retrieval (title, authors, publication date)
- Error handling and retry logic
- Support for major news outlets

### `core_transcriber.py`
Video processing pipeline:
- YouTube video downloading (yt-dlp)
- Audio extraction (FFmpeg)
- Speech-to-text transcription (OpenAI Whisper)
- Automatic subtitle generation
- Support for local video files

### `text_extraction.py`
Document processing:
- PDF text extraction (PDFPlumber, PyMuPDF)
- Word document (.docx) parsing
- Table of contents extraction
- Metadata preservation

### `preprocess.py`
Text preprocessing utilities:
- Tokenization and sentence splitting
- Special character handling
- Stopword removal
- Text normalization

### `rouge_evaluation.py`
Quality assessment:
- ROUGE-1, ROUGE-2, ROUGE-L metrics
- Precision, Recall, F1-score calculation
- Summary quality benchmarking

## 🤖 Model Information

### Fine-Tuned BART Model
- **Base Model**: facebook/bart-large
- **Fine-tuned on**: MediaSum dataset (news articles and video transcripts)
- **Location**: `./bart-finetuned-mediasum/`
- **Performance**: Optimized for news and media content summarization
- **Checkpoint**: Includes 4500-step checkpoint for resume training

### Extractive Model
- **Sentence Embeddings**: all-MiniLM-L6-v2 (Sentence Transformers)
- **Classifier**: XGBoost (pre-trained on sentence importance)
- **Language Model**: spaCy en_core_web_sm

## 📊 Evaluation Metrics

The project includes comprehensive evaluation capabilities using ROUGE metrics:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

## 🛠️ Advanced Features

### Multi-GPU Support
```python
device = "gpu"  # Automatic CUDA detection
summarizer = DocumentSummarizer(device=device)
```

### Batch Processing
Process multiple documents efficiently with caching and optimization.

### Custom Model Training
Located in `training and additional files/`:
- `abstractive_model_training.py`: Fine-tune BART on custom datasets
- `extractive_model_training.ipynb`: Train extractive models
- `transcript_abstractive_model_training.py`: Specialized video transcript training

## 📈 Performance Considerations

- **GPU Acceleration**: ~10x faster with CUDA
- **Model Size**: ~1.6GB for BART-large
- **Max Input**: 1024 tokens (~4000 characters)
- **Batch Size**: Adjustable based on GPU memory

## 🔒 Security & Privacy

- No data sent to external servers (except YouTube downloads)
- All processing happens locally
- API service available in `training and additional files/api_service.py`

## 🐛 Troubleshooting

### FFmpeg Not Found
```bash
# Windows
choco install ffmpeg

# Or add to system PATH if already installed
```

### Whisper Model Download Issues
```bash
# Force re-download
python -c "import whisper; whisper.load_model('base')"
```

### Out of Memory (CUDA)
- Reduce batch size in settings
- Use CPU mode: `device="cpu"`
- Process shorter documents

### Slow Performance
- Enable GPU acceleration
- Check CUDA availability: `nvidia-smi`
- Use fast mode for quick summaries

## 📦 Dependencies

Key packages used:
- **Transformers**: BART model and tokenizers
- **Torch**: Deep learning framework
- **Streamlit**: Web UI
- **Whisper**: Audio transcription
- **Sentence-Transformers**: Semantic embeddings
- **spaCy**: NLP processing
- **scikit-learn**: ML utilities
- **FFmpeg**: Video processing
- **PyMuPDF/PDFPlumber**: Document extraction

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Multi-language support
- Real-time streaming summarization
- Containerized deployment (Docker)
- Enhanced UI/UX features
- Additional source types




