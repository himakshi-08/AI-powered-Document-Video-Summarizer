# 🎯 AI-Powered Document & Video Summarizer

A comprehensive, intelligent summarization suite that leverages cutting-edge AI models to extract, condense, and summarize content from multiple sources including articles, videos, PDFs, and documents.

---

## ⚡ Quick Start (TL;DR)

```bash
# 1. Prerequisites: Install FFmpeg, Python 3.8+, and Git

# 2. Clone & Setup (5 minutes)
git clone https://github.com/himakshi-08/AI-powered-Document-Video-Summarizer.git
cd AI-powered-Document-Video-Summarizer
conda create -n bart-env python=3.10 -y
conda activate bart-env

# 3. Install Dependencies (10-20 minutes)
conda install -c conda-forge -y transformers datasets sentencepiece accelerate spacy beautifulsoup4 nltk scikit-learn tqdm pandas numpy
python -m spacy download en_core_web_sm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit>=1.25.0 yt-dlp ffmpeg-python openai-whisper rouge-score pdfplumber pymupdf python-docx newspaper3k selenium tokenizers evaluate sentence-transformers

# 4. Run (1 minute)
streamlit run main_app.py
```

**Result:** App opens at `http://localhost:8501` ✅

---

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

## 🏗️ Project Structure

```
AI-powered-Document-Video-Summarizer/
│
├── 📄 Core Application Files
├── ├── main_app.py                      # Main Streamlit web application
├── ├── abstractive_model.py             # Abstractive summarization engine (BART)
├── ├── extractive_summarizer.py         # Extractive summarization logic
├── ├── core_scraper.py                  # Web article scraping utilities
├── ├── core_transcriber.py              # Video transcription & audio extraction
├── ├── text_extraction.py               # PDF/Document text extraction
├── ├── preprocess.py                    # Text preprocessing & normalization
├── ├── rouge_evaluation.py              # ROUGE metrics for evaluation
├── ├── ui_styles.py                     # Custom Streamlit CSS styling
│
├── 📦 Models & Weights
├── ├── bart-finetuned-mediasum/         # Fine-tuned BART model (MediaSum)
│   ├── ├── config.json
│   ├── ├── generation_config.json
│   ├── ├── model.safetensors
│   ├── ├── tokenizer_config.json
│   ├── ├── vocab.json
│   ├── ├── merges.txt
│   ├── ├── special_tokens_map.json
│   ├── └── checkpoint-4500/             # Training checkpoint (4500 steps)
│   │
│   └── partial_model/                   # Alternative pre-trained model variant
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       └── special_tokens_map.json
│
├── 📊 Training & Logs
├── ├── logs/                            # TensorBoard event files
│   ├── events.out.tfevents.*            # Training metrics and loss curves
│
├── 🛠️ Training & Utilities
├── └── training and additional files/
│   ├── abstractive_model_training.py    # BART model fine-tuning script
│   ├── transcript_abstractive_model_training.py  # Video transcript training
│   ├── extractive_model_training.ipynb  # Jupyter notebook for extractive training
│   ├── api_service.py                   # REST API service (optional)
│   ├── scrapper.py                      # Advanced scraping utilities
│   ├── transcription.py                 # Audio transcription helpers
│   └── download_resource.py             # Model/data download utilities
│
├── 📋 Configuration
├── ├── requirements.txt                 # Python dependencies
├── ├── README.md                        # This documentation
│
└── 📁 Cache
    └── __pycache__/                     # Python cache directory
```

## 🚀 Getting Started

### Prerequisites

**Required:**
- ✅ Python 3.8 or higher
- ✅ Miniconda or Anaconda
- ✅ FFmpeg (system-level)
- ✅ Git (for cloning repository)

**Optional:**
- 🔧 CUDA 11.8+ (for GPU acceleration - significantly faster)
- 🔧 Nvidia GPU (for GPU support)

### System-Level Installation (FFmpeg)

**FFmpeg is required for video processing. Install based on your OS:**

```powershell
# Windows (using Chocolatey)
choco install ffmpeg

# Or if you don't have chocolatey, download from: https://ffmpeg.org/download.html
```

```bash
# macOS (using Homebrew)
brew install ffmpeg
```

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg

# Linux (Fedora/RHEL)
sudo yum install ffmpeg
```

### Step-by-Step Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/himakshi-08/AI-powered-Document-Video-Summarizer.git
cd AI-powered-Document-Video-Summarizer
```

#### Step 2: Create Conda Environment
```bash
conda create -n bart-env python=3.10
conda activate bart-env
```

#### Step 3: Install Core Dependencies (NLP & Deep Learning)
```bash
# Install transformers, datasets, and acceleration libraries
conda install -c conda-forge -y transformers datasets sentencepiece accelerate

# Install PyTorch (choose based on your setup)
# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (if no GPU):
pip install torch torchvision torchaudio

# Install TensorBoard for training monitoring
pip install tensorboard
```

#### Step 4: Install NLP & Preprocessing Tools
```bash
conda install -c conda-forge -y spacy beautifulsoup4 nltk scikit-learn tqdm pandas numpy

# Download spaCy English model
python -m spacy download en_core_web_sm
```

#### Step 5: Install Web & Application Framework
```bash
pip install streamlit>=1.25.0 requests yt-dlp ffmpeg-python
```

#### Step 6: Install Audio & Document Processing
```bash
# Audio transcription
pip install openai-whisper

# Document extraction
pip install pdfplumber pymupdf python-docx

# Web scraping
pip install newspaper3k selenium lxml_html_clean

# ML evaluation & utilities
pip install rouge-score tokenizers evaluate
```

#### Step 7: Install Semantic & Embedding Models
```bash
pip install sentence-transformers
```

#### Step 8: Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

### Quick Install (All-in-One Command)

If you prefer to install everything at once:

```bash
conda create -n bart-env python=3.10 -y && \
conda activate bart-env && \
conda install -c conda-forge -y transformers datasets sentencepiece accelerate spacy beautifulsoup4 nltk scikit-learn tqdm pandas numpy && \
python -m spacy download en_core_web_sm && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install tensorboard streamlit>=1.25.0 requests yt-dlp ffmpeg-python openai-whisper rouge-score && \
pip install pdfplumber pymupdf python-docx newspaper3k selenium lxml_html_clean tokenizers evaluate sentence-transformers
```

### Launch the Application

```bash
# Make sure you're in the project directory and conda environment is activated
conda activate bart-env

# Run the Streamlit application
streamlit run main_app.py
```

The application will automatically open at `http://localhost:8501` in your default browser.

### Verify Everything Works

To test if the installation is complete:
```bash
# Download and cache Whisper model (first-time only, ~140MB)
python -c "import whisper; whisper.load_model('base')"

# Load BART model
python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large')"
```

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

### Installation Issues

#### Python Version Error
```bash
# Check your Python version
python --version

# If version < 3.8, update Python or use a new conda environment
conda create -n bart-env python=3.10 -y
```

#### FFmpeg Not Found
```bash
# Windows - Verify FFmpeg is in PATH
ffmpeg -version

# If not found, reinstall or add to system PATH
# Download: https://ffmpeg.org/download.html
```

#### Conda Command Not Found
```bash
# Install Miniconda from: https://docs.conda.io/projects/miniconda/en/latest/
# Or add Anaconda/Miniconda to your system PATH
```

#### CUDA Not Available (for GPU users)
```bash
# Check if GPU is detected
nvidia-smi

# If not working, download appropriate CUDA from:
# https://developer.nvidia.com/cuda-downloads

# Reinstall PyTorch for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Runtime Issues

#### Whisper Model Download Fails
```bash
# Force model download to ~/.cache/whisper
python -c "import whisper; whisper.load_model('base')"

# Or manually download from Hugging Face Hub
```

#### Out of Memory (CUDA)
```bash
# Use CPU mode instead
# Edit main_app.py and set device="cpu"

# Or reduce batch size in settings
```

#### Streamlit Port Already in Use
```bash
# Run on different port
streamlit run main_app.py --server.port 8502
```

#### Slow Summarization on CPU
- Install GPU support (CUDA 11.8+)
- Use shorter documents
- Enable fast mode in UI
- Reduce max summary length

### Verification Checklist

Run these commands to verify everything works:

```bash
# 1. Python & Conda
python --version
conda --version

# 2. FFmpeg
ffmpeg -version

# 3. PyTorch & GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 4. Key Libraries
python -c "import streamlit; import transformers; import whisper; import spacy; print('✅ All core libraries loaded successfully')"

# 5. Models
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('facebook/bart-large'); print('✅ BART model accessible')"

# 6. Run App
streamlit run main_app.py
```

### Common Error Messages & Solutions

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run: `pip install streamlit>=1.25.0` |
| `ffmpeg: command not found` | Install FFmpeg via system package manager |
| `No module named 'torch'` | Run: `pip install torch` (or see PyTorch installation above) |
| `CUDA out of memory` | Reduce batch size or use CPU mode |
| `Cannot find spacy model` | Run: `python -m spacy download en_core_web_sm` |
| `Whisper download fails` | Check internet connection or use cached version |

## 📋 System Requirements Summary

### Minimum (CPU-Only)
- Python 3.8+
- 4GB RAM
- 2GB storage (models)
- ~30 minutes installation time

### Recommended (GPU-Accelerated)
- Python 3.10
- Nvidia GPU with 4GB+ VRAM
- 16GB+ system RAM
- 8GB storage (models)
- CUDA 11.8 compatible GPU
- ~15 minutes installation time (much faster inference)

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



