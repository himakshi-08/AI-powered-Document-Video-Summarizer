import re
from bs4 import BeautifulSoup
import spacy
from transformers import BartTokenizer
import torch
from typing import List, Dict, Union

class EnhancedTextPreprocessor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Load spaCy with full pipeline for better sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clean_text(self, text: str) -> str:
        """Improved text cleaning that preserves meaningful symbols"""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Remove citations but keep other numbers
        text = re.sub(r'\[[0-9]+\]', '', text)  # Only remove [1], [2] etc.
        
        # Remove only problematic symbols (keeps @, #, $ when part of words)
        text = re.sub(r'(?<!\w)[\/\\|<>{}[\]=+`~](?!\w)', '', text)
        
        # Normalize whitespace
        return re.sub(r'\s+', ' ', text).strip()

    def spacy_process(self, text: str, aggressive_lemmatize: bool = False) -> str:
        """Customizable spaCy processing with number preservation"""
        doc = self.nlp(text)
        
        processed = []
        for token in doc:
            # Always keep numbers
            if token.like_num:
                processed.append(token.text)
                continue
                
            # Conditional lemmatization
            lemma = token.lemma_ if aggressive_lemmatize else token.text
            
            # Only remove stopwords that don't carry meaning
            if token.is_stop and token.text.lower() in {'the', 'a', 'an', 'and', 'or'}:
                continue
                
            processed.append(lemma.lower())
            
        return " ".join(processed)

    def chunk_text(self, text: str, max_chunk_size: int = 2000, overlap: int = 100) -> List[str]:
        """
            Optimized chunking for BART (1024 token limit).
            Uses sentence boundaries and keeps chunks under ~2000 chars (~500-800 tokens).
        """
        if len(text) <= max_chunk_size:
            return [text]

        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
    
        chunks = []
        current_chunk = ""
    
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Overlap by carrying over the end of the previous chunk
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk += " " + sentence
    
        if current_chunk:
            chunks.append(current_chunk.strip())
    
        return chunks

    def preprocess(
        self,
        text: str,
        max_length: int = 1024,  # BART's max
        chunk: bool = False,
        max_chunk_size: int = 2000,  # ~500-800 tokens
        chunk_overlap: int = 100,
    ) -> Union[Dict, List[Dict]]:
        """
        Complete preprocessing pipeline with optional chunking
        
        Args:
            text: Input text to process
            max_length: Maximum token length for the model
            chunk: Whether to chunk long documents
            max_chunk_size: Maximum character length for each chunk
            chunk_overlap: Number of overlapping characters between chunks
        
        Returns:
            Single processed dict or list of processed dicts if chunking is enabled
        """
        cleaned = self.clean_text(text)
        
        if chunk and len(cleaned) > max_chunk_size:
            chunks = self.chunk_text(cleaned, max_chunk_size, chunk_overlap)
            results = []
            
            for chunk_text in chunks:
                processed = self.spacy_process(chunk_text, aggressive_lemmatize=False)
                
                inputs = self.tokenizer(
                    processed,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)
                
                results.append({
                    'original': chunk_text,
                    'cleaned': chunk_text,  # Already cleaned before chunking
                    'processed': processed,
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                })
            
            return results
        else:
            processed = self.spacy_process(cleaned, aggressive_lemmatize=False)
            
            inputs = self.tokenizer(
                processed,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            return {
                'original': text,
                'cleaned': cleaned,
                'processed': processed,
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }

# Example Usage
if __name__ == "__main__":
    preprocessor = EnhancedTextPreprocessor()
    
    sample_html = """
    <html>
        <body>
            <p>The quick brown foxes [1] are jumping over the 3 lazy dogs. Email me@example.com!</p>
            <p>Stock prices: $AAPL increased by 5.2% in Q2-2023.</p>
        </body>
    </html>
    """
    
    # Regular processing
    result = preprocessor.preprocess(sample_html)
    print("Single Document Processing:")
    print("Original:\n", result['original'])
    print("\nProcessed:\n", result['processed'])
    print("\nTokenized:", result['input_ids'].shape)
    
    # Create a long document for chunking example
    long_text = " ".join(["This is sentence {}. ".format(i) * 10 for i in range(50)])
    
    # Chunked processing
    chunked_results = preprocessor.preprocess(long_text, chunk=True, max_chunk_size=500, chunk_overlap=50)
    print("\n\nChunked Document Processing:")
    print(f"Total chunks: {len(chunked_results)}")
    print("\nFirst chunk processed text:", chunked_results[0]['processed'][:100] + "...")
    print("First chunk tokenized:", chunked_results[0]['input_ids'].shape)
    print("\nLast chunk processed text:", chunked_results[-1]['processed'][-100:])
    print("Last chunk tokenized:", chunked_results[-1]['input_ids'].shape)