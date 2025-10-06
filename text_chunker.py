import re
import nltk
from typing import List, Dict, Optional, Tuple
import tiktoken
from nltk.tokenize import sent_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class TextChunker:
    """
    Advanced text chunking service with sentence-level splitting and overlap support.
    Preserves semantic meaning while maintaining optimal chunk sizes for LLM processing.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 encoding_name: str = "cl100k_base"):
        """
        Initialize TextChunker with configuration.
        
        Args:
            chunk_size (int): Maximum tokens per chunk
            chunk_overlap (int): Number of overlapping tokens between chunks
            encoding_name (str): Tokenizer encoding name for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text (str): Input text
            
        Returns:
            int: Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def split_by_sentences(self, text: str, language: str = 'english') -> List[str]:
        """
        Split text into sentences using NLTK.
        
        Args:
            text (str): Input text
            language (str): Language for sentence tokenization
            
        Returns:
            List[str]: List of sentences
        """
        try:
            # Handle Korean text
            if language == 'korean' or self._detect_korean(text):
                # Korean sentence splitting using regex patterns
                sentences = self._split_korean_sentences(text)
            else:
                # Use NLTK for other languages
                sentences = sent_tokenize(text, language=language)
            
            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                cleaned = sentence.strip()
                if cleaned and len(cleaned) > 10:  # Filter very short sentences
                    cleaned_sentences.append(cleaned)
            
            return cleaned_sentences
        except Exception as e:
            self.logger.error(f"Error in sentence splitting: {e}")
            # Fallback to simple splitting
            return self._fallback_sentence_split(text)
    
    def _detect_korean(self, text: str) -> bool:
        """
        Detect if text contains Korean characters.
        
        Args:
            text (str): Input text
            
        Returns:
            bool: True if Korean detected
        """
        korean_pattern = re.compile(r'[가-힣]')
        return bool(korean_pattern.search(text))
    
    def _split_korean_sentences(self, text: str) -> List[str]:
        """
        Split Korean text into sentences using regex patterns.
        
        Args:
            text (str): Korean text
            
        Returns:
            List[str]: List of sentences
        """
        # Korean sentence ending patterns
        sentence_endings = r'[.!?。！？]+'
        
        # Split by sentence endings but keep the punctuation
        sentences = re.split(f'({sentence_endings})', text)
        
        # Reconstruct sentences with their punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                punctuation = sentences[i + 1].strip()
                sentence += punctuation
            
            if sentence and len(sentence) > 5:
                result.append(sentence)
        
        return result
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """
        Fallback sentence splitting using simple regex.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of sentences
        """
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks_with_sentences(self, text: str, language: str = 'english') -> List[Dict]:
        """
        Create chunks by combining sentences while respecting token limits.
        
        Args:
            text (str): Input text to chunk
            language (str): Language for sentence tokenization
            
        Returns:
            List[Dict]: List of chunk dictionaries with metadata
        """
        sentences = self.split_by_sentences(text, language)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        sentence_indices = []
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, current_tokens, sentence_indices
                    ))
                    current_chunk = ""
                    current_tokens = 0
                    sentence_indices = []
                
                # Split long sentence into sub-chunks
                sub_chunks = self._split_long_sentence(sentence)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk_dict(
                        sub_chunk, self.count_tokens(sub_chunk), [i]
                    ))
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk_dict(
                    current_chunk, current_tokens, sentence_indices
                ))
                
                # Start new chunk with overlap
                overlap_text, overlap_indices = self._create_overlap(
                    chunks[-1], sentences, self.chunk_overlap
                )
                current_chunk = overlap_text
                current_tokens = self.count_tokens(current_chunk)
                sentence_indices = overlap_indices
            
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_tokens = self.count_tokens(current_chunk)
            sentence_indices.append(i)
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk, current_tokens, sentence_indices
            ))
        
        return chunks
    
    def _create_chunk_dict(self, text: str, token_count: int, sentence_indices: List[int]) -> Dict:
        """
        Create chunk dictionary with metadata.
        
        Args:
            text (str): Chunk text
            token_count (int): Number of tokens
            sentence_indices (List[int]): Indices of sentences in chunk
            
        Returns:
            Dict: Chunk dictionary
        """
        return {
            'text': text.strip(),
            'token_count': token_count,
            'character_count': len(text),
            'sentence_count': len(sentence_indices),
            'sentence_indices': sentence_indices,
            'word_count': len(text.split())
        }
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a long sentence into smaller parts.
        
        Args:
            sentence (str): Long sentence to split
            
        Returns:
            List[str]: List of sentence parts
        """
        # Split by common delimiters while preserving meaning
        delimiters = [', ', '; ', ' - ', ' and ', ' but ', ' or ', ' 그리고 ', ' 하지만 ', ' 또는 ']
        
        parts = [sentence]
        
        for delimiter in delimiters:
            new_parts = []
            for part in parts:
                if self.count_tokens(part) > self.chunk_size:
                    split_parts = part.split(delimiter)
                    for i, split_part in enumerate(split_parts):
                        if i > 0:
                            split_part = delimiter.strip() + " " + split_part
                        new_parts.append(split_part)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # If still too long, split by character count
        final_parts = []
        for part in parts:
            if self.count_tokens(part) > self.chunk_size:
                # Character-based splitting as last resort
                char_limit = self.chunk_size * 3  # Rough estimate
                for i in range(0, len(part), char_limit):
                    final_parts.append(part[i:i + char_limit])
            else:
                final_parts.append(part)
        
        return [part.strip() for part in final_parts if part.strip()]
    
    def _create_overlap(self, previous_chunk: Dict, all_sentences: List[str], overlap_tokens: int) -> Tuple[str, List[int]]:
        """
        Create overlap text from previous chunk.
        
        Args:
            previous_chunk (Dict): Previous chunk dictionary
            all_sentences (List[str]): All sentences
            overlap_tokens (int): Target overlap tokens
            
        Returns:
            Tuple[str, List[int]]: Overlap text and sentence indices
        """
        if not previous_chunk['sentence_indices']:
            return "", []
        
        # Start from the last sentences of previous chunk
        overlap_text = ""
        overlap_indices = []
        current_tokens = 0
        
        # Work backwards from the end of previous chunk
        for i in reversed(previous_chunk['sentence_indices']):
            if i < len(all_sentences):
                sentence = all_sentences[i]
                sentence_tokens = self.count_tokens(sentence)
                
                if current_tokens + sentence_tokens <= overlap_tokens:
                    overlap_text = sentence + " " + overlap_text
                    overlap_indices.insert(0, i)
                    current_tokens += sentence_tokens
                else:
                    break
        
        return overlap_text.strip(), overlap_indices
    
    def create_semantic_chunks(self, text: str, language: str = 'english') -> List[Dict]:
        """
        Create semantically coherent chunks with improved overlap strategy.
        
        Args:
            text (str): Input text
            language (str): Language for processing
            
        Returns:
            List[Dict]: List of semantic chunks
        """
        # First, create sentence-based chunks
        base_chunks = self.create_chunks_with_sentences(text, language)
        
        # Enhance chunks with semantic information
        enhanced_chunks = []
        for i, chunk in enumerate(base_chunks):
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update({
                'chunk_id': i,
                'has_overlap': i > 0,
                'overlap_with_previous': i > 0,
                'overlap_with_next': i < len(base_chunks) - 1
            })
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about the chunking results.
        
        Args:
            chunks (List[Dict]): List of chunks
            
        Returns:
            Dict: Statistics dictionary
        """
        if not chunks:
            return {}
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        char_counts = [chunk['character_count'] for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'total_characters': sum(char_counts),
            'avg_tokens_per_chunk': sum(token_counts) / len(chunks),
            'avg_chars_per_chunk': sum(char_counts) / len(chunks),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'chunks_with_overlap': sum(1 for chunk in chunks if chunk.get('has_overlap', False))
        }


def main():
    """Example usage of TextChunker"""
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    # Sample text (mix of English and Korean)
    sample_text = """
    인공지능과 투자의 만남은 현대 금융 시장에서 가장 흥미로운 발전 중 하나입니다. 
    머신러닝 알고리즘은 방대한 양의 시장 데이터를 분석하여 패턴을 찾아냅니다.
    
    Artificial intelligence has revolutionized investment strategies. Machine learning algorithms 
    can process vast amounts of market data to identify patterns that human analysts might miss.
    
    특히 자연어 처리 기술은 뉴스, 소셜 미디어, 재무 보고서 등의 텍스트 데이터를 분석하여 
    시장 감정을 파악하는 데 매우 유용합니다.
    
    The integration of AI in investment decision-making has led to the development of robo-advisors,
    which provide automated portfolio management services to retail investors.
    """
    
    print("=== Text Chunker Test ===")
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Original token count: {chunker.count_tokens(sample_text)}")
    
    # Test sentence splitting
    sentences = chunker.split_by_sentences(sample_text, 'korean')
    print(f"\nSentences extracted: {len(sentences)}")
    for i, sentence in enumerate(sentences[:3]):
        print(f"{i+1}: {sentence}")
    
    # Test chunking
    chunks = chunker.create_semantic_chunks(sample_text, 'korean')
    print(f"\nChunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Characters: {chunk['character_count']}")
        print(f"  Sentences: {chunk['sentence_count']}")
        print(f"  Text preview: {chunk['text'][:100]}...")
    
    # Get statistics
    stats = chunker.get_chunk_statistics(chunks)
    print(f"\n=== Chunking Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
