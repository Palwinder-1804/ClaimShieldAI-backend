import logging
from typing import List

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._init_splitter()

    def _init_splitter(self) -> None:
        """
        Initializes the splitter. Prefers Tiktoken token-based splitting if tiktoken is available.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # Use tiktoken encoder for precise 512-token bounds if possible
            self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info("Token-based RecursiveCharacterTextSplitter initialized (tiktoken).")
        except ImportError:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size * 4,  # Rough character equivalent
                    chunk_overlap=self.chunk_overlap * 4
                )
                logger.info("Character-based RecursiveCharacterTextSplitter initialized (fallback).")
            except ImportError as e:
                logger.error(f"Failed to load langchain text splitters: {e}")
                self.splitter = None

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of 512 tokens (or characters as fallback) with 50 token/char overlap.
        """
        if not text:
            return []
            
        if self.splitter:
            try:
                return self.splitter.split_text(text)
            except Exception as e:
                logger.error(f"Text splitting failed: {e}. Falling back to paragraph split.")
                
        # Basic manual fallback split by paragraphs if imports failed
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

document_chunker = DocumentChunker()
