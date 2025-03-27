import PyPDF2
import io
import re

class DocumentHandler:
    def __init__(self):
        """Initialize document handler for parsing different document types."""
        pass
    
    def parse_document(self, uploaded_file):
        """
        Parse an uploaded document based on its file type.
        Returns the text content of the document.
        """
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # Handle different file types
            if file_extension == 'pdf':
                return self._parse_pdf(uploaded_file)
            elif file_extension == 'md':
                return self._parse_markdown(uploaded_file)
            elif file_extension == 'txt':
                return self._parse_text(uploaded_file)
            else:
                return None
        except Exception as e:
            print(f"Error parsing document: {e}")
            return None

    def get_text_chunks(self, text, max_chunk_size=200, overlap=20):
        """
        Split text into smaller chunks for processing.
        
        Args:
            text: The full document text
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Overlap between chunks for context preservation
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Split text into paragraphs and then chunk appropriately
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, store current chunk and start new one
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep overlap from previous chunk for context
                current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
                
            current_chunk += paragraph + "\n\n"
            
            # If current chunk is getting too big, split it
            while len(current_chunk) > max_chunk_size:
                chunks.append(current_chunk[:max_chunk_size])
                current_chunk = current_chunk[max_chunk_size-overlap:] if overlap > 0 else current_chunk[max_chunk_size:]
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk)
            
        return chunks

    def _parse_pdf(self, uploaded_file):
        """Extract text from PDF files."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"PDF parsing error: {e}")
            return None

    def _parse_markdown(self, uploaded_file):
        """Read markdown files as plain text."""
        try:
            content = uploaded_file.read().decode('utf-8')
            return content
        except Exception as e:
            print(f"Markdown parsing error: {e}")
            return None

    def _parse_text(self, uploaded_file):
        """Read text files."""
        try:
            content = uploaded_file.read().decode('utf-8')
            return content
        except Exception as e:
            print(f"Text file parsing error: {e}")
            return None
