import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.ocr_service import ocr_service
from langchain_core.messages import AIMessage

@pytest.mark.asyncio
async def test_ocr_service_digital_pdf():
    """
    Assert that if a PDF page contains digital text, we extract it directly
    using PyPDF and bypass the GPT-4o Vision API.
    """
    mock_pdf_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is digital text from page 1"
    mock_pdf_reader.pages = [mock_page]
    
    with patch("app.services.ocr_service.PdfReader", return_value=mock_pdf_reader):
        text = await ocr_service._extract_pdf_text(b"mock_pdf_bytes")
        assert "This is digital text from page 1" in text


@pytest.mark.asyncio
async def test_ocr_service_scanned_pdf_fallback():
    """
    Assert that if a PDF page has no digital text, we render the page
    using PyMuPDF and invoke the GPT-4o Vision API.
    """
    mock_pdf_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = ""  # Empty text -> triggers OCR fallback
    mock_pdf_reader.pages = [mock_page]
    
    mock_fitz_doc = MagicMock()
    mock_fitz_page = MagicMock()
    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = b"rendered_png_bytes"
    mock_fitz_page.get_pixmap.return_value = mock_pixmap
    mock_fitz_doc.__getitem__.return_value = mock_fitz_page
    mock_fitz_doc.close = MagicMock()
    
    from app.services.llm_service import llm_service
    mock_response = AIMessage(content="Transcribed text from GPT-4o Vision OCR!")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    
    with patch("app.services.ocr_service.PdfReader", return_value=mock_pdf_reader), \
         patch("fitz.open", return_value=mock_fitz_doc), \
         patch.object(llm_service, "model_primary", mock_model):
        
        text = await ocr_service._extract_pdf_text(b"mock_pdf_bytes")
        assert "Transcribed text from GPT-4o Vision OCR!" in text
        mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_ocr_service_extract_image_text():
    """
    Assert that direct image text extraction executes the GPT-4o Vision model.
    """
    from app.services.llm_service import llm_service
    mock_response = AIMessage(content="Direct Image OCR Output")
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response
    
    with patch.object(llm_service, "model_primary", mock_model):
        text = await ocr_service._extract_image_text(b"mock_image_bytes")
        assert "Direct Image OCR Output" in text
        mock_model.ainvoke.assert_called_once()
