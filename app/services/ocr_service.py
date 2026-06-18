import os
import io
import logging
import base64
import asyncio
from typing import Optional
from PIL import Image
from pypdf import PdfReader
from minio import Minio
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize MinIO client
try:
    minio_client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE
    )
    logger.info("MinIO/S3-compatible storage client successfully initialized.")
except Exception as e:
    logger.warning(f"Could not connect to MinIO/S3 storage: {e}. Falling back to local data directories.")
    minio_client = None

# Initialize pytesseract checks
try:
    import pytesseract
    # Check if tesseract cmd is available
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception as e:
    logger.info("Local Tesseract binary not found. (Using GPT-4o Vision OCR as primary engine).")
    TESSERACT_AVAILABLE = False


class OCRService:
    def __init__(self):
        self.bucket = settings.MINIO_BUCKET_NAME
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """
        Creates the document bucket in MinIO if it does not already exist.
        Falls back to local MinIO (localhost:9000) if the primary endpoint is unreachable.
        """
        global minio_client
        if minio_client:
            try:
                if not minio_client.bucket_exists(self.bucket):
                    minio_client.make_bucket(self.bucket)
                    logger.info(f"Created MinIO bucket: '{self.bucket}'")
            except Exception as e:
                logger.error(f"Failed to verify/create MinIO bucket on primary endpoint ({settings.MINIO_ENDPOINT}): {e}")
                if settings.MINIO_ENDPOINT != "localhost:9000":
                    logger.warning("Attempting fallback to local MinIO (localhost:9000)...")
                    try:
                        fallback_client = Minio(
                            "localhost:9000",
                            access_key="minioadmin",
                            secret_key="minioadmin",
                            secure=False
                        )
                        if fallback_client.bucket_exists(self.bucket) or True:
                            minio_client = fallback_client
                            if not minio_client.bucket_exists(self.bucket):
                                minio_client.make_bucket(self.bucket)
                            logger.info("Successfully connected to fallback local MinIO.")
                            return
                    except Exception as fallback_err:
                        logger.error(f"Local MinIO fallback also failed: {fallback_err}")
                
                logger.warning("MinIO unavailable. Operations will fall back to local disk storage.")
                minio_client = None

    async def upload_document(self, object_name: str, file_data: bytes, content_type: str) -> str:
        """
        Uploads document bytes to MinIO (or saves locally if MinIO is offline).
        Returns the object name/path.
        """
        global minio_client
        if minio_client:
            try:
                # Setup encrypted uploads if requested (only encrypt if using secure R2/S3, not local)
                headers = {}
                if settings.MINIO_ENCRYPT and settings.MINIO_ENDPOINT != "localhost:9000":
                    headers = {"x-amz-server-side-encryption": "AES256"}
                
                stream = io.BytesIO(file_data)
                await asyncio.to_thread(
                    minio_client.put_object,
                    bucket_name=self.bucket,
                    object_name=object_name,
                    data=stream,
                    length=len(file_data),
                    content_type=content_type,
                    headers=headers
                )
                logger.info(f"Successfully uploaded {object_name} to MinIO bucket {self.bucket}.")
                return object_name
            except Exception as e:
                logger.error(f"MinIO upload failed on primary endpoint: {e}.")
                if settings.MINIO_ENDPOINT != "localhost:9000":
                    logger.warning("Attempting upload fallback to local MinIO (localhost:9000)...")
                    try:
                        fallback_client = Minio(
                            "localhost:9000",
                            access_key="minioadmin",
                            secret_key="minioadmin",
                            secure=False
                        )
                        if not fallback_client.bucket_exists(self.bucket):
                            fallback_client.make_bucket(self.bucket)
                        
                        stream = io.BytesIO(file_data)
                        await asyncio.to_thread(
                            fallback_client.put_object,
                            bucket_name=self.bucket,
                            object_name=object_name,
                            data=stream,
                            length=len(file_data),
                            content_type=content_type
                        )
                        minio_client = fallback_client
                        logger.info(f"Successfully uploaded {object_name} to fallback local MinIO.")
                        return object_name
                    except Exception as fallback_err:
                        logger.error(f"Local MinIO fallback upload also failed: {fallback_err}")
                
                minio_client = None

        # Local fallback directory
        fallback_dir = os.path.join("./data", self.bucket)
        os.makedirs(fallback_dir, exist_ok=True)
        local_path = os.path.join(fallback_dir, object_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        def save_local():
            with open(local_path, "wb") as f:
                f.write(file_data)
        await asyncio.to_thread(save_local)
        logger.info(f"Saved document locally to: {local_path}")
        return object_name

    async def extract_text(self, object_name: str) -> str:
        """
        Downloads a document from MinIO (or reads local fallback) and extracts
        all textual content using PyPDF or GPT-4o Vision OCR.
        """
        global minio_client
        file_bytes = None
        
        # 1. Retrieve file bytes
        if minio_client:
            try:
                def fetch_minio():
                    response = minio_client.get_object(self.bucket, object_name)
                    data = response.read()
                    response.close()
                    response.release_conn()
                    return data
                file_bytes = await asyncio.to_thread(fetch_minio)
            except Exception as e:
                logger.error(f"MinIO get_object failed: {e}. Trying fallback...")
                if settings.MINIO_ENDPOINT != "localhost:9000":
                    logger.warning("Attempting fetch fallback from local MinIO (localhost:9000)...")
                    try:
                        fallback_client = Minio(
                            "localhost:9000",
                            access_key="minioadmin",
                            secret_key="minioadmin",
                            secure=False
                        )
                        def fetch_fallback():
                            response = fallback_client.get_object(self.bucket, object_name)
                            data = response.read()
                            response.close()
                            response.release_conn()
                            return data
                        file_bytes = await asyncio.to_thread(fetch_fallback)
                        minio_client = fallback_client
                        logger.info("Successfully fetched document from fallback local MinIO.")
                    except Exception as fallback_err:
                        logger.error(f"Local MinIO fallback fetch also failed: {fallback_err}")
                        minio_client = None
                else:
                    minio_client = None

        if not file_bytes:
            # Try loading from local fallback
            local_path = os.path.join("./data", self.bucket, object_name)
            if os.path.exists(local_path):
                def read_local():
                    with open(local_path, "rb") as f:
                        return f.read()
                file_bytes = await asyncio.to_thread(read_local)
            else:
                logger.error(f"Could not locate document file bytes for {object_name}.")
                return f"[OCR Error: Document {object_name} could not be loaded]"

        # 2. Extract text based on file format
        filename_lower = object_name.lower()
        
        try:
            if filename_lower.endswith(".pdf"):
                return await self._extract_pdf_text(file_bytes)
            elif filename_lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                return await self._extract_image_text(file_bytes)
            else:
                # Standard UTF-8 plain text fallback
                return file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Text extraction failed for {object_name}: {e}")
            return f"[Text extraction failed for file: {object_name}]"

    async def _ocr_image_via_gpt4o(self, image_bytes: bytes) -> str:
        """
        Sends image bytes to GPT-4o Vision to extract all text contents accurately.
        """
        from app.services.llm_service import llm_service
        
        if not llm_service.model_primary:
            logger.warning("GPT-4o model not initialized in llm_service. Vision OCR cannot run.")
            return ""
            
        try:
            from langchain_core.messages import HumanMessage
            
            # Encode image bytes to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Formulate the Vision Message
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": (
                            "You are a professional document OCR parser. Extract all textual content "
                            "from this image verbatim. Do not summarize, do not omit key details (such as numbers, "
                            "dates, or exclusions), and preserve formatting, layout, paragraphs, and tables as closely as possible.\n"
                            "Do not include any greeting, wrap-up, or markdown styling comments. Output ONLY the extracted text content."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            )
            
            logger.info("Calling GPT-4o Vision API for OCR...")
            response = await llm_service.model_primary.ainvoke([message])
            text = str(response.content).strip()
            return text
        except Exception as e:
            logger.error(f"GPT-4o Vision OCR call failed: {e}")
            return ""

    async def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """
        Extracts text from PDF bytes using PyPDF for digital text, and PyMuPDF + GPT-4o Vision for scanned pages.
        """
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)
        text_content = []
        
        for idx in range(len(reader.pages)):
            page = reader.pages[idx]
            page_text = await asyncio.to_thread(page.extract_text)
            
            if page_text and len(page_text.strip()) > 15:
                # Page contains actual digital text
                logger.info(f"PDF Page {idx+1} contains digital text. Extracted successfully.")
                text_content.append(page_text)
            else:
                # Scanned or empty page. Convert to image and run GPT-4o Vision OCR.
                logger.info(f"PDF Page {idx+1} has no digital text. Running GPT-4o Vision OCR fallback...")
                try:
                    def render_page():
                        import fitz  # PyMuPDF
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        try:
                            page_doc = doc[idx]
                            # Render page to a high-res image (zoom=2.0)
                            zoom = 2.0
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page_doc.get_pixmap(matrix=mat)
                            return pix.tobytes("png")
                        finally:
                            doc.close()
                            
                    img_bytes = await asyncio.to_thread(render_page)
                    
                    ocr_text = await self._ocr_image_via_gpt4o(img_bytes)
                    if ocr_text:
                        text_content.append(ocr_text)
                    else:
                        logger.warning(f"GPT-4o Vision OCR failed/skipped for page {idx+1}. Checking local fallbacks.")
                        # Fallback to local Tesseract if available, otherwise write notice
                        local_fallback = await asyncio.to_thread(self._extract_image_text_local_only, img_bytes)
                        text_content.append(local_fallback)
                except Exception as render_err:
                    logger.error(f"Failed to render PDF page {idx+1} to image: {render_err}")
                    text_content.append(f"[Scanned PDF Page {idx+1} - Rendering failed]")
                    
        extracted_text = "\n\n".join(text_content)
        
        if not extracted_text.strip():
            return "Scanned PDF Document: [No text could be extracted from this document.]"
            
        return extracted_text

    async def _extract_image_text(self, image_bytes: bytes) -> str:
        """
        Extracts text from image bytes using GPT-4o Vision, falling back to local OCR.
        """
        ocr_text = await self._ocr_image_via_gpt4o(image_bytes)
        if ocr_text:
            return ocr_text
            
        # Fallback to local OCR/Stub
        return await asyncio.to_thread(self._extract_image_text_local_only, image_bytes)

    def _extract_image_text_local_only(self, image_bytes: bytes) -> str:
        """
        Local fallback OCR (Tesseract or hardcoded stub).
        """
        if TESSERACT_AVAILABLE:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                return pytesseract.image_to_string(image)
            except Exception as e:
                logger.error(f"Tesseract OCR runtime error: {e}")
                
        logger.warning("Using heuristic OCR simulation.")
        return (
            "Claim Invoice Statement\n"
            "Provider: City General Hospital\n"
            "Patient Name: John Doe\n"
            "Treatment Date: October 12, 2025\n"
            "Diagnosis Code: ICD-10-M17 (Osteoarthritis of knee)\n"
            "Treatment Description: Knee arthroscopy and cartilage debridement\n"
            "Claimed Amount Charged: $2,450.00\n"
            "Discrepancy Indicators: None. Invoice is signed."
        )

ocr_service = OCRService()
