import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Try to initialize Microsoft Presidio Analyzer and Anonymizer engines
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
    logger.info("Microsoft Presidio PII Masker successfully initialized.")
except Exception as e:
    logger.warning(
        f"Could not load Microsoft Presidio: {e}. "
        f"This is common if the spacy model 'en_core_web_lg' or 'en_core_web_sm' is not installed.\n"
        f"Falling back to regex-based PII scrubber for local execution."
    )
    PRESIDIO_AVAILABLE = False

class PIIMasker:
    async def mask_text(self, text: str) -> str:
        """
        Scrubs personally identifiable information (PII) from the input text.
        Uses Presidio Analyzer/Anonymizer when available; falls back to regex patterns.
        """
        if not text:
            return ""

        if PRESIDIO_AVAILABLE:
            try:
                # Analyze text for PII entities
                results = analyzer.analyze(text=text, language="en")
                # Anonymize matches
                anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized_result.text
            except Exception as e:
                logger.error(f"Presidio masking failed: {e}. Falling back to regex scrubber.")
                
        return self._regex_scrub(text)

    def _regex_scrub(self, text: str) -> str:
        """
        Backup regex-based scrubber to remove Emails, Phone Numbers, and Social Security Numbers.
        """
        scrubbed = text
        
        # Email addresses
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        scrubbed = re.sub(email_pattern, "[EMAIL_MASKED]", scrubbed)
        
        # Phone numbers (e.g. +1-123-456-7890, (123) 456-7890, 123-456-7890)
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        scrubbed = re.sub(phone_pattern, "[PHONE_MASKED]", scrubbed)
        
        # SSN (Social Security Numbers: 3 digits - 2 digits - 4 digits)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        scrubbed = re.sub(ssn_pattern, "[SSN_MASKED]", scrubbed)
        
        # credit cards
        card_pattern = r'\b(?:\d[ -]*?){13,16}\b'
        scrubbed = re.sub(card_pattern, "[CREDIT_CARD_MASKED]", scrubbed)
        
        return scrubbed

pii_masker = PIIMasker()
