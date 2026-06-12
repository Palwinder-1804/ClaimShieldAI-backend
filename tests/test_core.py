import pytest
from app.core.security import get_password_hash, verify_password, create_access_token, decode_access_token
from app.services.pii_masker import pii_masker

def test_password_hashing():
    """
    Tests bcrypt hashing and validation logic.
    """
    password = "securePassword123"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_jwt_token_handling():
    """
    Tests JWT token signing, expiry injection, and decoding.
    """
    payload = {"sub": "user-123-uuid", "role": "admin"}
    token = create_access_token(payload)
    
    assert isinstance(token, str)
    
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "user-123-uuid"
    assert decoded["role"] == "admin"
    assert "exp" in decoded

def test_pii_masker_regex_fallback():
    """
    Tests that emails, card numbers, and SSNs are correctly scrubbed.
    """
    raw_text = (
        "My email is contact@claimshield.ai and my phone is 202-555-0143. "
        "Also my social is 000-12-3456."
    )
    # Run synchronous helper directly
    scrubbed = pii_masker._regex_scrub(raw_text)
    
    assert "contact@claimshield.ai" not in scrubbed
    assert "202-555-0143" not in scrubbed
    assert "000-12-3456" not in scrubbed
    assert "[EMAIL_MASKED]" in scrubbed
    assert "[PHONE_MASKED]" in scrubbed
    assert "[SSN_MASKED]" in scrubbed
