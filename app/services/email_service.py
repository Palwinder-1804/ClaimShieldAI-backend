import logging
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configure fastapi-mail SMTP connection config
conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME or "",
    MAIL_PASSWORD=settings.MAIL_PASSWORD or "",
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_FROM_NAME=settings.PROJECT_NAME,
    MAIL_STARTTLS=settings.MAIL_STARTTLS,
    MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
    USE_CREDENTIALS=True if settings.MAIL_USERNAME else False,
    VALIDATE_CERTS=False  # Disabled for simple integration or local dev smtp testing (like MailHog)
)

class EmailService:
    @staticmethod
    async def send_verification_email(email_to: str, token: str) -> None:
        """
        Sends an email verification link containing a secure token to the user.
        If SMTP credentials are not configured, prints the email details to logs for local development.
        """
        verification_link = f"http://localhost:8000/auth/verify?token={token}"
        
        html_content = f"""
        <html>
            <body>
                <h2>Welcome to ClaimShield AI</h2>
                <p>Please verify your email address by clicking the link below:</p>
                <p><a href="{verification_link}">{verification_link}</a></p>
                <p>This link will expire in 24 hours.</p>
                <p>Best regards,<br/>ClaimShield AI Team</p>
            </body>
        </html>
        """
        
        message = MessageSchema(
            subject="Verify Your ClaimShield AI Account",
            recipients=[email_to],
            body=html_content,
            subtype=MessageType.html
        )
        
        # If mail credentials are dummy, log to console instead of throwing errors
        if not settings.MAIL_USERNAME or "smtp-username" in settings.MAIL_USERNAME:
            logger.info("==========================================================")
            logger.info(f"DEVELOPMENT MODE: Verification email generated for {email_to}")
            logger.info(f"Link: {verification_link}")
            logger.info("==========================================================")
            return

        try:
            fm = FastMail(conf)
            await fm.send_message(message)
            logger.info(f"Verification email successfully sent to {email_to}")
        except Exception as e:
            logger.error(f"Failed to send verification email to {email_to}: {e}")
            # Fallback output
            logger.info(f"FALLBACK LINK PRINT: {verification_link}")

email_service = EmailService()
