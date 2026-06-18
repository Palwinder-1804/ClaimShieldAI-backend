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
        verification_link = f"{settings.BACKEND_URL}/auth/verify?token={token}"
        
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

    @staticmethod
    async def send_claim_failure_email(email_to: str, claim_id: str, error_message: str) -> None:
        """
        Sends an email notification to the user when their claim processing fails permanently.
        """
        html_content = f"""
        <html>
            <body>
                <h2>Claim Processing Update</h2>
                <p>Dear User,</p>
                <p>We encountered an unexpected error while processing your claim.</p>
                <p><strong>Claim ID:</strong> {claim_id}</p>
                <p>Our engineering team has been notified and is looking into the issue. You do not need to resubmit at this time.</p>
                <p>Best regards,<br/>ClaimShield AI Team</p>
            </body>
        </html>
        """
        
        message = MessageSchema(
            subject=f"Claim Processing Update: Claim {claim_id}",
            recipients=[email_to],
            body=html_content,
            subtype=MessageType.html
        )
        
        if not settings.MAIL_USERNAME or "smtp-username" in settings.MAIL_USERNAME:
            logger.info("==========================================================")
            logger.info(f"DEVELOPMENT MODE: Claim failure email generated for user {email_to}")
            logger.info(f"Claim ID: {claim_id}")
            logger.info("==========================================================")
            return

        try:
            fm = FastMail(conf)
            await fm.send_message(message)
            logger.info(f"Claim failure notification sent to user {email_to}")
        except Exception as e:
            logger.error(f"Failed to send claim failure email to user {email_to}: {e}")

    @staticmethod
    async def send_admin_failure_alert(claim_id: str, error_message: str) -> None:
        """
        Sends a critical failure alert to the system administrator when a claim fails permanently.
        """
        admin_email = settings.MAIL_FROM or settings.MAIL_USERNAME or "admin@claimshield.ai"
        
        html_content = f"""
        <html>
            <body>
                <h2>CRITICAL: Claim Processing Pipeline Permanent Failure</h2>
                <p>A claim has failed processing permanently after exceeding all retry attempts.</p>
                <p><strong>Claim ID:</strong> {claim_id}</p>
                <p><strong>Error Details:</strong> {error_message}</p>
                <p>Please check the worker logs, LangSmith traces, or database audit logs for troubleshooting.</p>
                <p>Best regards,<br/>ClaimShield AI Monitoring</p>
            </body>
        </html>
        """
        
        message = MessageSchema(
            subject=f"CRITICAL SYSTEM ALERT: Claim {claim_id} Failed Permanently",
            recipients=[admin_email],
            body=html_content,
            subtype=MessageType.html
        )
        
        if not settings.MAIL_USERNAME or "smtp-username" in settings.MAIL_USERNAME:
            logger.info("==========================================================")
            logger.info(f"DEVELOPMENT MODE: Admin critical alert email generated for {admin_email}")
            logger.info(f"Claim ID: {claim_id}")
            logger.info(f"Error: {error_message}")
            logger.info("==========================================================")
            return

        try:
            fm = FastMail(conf)
            await fm.send_message(message)
            logger.info(f"Admin failure alert email successfully sent to {admin_email}")
        except Exception as e:
            logger.error(f"Failed to send admin failure alert email to {admin_email}: {e}")

    @staticmethod
    async def send_feedback_email(claim_id: str, rating: int, agreed: bool, comment: str, submitter_email: str) -> None:
        """
        Sends claim validation feedback details to the ClaimShield AI Admin.
        """
        admin_email = settings.MAIL_FROM or settings.MAIL_USERNAME or "palwinder1874singh@gmail.com"
        agreement_text = "Agreed with system decision" if agreed else "Disagreed with system decision"
        stars = "★" * rating + "☆" * (5 - rating)
        
        html_content = f"""
        <html>
            <body>
                <h2>Claim Validation Feedback Received</h2>
                <p>Hello Admin,</p>
                <p>New validation feedback has been submitted for a claim decision audit.</p>
                <p><strong>Claim ID:</strong> {claim_id}</p>
                <p><strong>Submitted By (User):</strong> {submitter_email}</p>
                <p><strong>Agreement status:</strong> {agreement_text}</p>
                <p><strong>System Rating:</strong> {stars} ({rating}/5)</p>
                {"<p><strong>Comments:</strong> " + comment + "</p>" if comment else ""}
                <p>Best regards,<br/>ClaimShield AI Monitoring</p>
            </body>
        </html>
        """
        
        message = MessageSchema(
            subject=f"Claim Feedback Submission Alert: Claim {claim_id}",
            recipients=[admin_email],
            body=html_content,
            subtype=MessageType.html
        )
        
        if not settings.MAIL_USERNAME or "smtp-username" in settings.MAIL_USERNAME:
            logger.info("==========================================================")
            logger.info(f"DEVELOPMENT MODE: Validation feedback email generated for Admin: {admin_email}")
            logger.info(f"Claim ID: {claim_id} | Submitter: {submitter_email} | Rating: {rating} | Agreed: {agreed}")
            logger.info("==========================================================")
            return

        try:
            fm = FastMail(conf)
            await fm.send_message(message)
            logger.info(f"Validation feedback email successfully sent to Admin: {admin_email}")
        except Exception as e:
            logger.error(f"Failed to send validation feedback email to Admin: {admin_email}: {e}")

email_service = EmailService()
