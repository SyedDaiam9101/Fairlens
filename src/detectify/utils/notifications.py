"""Detectify Notifications - Email and Alerts."""
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from detectify.config import settings
from detectify.utils.logger import logger


class EmailNotifier:
    """Sends email notifications via SMTP (Gmail)."""

    def __init__(self):
        self.last_sent_time = 0
        self.cooldown = settings.notification_cooldown

    def send_alert(
        self,
        subject: str,
        message: str,
        image_path: Optional[str | Path] = None,
    ) -> bool:
        """
        Send an email alert.
        
        Args:
            subject: Email subject.
            message: Email body text.
            image_path: Optional path to an image to attach (the crop).
            
        Returns:
            True if successful, False otherwise.
        """
        if not settings.enable_notifications:
            return False

        if not settings.smtp_user or not settings.smtp_password or not settings.notification_recipient:
            logger.warning("SMTP settings incomplete. Cannot send alert.")
            return False

        # Cooldown check
        current_time = time.time()
        if current_time - self.last_sent_time < self.cooldown:
            logger.debug("Notification suppressed due to cooldown.")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = settings.smtp_user
            msg["To"] = settings.notification_recipient
            msg["Subject"] = subject

            msg.attach(MIMEText(message, "plain"))

            # Attach image crop if provided
            if image_path:
                image_path = Path(image_path)
                if image_path.exists():
                    with open(image_path, "rb") as f:
                        img_data = f.read()
                        image = MIMEImage(img_data, name=image_path.name)
                        msg.attach(image)

            # Connect and send
            with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.send_message(msg)

            self.last_sent_time = current_time
            logger.info(f"Alert email sent to {settings.notification_recipient}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


# Global notifier instance
notifier = EmailNotifier()
