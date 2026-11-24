"""Email delivery service handling automatic task reminders via SMTP or safe mock simulation."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def build_reminder_email_content(student_name: str, task_title: str, due_date_str: str) -> Dict[str, str]:
    """
    Constructs the standard pending task reminder email text and subject
    per the exact system requirements.
    """
    subject = "Pending Task Reminder"
    body = (
        f"Subject: Pending Task Reminder\n\n"
        f"Hello {student_name},\n\n"
        f'Your assigned task "{task_title}" is still pending.\n\n'
        f"Due Date: {due_date_str}\n\n"
        f"Please log in to your student dashboard and complete the task.\n\n"
        f"Regards,\n"
        f"Teacher / AI Viva Coach\n"
    )
    return {"subject": subject, "body": body}


def send_task_reminder_email(
    student_name: str,
    student_email: str,
    task_title: str,
    due_date_str: str
) -> Dict[str, Any]:
    """
    Sends an automatic pending task reminder to the student's verified email address.
    
    If SMTP credentials are not configured in environment variables,
    it records a safe 'mock_simulated' delivery status without claiming real delivery.
    """
    settings = get_settings()
    content = build_reminder_email_content(student_name, task_title, due_date_str)
    subject = content["subject"]
    body = content["body"]

    # 1. Check if real SMTP credentials are provided
    has_smtp_credentials = bool(
        settings.SMTP_HOST and
        settings.SMTP_USER and
        settings.SMTP_PASSWORD and
        settings.SMTP_HOST != "smtp.example.com"
    )

    if not has_smtp_credentials:
        logger.info(
            f"[MOCK EMAIL REMINDER] To: {student_email} | Subject: '{subject}' | Task: '{task_title}' "
            f"(Simulated in development mode as SMTP credentials are not configured in backend env)."
        )
        return {
            "status": "mock_simulated",
            "delivered": False,
            "subject": subject,
            "recipient": student_email,
            "message": "Reminder recorded in safe mock simulation mode (SMTP unconfigured in environment).",
        }

    # 2. Attempt real SMTP transmission
    try:
        msg = MIMEMultipart()
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = student_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            if settings.SMTP_PORT in (587, 25):
                server.starttls()
                server.ehlo()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"[REAL EMAIL DELIVERED] Reminder sent to {student_email} for task '{task_title}'.")
        return {
            "status": "sent",
            "delivered": True,
            "subject": subject,
            "recipient": student_email,
            "message": "Reminder email delivered successfully via SMTP.",
        }
    except Exception as exc:
        logger.error(f"[EMAIL DELIVERY FAILED] Could not send reminder to {student_email}: {exc}")
        return {
            "status": "failed",
            "delivered": False,
            "subject": subject,
            "recipient": student_email,
            "error": str(exc),
        }
