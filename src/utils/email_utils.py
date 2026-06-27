"""
Email alert utility — used by deploy flows to notify on failure/success.

Reads SMTP config from env vars (same as Grafana SMTP config in docker-compose).
Silently skips if SMTP_USER or ALERT_EMAIL is not configured.
"""
import logging
import os
import smtplib
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

_SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com:587")
_SMTP_USER = os.getenv("SMTP_USER", "")
_SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
_SMTP_FROM = os.getenv("SMTP_FROM", "no-reply@cac-mlops.local")
_ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")


def send_alert(subject: str, body: str) -> None:
    """Send an email alert. Silently skips if SMTP is not configured."""
    if not _SMTP_USER or not _ALERT_EMAIL:
        logger.debug("SMTP non configuré — email ignoré: %s", subject)
        return

    try:
        host, _, port_str = _SMTP_HOST.partition(":")
        port = int(port_str) if port_str else 587

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"[cac-mlops] {subject}"
        msg["From"] = _SMTP_FROM
        msg["To"] = _ALERT_EMAIL

        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(_SMTP_USER, _SMTP_PASSWORD)
            smtp.send_message(msg)

        logger.info("Email envoyé → %s : %s", _ALERT_EMAIL, subject)
    except Exception as exc:
        logger.warning("Échec envoi email (%s): %s", subject, exc)
