import asyncio
import json
import os
import smtplib
from pathlib import Path
from typing import Optional

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _build_message(
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
) -> MIMEMultipart:
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain", "utf-8"))

    if attachment_path:
        attachment_file = Path(attachment_path).expanduser().resolve()
        with attachment_file.open("rb") as handle:
            part = MIMEApplication(handle.read(), Name=attachment_file.name)
        part["Content-Disposition"] = f'attachment; filename="{attachment_file.name}"'
        message.attach(part)

    return message


def _send_message_sync(
    smtp_server: str,
    smtp_port: int,
    sender: str,
    password: str,
    recipient: str,
    message: MIMEMultipart,
) -> None:
    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, [recipient], message.as_string())


async def run(
    recipient: str,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
) -> str:
    try:
        email_user = str(os.getenv("EMAIL_USER") or "").strip()
        email_pass = str(os.getenv("EMAIL_PASS") or "").strip()
        smtp_server = str(os.getenv("SMTP_SERVER") or "smtp.gmail.com").strip() or "smtp.gmail.com"
        smtp_port_raw = str(os.getenv("SMTP_PORT") or "587").strip() or "587"

        if not email_user or not email_pass:
            return json.dumps({
                "status": "error",
                "message": "EMAIL_USER and EMAIL_PASS are not configured. Please update your .env file.",
            })

        try:
            smtp_port = int(smtp_port_raw)
        except ValueError:
            return json.dumps({
                "status": "error",
                "message": "SMTP_PORT must be a valid integer",
            })

        if attachment_path:
            attachment_file = Path(attachment_path).expanduser().resolve()
            if not attachment_file.is_file():
                return json.dumps({
                    "status": "error",
                    "message": f"Attachment not found: {attachment_file}",
                })

        message = _build_message(
            sender=email_user,
            recipient=str(recipient),
            subject=str(subject),
            body=str(body),
            attachment_path=attachment_path,
        )

        await asyncio.to_thread(
            _send_message_sync,
            smtp_server,
            smtp_port,
            email_user,
            email_pass,
            str(recipient),
            message,
        )

        return json.dumps({
            "status": "success",
            "message": f"Email sent to {recipient}",
        })
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": f"Failed to send email: {exc}",
        })


async def send_email(
    recipient: str,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
) -> str:
    return await run(
        recipient=recipient,
        subject=subject,
        body=body,
        attachment_path=attachment_path,
    )
