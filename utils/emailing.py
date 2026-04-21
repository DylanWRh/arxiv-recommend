from __future__ import annotations

import datetime as dt
import smtplib
from email.message import EmailMessage

from utils.runtime import debug_log, str_env


def build_email(
    recipient: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    text_report: str,
    html_report: str,
) -> EmailMessage:
    msg = EmailMessage()
    sender = str_env("EMAIL_FROM", str_env("SMTP_USER", "noreply@example.com"))
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = (
        f"arXiv recommendations ({start_utc.strftime('%Y-%m-%d')} to {end_utc.strftime('%Y-%m-%d')})"
    )
    msg.set_content(text_report)
    msg.add_alternative(html_report, subtype="html")
    return msg


def send_email(message: EmailMessage, dbg: bool = False) -> None:
    host = str_env("SMTP_HOST")
    if not host:
        raise ValueError("SMTP_HOST is not set.")
    port = int(str_env("SMTP_PORT", "587"))
    use_tls = str_env("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}
    username = str_env("SMTP_USER", "")
    password = str_env("SMTP_PASS", "")
    use_implicit_ssl = use_tls and port in {465, 994}
    smtp_cls = smtplib.SMTP_SSL if use_implicit_ssl else smtplib.SMTP

    debug_log(
        dbg,
        (
            f"Connecting to SMTP host={host} port={port} "
            f"tls={use_tls} implicit_ssl={use_implicit_ssl}."
        ),
    )
    with smtp_cls(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls and not use_implicit_ssl:
            debug_log(dbg, "Starting TLS negotiation.")
            smtp.starttls()
            smtp.ehlo()
        if username:
            debug_log(dbg, "Authenticating with SMTP server.")
            smtp.login(username, password)
        debug_log(dbg, "Sending email message.")
        smtp.send_message(message)
    debug_log(dbg, "SMTP send completed.")
