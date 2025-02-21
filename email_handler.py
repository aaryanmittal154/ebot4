import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from config import CONFIG

@dataclass
class EmailData:
    subject: str
    content: str
    thread_id: Optional[str]
    references: List[str]
    sender: str
    message_id: str

class EmailHandler:
    def __init__(self):
        self.imap = imaplib.IMAP4_SSL(CONFIG["email"].IMAP_SERVER)
        self.smtp = smtplib.SMTP(CONFIG["email"].SMTP_SERVER, CONFIG["email"].SMTP_PORT)
        self._connect()

    def _connect(self):
        """Establish connections to IMAP and SMTP servers"""
        try:
            self.imap.login(CONFIG["email"].EMAIL, CONFIG["email"].PASSWORD)
            self.smtp.starttls()
            self.smtp.login(CONFIG["email"].EMAIL, CONFIG["email"].PASSWORD)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to email servers: {str(e)}")

    def fetch_new_emails(self) -> List[EmailData]:
        """Fetch new unread emails from the inbox"""
        self.imap.select('INBOX')
        _, messages = self.imap.search(None, 'UNSEEN')

        emails = []
        for msg_num in messages[0].split():
            try:
                _, msg_data = self.imap.fetch(msg_num, '(RFC822)')
                email_body = msg_data[0][1]
                msg = email.message_from_bytes(email_body)

                content = self._get_email_content(msg)
                thread_id = msg.get('Thread-Index') or msg.get('In-Reply-To')
                references = msg.get('References', '').split() if msg.get('References') else []

                email_data = EmailData(
                    subject=msg['subject'],
                    content=content,
                    thread_id=thread_id,
                    references=references,
                    sender=msg['from'],
                    message_id=msg['message-id']
                )
                emails.append(email_data)
            except Exception as e:
                print(f"Error processing email {msg_num}: {str(e)}")
                continue

        return emails

    def _get_email_content(self, msg) -> str:
        """Extract email content from message"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode()
        return msg.get_payload(decode=True).decode()

    def send_reply(self, original_email: EmailData, reply_content: str):
        """Send a reply to an email"""
        msg = MIMEMultipart()
        msg['From'] = CONFIG["email"].EMAIL
        msg['To'] = original_email.sender
        msg['Subject'] = f"Re: {original_email.subject}"
        msg['In-Reply-To'] = original_email.message_id
        msg['References'] = ' '.join([original_email.message_id] + original_email.references)

        msg.attach(MIMEText(reply_content, 'plain'))

        try:
            self.smtp.send_message(msg)
        except Exception as e:
            print(f"Failed to send reply: {str(e)}")
