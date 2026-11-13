"""Gmail-related service helpers."""

from .client import (
    GMAIL_NOT_CONNECTED_ERROR,
    disconnect_account,
    execute_gmail_tool,
    fetch_status,
    get_active_gmail_email,
    get_active_gmail_user_id,
    initiate_connect,
    is_gmail_connected,
    resolve_active_gmail_email,
)
from .importance_classifier import classify_email_importance
from .importance_watcher import ImportantEmailWatcher, get_important_email_watcher
from .processing import EmailTextCleaner, ProcessedEmail, parse_gmail_fetch_response
from .seen_store import GmailSeenStore

__all__ = [
    "GMAIL_NOT_CONNECTED_ERROR",
    "is_gmail_connected",
    "execute_gmail_tool",
    "fetch_status",
    "initiate_connect",
    "disconnect_account",
    "get_active_gmail_user_id",
    "get_active_gmail_email",
    "resolve_active_gmail_email",
    "classify_email_importance",
    "ImportantEmailWatcher",
    "get_important_email_watcher",
    "EmailTextCleaner",
    "ProcessedEmail",
    "parse_gmail_fetch_response",
    "GmailSeenStore",
]
