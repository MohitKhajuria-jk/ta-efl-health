"""Workstation sessions: existing PBKDF2 identities, opaque cookies and CSRF protection."""
import secrets
import threading
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Request

COOKIE = "ta_efl_session"


class Sessions:
    def __init__(self):
        self.lock = threading.RLock()
        self.sessions = {}
        self.attempts = defaultdict(deque)

    def check_rate(self, key, limit=10, window=60):
        with self.lock:
            now = time.monotonic()
            q = self.attempts[key]
            while q and q[0] < now - window:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(429, "Too many requests. Try again shortly.")
            q.append(now)

    def create(self, username):
        with self.lock:
            now = time.monotonic()
            self.sessions = {k:v for k,v in self.sessions.items() if v["expires"] > now}
            if len(self.sessions) >= 128:
                self.sessions.pop(next(iter(self.sessions)))
            token = secrets.token_urlsafe(32)
            session = dict(username=username, csrf=secrets.token_urlsafe(32), expires=now + 8 * 3600)
            self.sessions[token] = session
            return token, session

    def get(self, request: Request):
        token = request.cookies.get(COOKIE, "")
        with self.lock:
            session = self.sessions.get(token)
            if not session or session["expires"] < time.monotonic():
                self.sessions.pop(token, None)
                return None
            return dict(session)


def authorized(request: Request):
    session = request.app.state.sessions.get(request)
    if not session:
        raise HTTPException(401, "Please sign in to your research workspace.")
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), session["csrf"]):
            raise HTTPException(403, "Session verification failed. Refresh and try again.")
    return session
