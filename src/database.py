import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ARRAY, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(Path(__file__).parent.parent / ".env")


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# --- Models ---

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    paper_names = Column(ARRAY(Text), default=[])

    messages = relationship("Message", back_populates="session", cascade="all, delete")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"))
    role = Column(String)       # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


# --- Create tables ---
def init_db():
    Base.metadata.create_all(bind=engine)


# --- DB Operations ---

def create_session(paper_names: list[str]) -> str:
    """Create a new chat session, return session_id."""
    with SessionLocal() as db:
        session = Session(
            id=str(uuid.uuid4()),
            paper_names=paper_names
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session.id


def save_message(session_id: str, role: str, content: str):
    """Save a single message to the DB."""
    with SessionLocal() as db:
        msg = Message(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content
        )
        db.add(msg)
        db.commit()


def get_last_n_messages(session_id: str, n: int = 5) -> list[dict]:
    """Fetch last N messages for a session (for LLM context window)."""
    with SessionLocal() as db:
        msgs = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(n)
            .all()
        )
        # Return in chronological order
        return [{"role": m.role, "content": m.content} for m in reversed(msgs)]


def get_all_messages(session_id: str) -> list[dict]:
    """Fetch full chat history for display in UI."""
    with SessionLocal() as db:
        msgs = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in msgs]


def get_session(session_id: str) -> dict | None:
    """Get session metadata."""
    with SessionLocal() as db:
        s = db.query(Session).filter(Session.id == session_id).first()
        if not s:
            return None
        return {"id": s.id, "paper_names": s.paper_names, "created_at": str(s.created_at)}
