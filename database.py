from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    ForeignKey,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import sqlalchemy
from pathlib import Path

Base = declarative_base()


class DBAudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sentences = relationship("DBSentence", back_populates="audio_file")


class DBSentence(Base):
    __tablename__ = "sentences"

    id = Column(Integer, primary_key=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"))
    text = Column(String, nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    audio_file = relationship("DBAudioFile", back_populates="sentences")
    translations = relationship("DBTranslation", back_populates="sentence")


class DBTranslation(Base):
    __tablename__ = "translations"

    id = Column(Integer, primary_key=True)
    sentence_id = Column(Integer, ForeignKey("sentences.id"))
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    language = Column(String, nullable=False)
    translation = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sentence = relationship("DBSentence", back_populates="translations")

    # Add a unique constraint to prevent duplicate translations
    __table_args__ = (
        sqlalchemy.UniqueConstraint(
            "sentence_id", "provider", "model", "language", name="unique_translation"
        ),
    )


# Create database engine and session factory
engine = create_engine("sqlite:///translations.db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
