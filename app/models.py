from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True)
    hashed_password = Column(String(255))
    role = Column(String(10), default='user')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    history = relationship("UserHistory", back_populates="user")
'''
class UserHistory(Base):
    __tablename__ = 'userhistory'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_title = Column(String(255))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="history")
'''

class UserHistory(Base):
    __tablename__ = 'userhistory'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    movie_title = Column(String(255))
    recommended_titles = Column(String(1000))  # <-- NEW: store JSON string
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="history")
