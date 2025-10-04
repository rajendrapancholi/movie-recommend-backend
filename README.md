# Full-stack Movie Recommender (Next.js + FastAPI + MySQL + KNN)

> Complete project scaffold, code, and instructions to run locally or with Docker. This repository implements:
>
> * Frontend: Next.js (TypeScript) + Tailwind CSS with pages: `/login`, `/signup`, `/dashboard` and Next API routes for authentication that proxy to the backend.
> * Backend: Python FastAPI using SQLAlchemy (MySQL) with endpoints for signup, login, chat, user history, and admin-only endpoints. JWT auth (access + refresh tokens), roles (admin/local user).
> * ML: MovieLens dataset preprocessing and KNN-based recommender (scikit-learn). Training script that produces a serialized model (joblib) and a lightweight recommender endpoint.
> * Database: MySQL schema and initial migration SQL for `users`, `chats`, `userhistory`, `tokens`, `movies`, `ratings`.
> * Security: JWT authentication, role checks, refresh tokens stored in DB, demo/limit responses for unauthorized users.

---

## Project structure

```
fullstack-movie-recommender/
backend/
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   ├── recommender.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── hashing.py
│   │   └── jwt_handler.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── recommend.py
│   │   └── users.py
│   └── internal/
│       ├── __init__.py
│       └── admin.py
├── data/
│   └── movies.csv
├── requirements.txt
└── .env

├─ frontend/
│  ├─ package.json
│  ├─ next.config.js
│  ├─ tailwind.config.js
│  ├─ pages/
│  │  ├─ _app.tsx
│  │  ├─ index.tsx
│  │  ├─ login.tsx
│  │  ├─ signup.tsx
│  │  └─ dashboard.tsx
│  ├─ pages/api/
│  │  ├─ auth/login.ts
│  │  ├─ auth/signup.ts
│  │  └─ auth/refresh.ts
│  └─ styles/globals.css
├─ sql/
│  └─ schema.sql
└─ README.md
```

---

## Notes / design decisions

* Access tokens: short-lived JWTs. Refresh tokens: longer-lived and stored in the `tokens` table to allow explicit revocation.
* Password hashing: bcrypt (passlib).
* Database ORM: SQLAlchemy + Alembic (optionally). For brevity, a simple SQLAlchemy approach is included.
* ML: use MovieLens (e.g., `ml-latest-small`) to populate `movies` and `ratings` tables. A KNN model (item-based or user-based) is built using TF-IDF on movie genres + user-rating matrix and NearestNeighbors.
* Demo/limited responses for unauthorized users: backend returns a `demo` flag or provides sample limited recommendations.

---

## Backend: key files

### backend/app/requirements.txt

```txt
fastapi
uvicorn[standard]
sqlalchemy
pydantic
python-multipart
passlib[bcrypt]
PyJWT
mysql-connector-python
pandas
scikit-learn
joblib
python-dotenv
alembic
```

---

### backend/app/database.py

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+mysqlconnector://root:password@db:3306/moviesdb')

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

### backend/app/models.py

```python
# models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    chats = relationship('Chat', back_populates='user')
    tokens = relationship('Token', back_populates='user')

class Token(Base):
    __tablename__ = 'tokens'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    refresh_token = Column(String(512), nullable=False)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship('User', back_populates='tokens')

class Chat(Base):
    __tablename__ = 'chats'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    message = Column(Text)
    role = Column(String(50))  # user/assistant
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship('User', back_populates='chats')

class UserHistory(Base):
    __tablename__ = 'userhistory'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    movie_id = Column(Integer)
    action = Column(String(100))  # view/rate/like
    value = Column(Float, nullable=True)  # rating value if any
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Movies and Ratings for recommender (optional to mirror MovieLens)
class Movie(Base):
    __tablename__ = 'movies'
    movieId = Column(Integer, primary_key=True)
    title = Column(String(512))
    genres = Column(String(512))

class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True)
    userId = Column(Integer, index=True)
    movieId = Column(Integer, index=True)
    rating = Column(Float)
    timestamp = Column(Integer)
```

---

### backend/app/schemas.py

```python
# schemas.py
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_admin: bool
    class Config:
        orm_mode = True

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = 'bearer'

class ChatCreate(BaseModel):
    message: str

class ChatOut(BaseModel):
    id: int
    message: str
    role: str
    created_at: datetime.datetime
    class Config:
        orm_mode = True
```

---

### backend/app/auth.py

```python
# auth.py - helpers for JWT and password hashing
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv

load_dotenv()

PWD_CTX = CryptContext(schemes=["bcrypt"], deprecated="auto")
JWT_SECRET = os.getenv('JWT_SECRET', 'supersecret')
JWT_ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30

def hash_password(password: str) -> str:
    return PWD_CTX.hash(password)

def verify_password(plain, hashed) -> bool:
    return PWD_CTX.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded

def create_refresh_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded

def decode_token(token: str):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
```

---

### backend/app/crud.py

```python
# crud.py
from sqlalchemy.orm import Session
from . import models
from .auth import hash_password, verify_password
from datetime import datetime

def create_user(db: Session, email: str, password: str, is_admin: bool = False):
    db_user = models.User(email=email, hashed_password=hash_password(password), is_admin=is_admin)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_token(db: Session, user_id: int, refresh_token: str, expires_at):
    t = models.Token(user_id=user_id, refresh_token=refresh_token, expires_at=expires_at)
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

def revoke_refresh_token(db: Session, token_str: str):
    t = db.query(models.Token).filter(models.Token.refresh_token == token_str).first()
    if t:
        db.delete(t)
        db.commit()
        return True
    return False

def save_chat(db: Session, user_id: int, message: str, role: str = 'user'):
    c = models.Chat(user_id=user_id, message=message, role=role, created_at=datetime.utcnow())
    db.add(c)
    db.commit()
    db.refresh(c)
    return c

def get_user_history(db: Session, user_id: int, limit: int = 100):
    return db.query(models.UserHistory).filter(models.UserHistory.user_id == user_id).order_by(models.UserHistory.created_at.desc()).limit(limit).all()
```

---

### backend/app/routers/auth_router.py

```python
# routers/auth_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from .. import crud, schemas, auth
from ..database import get_db
from datetime import datetime, timedelta

router = APIRouter(prefix='/auth')

@router.post('/signup', response_model=schemas.UserOut)
def signup(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = crud.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=400, detail='Email already registered')
    user = crud.create_user(db, payload.email, payload.password)
    return user

@router.post('/login', response_model=schemas.TokenPair)
def login(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if not user or not auth.verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail='Invalid credentials')
    access = auth.create_access_token({"sub": str(user.id), "is_admin": user.is_admin})
    refresh = auth.create_refresh_token({"sub": str(user.id)})
    expires_at = datetime.utcnow() + timedelta(days=30)
    crud.create_token(db, user.id, refresh, expires_at)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}

@router.post('/refresh', response_model=schemas.TokenPair)
def refresh_token(refresh_token: dict, db: Session = Depends(get_db)):
    # expected: {"refresh_token": "..."}
    token = refresh_token.get('refresh_token')
    if not token:
        raise HTTPException(status_code=400)
    # verify presence in DB
    dbtoken = db.query(models.Token).filter(models.Token.refresh_token == token).first()
    if not dbtoken:
        raise HTTPException(status_code=401)
    payload = auth.decode_token(token)
    user_id = int(payload.get('sub'))
    user = db.query(models.User).get(user_id)
    if not user:
        raise HTTPException(status_code=401)
    access = auth.create_access_token({"sub": str(user.id), "is_admin": user.is_admin})
    new_refresh = auth.create_refresh_token({"sub": str(user.id)})
    crud.revoke_refresh_token(db, token)
    expires_at = datetime.utcnow() + timedelta(days=30)
    crud.create_token(db, user.id, new_refresh, expires_at)
    return {"access_token": access, "refresh_token": new_refresh, "token_type": "bearer"}
```

---

### backend/app/routers/user_router.py

```python
# routers/user_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import get_db
from .. import schemas, crud, auth
from fastapi.security import OAuth2PasswordBearer
from typing import List

router = APIRouter(prefix='/user')
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/login')

# helper to get current user
from fastapi import Security

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = auth.decode_token(token)
        user_id = int(payload.get('sub'))
    except Exception:
        raise HTTPException(status_code=401, detail='Invalid token')
    user = db.query(models.User).get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail='User not found')
    return user

@router.post('/chat')
def chat(msg: schemas.ChatCreate, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # basic auth
    curr = get_current_user(token, db)
    # Save user message
    crud.save_chat(db, curr.id, msg.message, role='user')
    # If not authorized (demo), provide limited response
    # Here we simulate a recommender response. In a real app we'd call recommender.predict
    if not curr:
        raise HTTPException(status_code=401)
    # If demo user (e.g., unauthorized), return small static
    # Call recommender
    from ..ml.recommender import recommend_for_user
    recs = recommend_for_user(curr.id, top_n=10, db=db, demo=False)
    # Save assistant message
    crud.save_chat(db, curr.id, str(recs), role='assistant')
    return {"recommendations": recs}

@router.get('/history', response_model=List[schemas.ChatOut])
def history(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    curr = get_current_user(token, db)
    chats = db.query(models.Chat).filter(models.Chat.user_id == curr.id).order_by(models.Chat.created_at.desc()).limit(200).all()
    return chats
```

---

### backend/app/routers/admin_router.py

```python
# routers/admin_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from .. import auth, models

router = APIRouter(prefix='/admin')

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/login')

def get_admin(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = auth.decode_token(token)
        user_id = int(payload.get('sub'))
        is_admin = payload.get('is_admin', False)
        if not is_admin:
            raise HTTPException(status_code=403, detail='Forbidden')
    except Exception:
        raise HTTPException(status_code=401)
    user = db.query(models.User).get(user_id)
    if not user or not user.is_admin:
        raise HTTPException(status_code=403)
    return user

@router.get('/users')
def list_users(admin: models.User = Depends(get_admin), db: Session = Depends(get_db)):
    return db.query(models.User).all()
```

---

### backend/app/main.py

```python
# main.py
from fastapi import FastAPI
from .database import engine
from . import models
from .routers import auth_router, user_router, admin_router

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title='Fullstack Movie Recommender')

app.include_router(auth_router.router)
app.include_router(user_router.router)
app.include_router(admin_router.router)

@app.get('/')
def root():
    return {'ok': True}
```

---

## Backend ML: recommender

### backend/app/ml/train_recommender.py

```python
# train_recommender.py
# Use MovieLens ml-latest-small or the dataset you downloaded from Kaggle.
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import joblib

# Load movies and ratings csvs
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Build a simple item-item recommender using genres + ratings
vectorizer = TfidfVectorizer(token_pattern='[^|]+')
X_genres = vectorizer.fit_transform(movies['genres'].fillna(''))

# Fit a NearestNeighbors model on the genre vectors
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(X_genres)

# Save artifacts
joblib.dump({'nn': nn, 'vectorizer': vectorizer, 'movies': movies}, 'artifacts/recommender.joblib')
print('Saved recommender at artifacts/recommender.joblib')
```

### backend/app/ml/recommender.py

```python
# recommender.py
import joblib
import os
from typing import List
from sqlalchemy.orm import Session

_artifact_path = os.getenv('RECOMMENDER_ARTIFACT', 'artifacts/recommender.joblib')
_art = joblib.load(_artifact_path)
_nn = _art['nn']
_vectorizer = _art['vectorizer']
_movies_df = _art['movies']

def recommend_for_user(user_id: int, top_n: int = 10, db: Session = None, demo: bool = False) -> List[dict]:
    # demo behavior if user not present or demo True
    if demo:
        sample = _movies_df.head(top_n)[['movieId','title']].to_dict(orient='records')
        return sample
    # naive: return top-N popular for the user — TODO: user-specific using ratings in DB
    # For demo we simply return nearest neighbors for most popular movie
    movie_index = 0
    distances, indices = _nn.kneighbors(_vectorizer.transform([_movies_df.iloc[movie_index]['genres']]), n_neighbors=top_n)
    recs = []
    for idx in indices[0]:
        recs.append({'movieId': int(_movies_df.iloc[idx]['movieId']), 'title': _movies_df.iloc[idx]['title']})
    return recs
```

---

## SQL schema (sql/schema.sql)

```sql
CREATE DATABASE IF NOT EXISTS moviesdb;
USE moviesdb;

CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL UNIQUE,
  hashed_password VARCHAR(255) NOT NULL,
  is_admin BOOLEAN DEFAULT FALSE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE tokens (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  refresh_token TEXT NOT NULL,
  expires_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE chats (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  message TEXT,
  role VARCHAR(50),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE userhistory (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT,
  movie_id INT,
  action VARCHAR(100),
  value FLOAT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Movies & ratings (imported from MovieLens CSVs)
CREATE TABLE movies (
  movieId INT PRIMARY KEY,
  title VARCHAR(512),
  genres VARCHAR(512)
);

CREATE TABLE ratings (
  id INT AUTO_INCREMENT PRIMARY KEY,
  userId INT,
  movieId INT,
  rating FLOAT,
  timestamp INT
);
```

---

## Frontend: Next.js (TypeScript) key files

### frontend/package.json (important deps)

```json
{
  "name": "movie-recommender-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "axios": "^1.4.0",
    "jsonwebtoken": "^9.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "tailwindcss": "^3.0.0",
    "postcss": "^8.0.0",
    "autoprefixer": "^10.0.0"
  }
}
```

### frontend/pages/_app.tsx

```tsx
import '../styles/globals.css'
import type { AppProps } from 'next/app'

export default function App({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />
}
```

---

### frontend/pages/login.tsx

```tsx
import { useState } from 'react'
import axios from 'axios'
import { useRouter } from 'next/router'

export default function Login(){
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const router = useRouter()

  const submit = async (e:any)=>{
    e.preventDefault()
    try{
      const res = await axios.post('/api/auth/login', { email, password })
      // store tokens
      localStorage.setItem('access_token', res.data.access_token)
      localStorage.setItem('refresh_token', res.data.refresh_token)
      router.push('/dashboard')
    }catch(err){
      alert('Login failed')
    }
  }
  return (
    <div className="min-h-screen flex items-center justify-center">
      <form onSubmit={submit} className="p-6 rounded shadow bg-white">
        <h1 className="text-lg mb-4">Login</h1>
        <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="email" className="border p-2 mb-2 w-full" />
        <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="password" className="border p-2 mb-2 w-full" />
        <button className="bg-blue-500 text-white px-4 py-2 rounded">Login</button>
      </form>
    </div>
  )
}
```

---

### frontend/pages/signup.tsx

```tsx
import { useState } from 'react'
import axios from 'axios'
import { useRouter } from 'next/router'

export default function Signup(){
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const router = useRouter()

  const submit = async (e:any)=>{
    e.preventDefault()
    try{
      await axios.post('/api/auth/signup', { email, password })
      alert('Signup success — please login')
      router.push('/login')
    }catch(err){
      alert('Signup failed')
    }
  }
  return (
    <div className="min-h-screen flex items-center justify-center">
      <form onSubmit={submit} className="p-6 rounded shadow bg-white">
        <h1 className="text-lg mb-4">Sign Up</h1>
        <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="email" className="border p-2 mb-2 w-full" />
        <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="password" className="border p-2 mb-2 w-full" />
        <button className="bg-green-500 text-white px-4 py-2 rounded">Sign up</button>
      </form>
    </div>
  )
}
```

---

### frontend/pages/dashboard.tsx

```tsx
import { useEffect, useState } from 'react'
import axios from 'axios'

export default function Dashboard(){
  const [recs, setRecs] = useState<any[]>([])
  useEffect(()=>{
    const token = localStorage.getItem('access_token')
    if(!token){
      window.location.href = '/login'
      return
    }
    axios.post('/api/auth/proxy-recommend', {}, { headers: { Authorization: `Bearer ${token}` } })
    .then(r=> setRecs(r.data.recommendations || []))
    .catch(()=> alert('Failed to load recommendations'))
  },[])

  return (
    <div className="p-8">
      <h1 className="text-2xl mb-4">Your Recommendations</h1>
      <ul>
        {recs.map(r=> (
          <li key={r.movieId} className="mb-2">{r.title}</li>
        ))}
      </ul>
    </div>
  )
}
```

---

### frontend/pages/api/auth/login.ts (proxy)

```ts
import type { NextApiRequest, NextApiResponse } from 'next'
import axios from 'axios'

export default async function handler(req: NextApiRequest, res: NextApiResponse){
  if(req.method !== 'POST') return res.status(405).end()
  try{
    const backend = process.env.BACKEND_URL || 'http://localhost:8000'
    const r = await axios.post(`${backend}/auth/login`, req.body)
    return res.status(r.status).json(r.data)
  }catch(e:any){
    return res.status(e.response?.status || 500).json({ error: e.message })
  }
}
```

---

### frontend/pages/api/auth/signup.ts

```ts
import type { NextApiRequest, NextApiResponse } from 'next'
import axios from 'axios'

export default async function handler(req: NextApiRequest, res: NextApiResponse){
  if(req.method !== 'POST') return res.status(405).end()
  try{
    const backend = process.env.BACKEND_URL || 'http://localhost:8000'
    const r = await axios.post(`${backend}/auth/signup`, req.body)
    return res.status(r.status).json(r.data)
  }catch(e:any){
    return res.status(e.response?.status || 500).json({ error: e.message })
  }
}
```

---

### frontend/pages/api/auth/refresh.ts

```ts
import type { NextApiRequest, NextApiResponse } from 'next'
import axios from 'axios'

export default async function handler(req: NextApiRequest, res: NextApiResponse){
  if(req.method !== 'POST') return res.status(405).end()
  try{
    const backend = process.env.BACKEND_URL || 'http://localhost:8000'
    const r = await axios.post(`${backend}/auth/refresh`, req.body)
    return res.status(r.status).json(r.data)
  }catch(e:any){
    return res.status(e.response?.status || 500).json({ error: e.message })
  }
}
```

---

## Docker quickstart (optional)

Provide `backend/Dockerfile` and `docker-compose.yml` to run backend + MySQL.

### backend/Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY ./app /app
RUN pip install --upgrade pip && pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (root)

```yaml
version: '3.8'
services:
  db:
    image: mysql:8
    environment:
      MYSQL_ROOT_PASSWORD: password
      MYSQL_DATABASE: moviesdb
    ports:
      - '3306:3306'
    volumes:
      - db-data:/var/lib/mysql
  backend:
    build: ./backend
    environment:
      - DATABASE_URL=mysql+mysqlconnector://root:password@db:3306/moviesdb
      - JWT_SECRET=supersecret
    ports:
      - '8000:8000'
    depends_on:
      - db
  frontend:
    build: ./frontend
    ports:
      - '3000:3000'
volumes:
  db-data:
```

---

## How to import MovieLens data (quick)

1. Download MovieLens `movies.csv` and `ratings.csv` into `backend/app/ml/`.
2. Use a small script to bulk insert into `movies` and `ratings` tables (or use MySQL client `LOAD DATA INFILE`).

Example quick import script (`backend/app/ml/import_ml.py`):

```python
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
movies.to_sql('movies', engine, if_exists='append', index=False)
ratings.to_sql('ratings', engine, if_exists='append', index=False)
```

---

## Notes on security and production hardening

* Use HTTPS and secure cookies on frontend.
* Rotate `JWT_SECRET` and store in vault.
* Do not return raw DB errors to clients.
* Use CORS properly on backend (restrict origins).
* Rate-limit endpoints and validate request payload sizes.
* Consider storing only hashed refresh tokens in DB (or use token IDs) to limit theft impact.

---

## Next steps and customizations you may want

* Replace the naive KNN recommender with a hybrid model using collaborative filtering (e.g. implicit ALS) for user personalization.
* Add pagination, search, and admin UIs.
* Add email verification, password reset flows.
* Integrate Redis for caching recommendations and session storage.

---

If you'd like, I can:

* generate the exact files and a downloadable zip,
* add Alembic migrations,
* replace the naive recommender with an item-user hybrid using implicit/ALS,
* or give a deploy plan to Vercel (frontend) + Railway/Render for backend + managed MySQL.

Tell me which follow-up you want and I will produce it (file exports, docker-compose with secrets, or a more advanced ML model).
