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
