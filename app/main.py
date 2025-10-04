from fastapi import FastAPI
from .database import Base, engine
from . import models
from .routers import auth, recommend, users
from .internal import admin
import os

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Movie Recommendation System")

PORT = int(os.getenv("PORT", 8000))

# Register routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(recommend.router)
app.include_router(admin.router)

@app.get("/")
def root():
    return {"message": "Welcome to the Movie Recommendation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)

'''
pip install -r requirements.txt
uvicorn app.main:app --reload
'''