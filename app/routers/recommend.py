# app/routers/recommend.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..database import get_db
from ..dependencies import get_current_user
from .. import models
from ..recommender import get_recommendations
import json

router = APIRouter(prefix="/recommend", tags=["Recommendation"])

@router.get("/")
def recommend(movie: str, limit: int = 10, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    """
    Hybrid recommendation endpoint.
    - movie: query title (string)
    - limit: number of recs requested
    - requires Authorization Bearer JWT
    """
    # fetch app-user DB object
    current_user = db.query(models.User).filter(models.User.email == user["sub"]).first()
    if not current_user:
        raise HTTPException(status_code=401, detail="Invalid user")

    # get user's stored history titles from DB (if any)
    hist_rows = db.query(models.UserHistory).filter(models.UserHistory.user_id == current_user.id).order_by(models.UserHistory.timestamp.desc()).limit(50).all()
    user_history_titles = [r.movie_title for r in hist_rows] if hist_rows else None

    # get recommendations (hybrid)
    recs = get_recommendations(query_title=movie, top_x=limit, user_history_titles=user_history_titles)

    # log this request into userhistory (store query and recommended titles as JSON)
    recommended_titles = [r["title"] for r in recs]
    hist_entry = models.UserHistory(
        user_id=current_user.id,
        movie_title=movie,
        recommended_titles=json.dumps(recommended_titles)
    )
    db.add(hist_entry)
    db.commit()

    return {"query": movie, "recommendations": recs}
