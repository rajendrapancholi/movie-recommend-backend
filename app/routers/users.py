from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db
from ..dependencies import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me", response_model=schemas.UserOut)
def read_users_me(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == current_user["sub"]).first()
    return user
