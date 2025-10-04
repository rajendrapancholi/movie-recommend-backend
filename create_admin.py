# create_admin.py
from app.database import get_db
from app.models import User
from app.utils.hashing import hash_password

def main():
    db = next(get_db())

    # Check if admin already exists
    existing = db.query(User).filter(User.role == "admin").first()
    if existing:
        print(f"Admin already exists: {existing.username}")
        return

    # Create admin user
    admin_user = User(
        username="admin",
        email="admin@gmail.com",
        hashed_password=hash_password("RajendraPancholi$$123"),
        role="admin"
    )

    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)

    print(f"Admin created successfully: {admin_user.username} ({admin_user.email})")

if __name__ == "__main__":
    main()
