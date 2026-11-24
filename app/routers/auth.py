import json
from datetime import datetime
"""Authentication router handling User registration, login, and profile retrieval."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import User, InterviewSession
from app.schemas import (
    UserCreate,
    TeacherRegisterRequest,
    UserLogin,
    UserResponse,
    Token,
    UserProfileUpdate,
    ChangePasswordRequest,
    MessageResponse,
    GoogleLoginRequest,
    UserProfileSummary,
    VivaStats,
    InterviewStats,
    QuickStats,
    RecentActivityItem,
    LearningGoalItem,
    AddGoalRequest,
)
from app.security import (
    verify_password,
    hash_password,
    create_access_token,
    get_current_user,
    require_teacher,
    require_student,
)
from app.services import db_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account"
)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user candidate account.
    - Validates name, unique email, and password (min 8 chars).
    - Hashes password using bcrypt.
    - Returns the created user object (excluding password).
    """
    # Check if a user with this email already exists
    existing_user = db_service.get_user_by_email(db, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists."
        )

    try:
        user = db_service.create_user(db, user_in=user_in, role="student")
        logger.info(f"Successfully registered student user with ID {user.id}")
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error during registration: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating your account."
        )


@router.post(
    "/register-teacher",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new authorized teacher account"
)
def register_teacher(teacher_in: TeacherRegisterRequest, db: Session = Depends(get_db)):
    """
    Register an authorized educator account.
    - Validates server-configured teacher invitation code.
    - Rejects invalid or missing invitation codes with 403 Forbidden.
    - Creates user account with role='teacher'.
    """
    settings = get_settings()
    if not teacher_in.invitation_code or teacher_in.invitation_code.strip() != settings.TEACHER_INVITATION_CODE.strip():
        logger.warning(f"Unauthorized teacher registration attempt for email {teacher_in.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid teacher authorization code. Please contact your system administrator."
        )

    existing_user = db_service.get_user_by_email(db, email=teacher_in.email)
    if existing_user:
        # User already exists; since they supplied the verified teacher authorization code,
        # update their credentials and upgrade them to teacher role.
        existing_user.role = "teacher"
        existing_user.hashed_password = hash_password(teacher_in.password)
        if teacher_in.name and teacher_in.name.strip():
            existing_user.name = teacher_in.name.strip()
        db.commit()
        db.refresh(existing_user)
        logger.info(f"Successfully claimed/upgraded existing account {existing_user.id} to teacher role.")
        return existing_user

    try:
        user = db_service.create_user(db, user_in=teacher_in, role="teacher")
        logger.info(f"Successfully registered authorized teacher with ID {user.id}")
        return user
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error during teacher registration: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating teacher account."
        )


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Authenticate and receive a JWT access token"
)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate candidate credentials using email and password.
    - Verifies password hash using native bcrypt.
    - Validates portal-specific role requirements if requested.
    - Generates signed JWT access token containing role claim.
    - Returns token and public user profile with role.
    """
    user = db_service.get_user_by_email(db, email=login_in.email)
    if not user or not verify_password(login_in.password, user.hashed_password):
        if user and user.auth_provider == "google":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="This account was registered using Google Sign-In. Please sign in with Google.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Do not distinguish whether email or password was wrong to prevent enumeration
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate portal-specific role constraint if specified by frontend portal
    if login_in.role:
        expected_role = login_in.role.strip().lower()
        user_role = (user.role or "student").strip().lower()
        if expected_role == "teacher" and user_role != "teacher":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: This account has student privileges. Please switch to the Student Login."
            )
        if expected_role == "student" and user_role != "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This is an authorized teacher account. Please use the Teacher Login."
            )

    # Generate JWT Bearer access token containing role claim
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "name": user.name, "role": user.role}
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.get(
    "/teacher/verify",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify teacher authorization and permissions"
)
def verify_teacher_status(current_user: User = Depends(require_teacher)):
    """
    Verifies that the caller has authenticated teacher privileges.
    Protected by require_teacher dependency.
    """
    return MessageResponse(
        message=f"Welcome Educator {current_user.name}. Teacher credentials verified successfully."
    )


@router.post(
    "/google",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Authenticate or register using Google Sign-In"
)
def google_login(payload: GoogleLoginRequest, db: Session = Depends(get_db)):
    """
    Verify Google OAuth 2.0 credential and authenticate or register candidate.
    - Validates server configuration (GOOGLE_CLIENT_ID).
    - Verifies ID token signature, issuer, audience, and expiration via google.oauth2.id_token.
    - Checks email_verified claim.
    - Enforces strict account linking rules to prevent account hijacking.
    - Returns application JWT token and public user profile.
    """
    settings = get_settings()

    # 1. Verify server configuration
    client_id = settings.GOOGLE_CLIENT_ID
    if not client_id or "your_google_client_id" in client_id.lower():
        logger.error("Google Sign-In rejected: GOOGLE_CLIENT_ID is not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google Sign-In is not configured on this server. Please configure GOOGLE_CLIENT_ID."
        )

    # 2. Cryptographic signature and token verification using official Google Auth library
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests

    try:
        id_info = id_token.verify_oauth2_token(
            payload.credential,
            google_requests.Request(),
            audience=client_id,
            clock_skew_in_seconds=10
        )
    except ValueError as exc:
        logger.warning(f"Google ID token verification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Google authentication credential.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        logger.error(f"Unexpected error verifying Google credential: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify Google authentication credential.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Verify issuer
    issuer = id_info.get("iss")
    if issuer not in ["accounts.google.com", "https://accounts.google.com"]:
        logger.warning(f"Invalid Google token issuer: {issuer}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token issuer.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Verify email verification status
    email = id_info.get("email")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google credential does not contain an email address."
        )

    if not id_info.get("email_verified", False):
        logger.warning(f"Google email not verified: {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The Google account email is not verified by Google."
        )

    google_sub = id_info.get("sub")
    if not google_sub:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google credential does not contain a valid subject identifier."
        )

    clean_email = email.lower().strip()
    name = (id_info.get("name") or clean_email.split("@")[0]).strip()
    picture = id_info.get("picture")

    # 5. Account resolution & Conflict prevention
    # First search by Google subject ID
    user = db_service.get_user_by_google_id(db, google_id=google_sub)
    if not user:
        # Check if an account already exists with this email address
        existing_email_user = db_service.get_user_by_email(db, email=clean_email)
        if existing_email_user:
            if existing_email_user.google_id == google_sub:
                user = existing_email_user
            else:
                # Security Rule: Do not automatically link accounts solely because emails match.
                # Protect existing email/password accounts against takeover.
                logger.warning(
                    f"Account conflict: {clean_email} already exists with auth_provider={existing_email_user.auth_provider}. Google sub={google_sub}"
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="An account with this email address already exists using email and password authentication. Please sign in with your email and password."
                )
        else:
            # Register new candidate account authenticated via Google
            try:
                user = db_service.create_google_user(
                    db,
                    email=clean_email,
                    name=name,
                    google_id=google_sub,
                    avatar_url=picture
                )
                logger.info(f"Registered new Google candidate: ID {user.id}, email {user.email}")
            except Exception as exc:
                logger.error(f"Error persisting new Google user: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user account from Google credential."
                )

    # 6. Issue application JWT Bearer token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id, "name": user.name}
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get currently authenticated user"
)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """
    Protected route: returns the public profile of the currently logged-in user.
    Requires Bearer token authorization header: 'Authorization: Bearer <token>'.
    """
    return current_user


@router.put(
    "/profile",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Update candidate profile details"
)
def update_profile(
    profile_in: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Protected route: updates candidate profile details (such as full name and avatar URL).
    """
    if profile_in.name and profile_in.name.strip():
        current_user.name = profile_in.name.strip()
    if profile_in.avatar_url is not None:
        current_user.avatar_url = profile_in.avatar_url.strip() if profile_in.avatar_url.strip() else None
    if profile_in.bio is not None:
        current_user.bio = profile_in.bio.strip() if profile_in.bio.strip() else None
    if profile_in.learning_interests is not None:
        current_user.learning_interests = profile_in.learning_interests.strip() if profile_in.learning_interests.strip() else None
    if profile_in.learning_goals is not None:
        current_user.learning_goals = profile_in.learning_goals.strip() if profile_in.learning_goals.strip() else None

    db.commit()
    db.refresh(current_user)
    logger.info(f"Updated profile for user ID {current_user.id}")
    return current_user


@router.get(
    "/profile-summary",
    response_model=UserProfileSummary,
    status_code=status.HTTP_200_OK,
    summary="Get user profile summary with viva, interview, and activity statistics"
)
def get_user_profile_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Computes cross-session viva and interview statistics, quick metrics,
    and recent activity timeline for the authenticated candidate.
    """
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == current_user.id)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )

    viva_sessions = []
    interview_sessions = []

    for s in sessions:
        mode = (s.interview_mode or "").lower()
        if "viva" in mode or mode in ("academic_viva", "oral_viva"):
            viva_sessions.append(s)
        else:
            interview_sessions.append(s)

    def compute_stats(sess_list):
        total = len(sess_list)
        completed = [s for s in sess_list if s.status == "completed" or s.report is not None]
        in_progress = [s for s in sess_list if s.status == "in_progress" and s.report is None]
        scores = [
            s.report.overall_score for s in completed
            if s.report and s.report.overall_score is not None
        ]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        high_score = round(max(scores), 1) if scores else 0.0
        q_answered = sum(
            len([q for q in s.questions if q.answer is not None])
            for s in sess_list
        )
        topics = list(dict.fromkeys(s.topic for s in sess_list if s.topic))[:5]

        return {
            "total_sessions": total,
            "completed_sessions": len(completed),
            "in_progress_sessions": len(in_progress),
            "average_score": avg_score,
            "highest_score": high_score,
            "questions_answered": q_answered,
            "top_topics": topics,
        }

    v_stats = compute_stats(viva_sessions)
    i_stats = compute_stats(interview_sessions)

    total_all = len(sessions)
    completed_all = v_stats["completed_sessions"] + i_stats["completed_sessions"]
    in_progress_all = v_stats["in_progress_sessions"] + i_stats["in_progress_sessions"]
    all_scores = [
        s.report.overall_score for s in sessions
        if s.report and s.report.overall_score is not None
    ]
    overall_avg = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
    total_q_answered = v_stats["questions_answered"] + i_stats["questions_answered"]
    comp_rate = round((completed_all / total_all * 100), 1) if total_all > 0 else 0.0

    q_stats = {
        "total_sessions": total_all,
        "completed_sessions": completed_all,
        "in_progress_sessions": in_progress_all,
        "overall_average_score": overall_avg,
        "total_questions_answered": total_q_answered,
        "completion_rate": comp_rate,
    }

    # Build chronological recent activities
    activities = []
    for s in sessions[:15]:
        is_viva = s in viva_sessions
        act_type = "viva" if is_viva else "interview"
        label = "Viva Session" if is_viva else "Technical Interview"
        status_text = "Completed" if (s.status == "completed" or s.report) else "In Progress"
        title = f"{status_text} {label}: {s.topic}"
        desc = f"Subject: {s.subject} • Level: {s.target_level}"
        ts = s.completed_at or s.started_at

        activities.append({
            "id": f"sess-{s.id}",
            "activity_type": act_type,
            "title": title,
            "description": desc,
            "status": s.status,
            "score": s.report.overall_score if s.report else None,
            "questions_count": len(s.questions),
            "timestamp": ts,
            "session_id": s.id,
        })

    # Add account registration activity
    if current_user.created_at:
        activities.append({
            "id": f"user-reg-{current_user.id}",
            "activity_type": "account",
            "title": "Account Registered",
            "description": f"Registered for AI Viva & Interview Coach via {current_user.auth_provider.capitalize()}",
            "status": "completed",
            "score": None,
            "questions_count": None,
            "timestamp": current_user.created_at,
            "session_id": None,
        })

    # Sort all activities newest first
    activities.sort(key=lambda a: a["timestamp"], reverse=True)

    return {
        "user": current_user,
        "viva_stats": v_stats,
        "interview_stats": i_stats,
        "quick_stats": q_stats,
        "recent_activities": activities[:20],
    }


@router.post(
    "/change-password",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Change user password"
)
def change_password(
    pwd_in: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Protected route: verifies current password and updates password hash.
    """
    if current_user.auth_provider == "google" and current_user.hashed_password.startswith("!google_oauth_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account is authenticated through Google Sign-In and does not have a local password."
        )

    if not verify_password(pwd_in.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect."
        )

    current_user.hashed_password = hash_password(pwd_in.new_password)
    db.commit()
    logger.info(f"Password changed for user ID {current_user.id}")
    return MessageResponse(message="Password successfully updated.", success=True)

@router.get(
    "/goals",
    response_model=list[LearningGoalItem],
    status_code=status.HTTP_200_OK,
    summary="Get user learning goals"
)
def get_user_learning_goals(
    current_user: User = Depends(get_current_user),
):
    """Returns the authenticated candidate's learning goals."""
    if not current_user.learning_goals:
        return []
    try:
        data = json.loads(current_user.learning_goals)
        return data if isinstance(data, list) else []
    except Exception:
        return []


@router.post(
    "/goals",
    response_model=list[LearningGoalItem],
    status_code=status.HTTP_201_CREATED,
    summary="Add a learning goal"
)
def add_user_learning_goal(
    goal_in: AddGoalRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Adds a new learning goal to the authenticated candidate's profile."""
    goals = []
    if current_user.learning_goals:
        try:
            parsed = json.loads(current_user.learning_goals)
            if isinstance(parsed, list):
                goals = parsed
        except Exception:
            goals = []

    new_goal = {
        "id": f"goal-{int(datetime.now().timestamp() * 1000)}",
        "title": goal_in.title.strip(),
        "target_date": goal_in.target_date.strip() if goal_in.target_date else None,
        "completed": False,
        "created_at": datetime.now().isoformat(),
    }
    goals.append(new_goal)
    current_user.learning_goals = json.dumps(goals)
    db.commit()
    db.refresh(current_user)
    return goals


@router.delete(
    "/goals/{goal_id}",
    response_model=list[LearningGoalItem],
    status_code=status.HTTP_200_OK,
    summary="Remove a learning goal"
)
def delete_user_learning_goal(
    goal_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Removes a learning goal by ID."""
    goals = []
    if current_user.learning_goals:
        try:
            parsed = json.loads(current_user.learning_goals)
            if isinstance(parsed, list):
                goals = [g for g in parsed if g.get("id") != goal_id]
        except Exception:
            goals = []

    current_user.learning_goals = json.dumps(goals)
    db.commit()
    db.refresh(current_user)
    return goals


@router.patch(
    "/goals/{goal_id}/toggle",
    response_model=list[LearningGoalItem],
    status_code=status.HTTP_200_OK,
    summary="Toggle learning goal completion"
)
def toggle_user_learning_goal(
    goal_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggles a learning goal's completed status."""
    goals = []
    if current_user.learning_goals:
        try:
            parsed = json.loads(current_user.learning_goals)
            if isinstance(parsed, list):
                for g in parsed:
                    if g.get("id") == goal_id:
                        g["completed"] = not g.get("completed", False)
                goals = parsed
        except Exception:
            goals = []

    current_user.learning_goals = json.dumps(goals)
    db.commit()
    db.refresh(current_user)
    return goals

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
AVATARS_DIR = UPLOAD_DIR / "avatars"
AVATARS_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/avatar",
    response_model=UserResponse,
    summary="Upload user avatar image from gallery or file picker"
)
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Accepts an image file from the user's gallery or device, validates it,
    auto-rotates based on EXIF, crops/resizes to 400x400 square thumbnail,
    and updates current_user.avatar_url.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded."
        )
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large (maximum size is 10MB)."
        )

    try:
        import io
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(contents))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = ImageOps.fit(img, (400, 400), Image.Resampling.LANCZOS)

        timestamp = int(datetime.now().timestamp())
        filename = f"avatar_user_{current_user.id}_{timestamp}.jpg"
        target_path = AVATARS_DIR / filename
        img.save(target_path, "JPEG", quality=88, optimize=True)
    except Exception as e:
        logger.error(f"Failed to process avatar image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Please upload a valid image file (JPG, PNG, WebP): {str(e)}"
        )

    # Clean up previous local avatar file if existed
    if current_user.avatar_url and current_user.avatar_url.startswith("/uploads/avatars/"):
        old_file = UPLOAD_DIR / current_user.avatar_url.replace("/uploads/", "")
        if old_file.exists() and old_file.is_file():
            try:
                old_file.unlink()
            except Exception:
                pass

    current_user.avatar_url = f"/uploads/avatars/{filename}"
    db.commit()
    db.refresh(current_user)
    return current_user

