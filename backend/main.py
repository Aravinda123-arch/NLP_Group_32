from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)


from backend.model_manager import (
    model_manager,
)

from backend.prediction_service import (
    predict_news,
)

from backend.schemas import (
    NewsRequest,
    PredictionResponse,
    HealthResponse,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    # ========================================================
    # STARTUP
    # ========================================================

    try:

        model_manager.load_models()

    except Exception as error:

        print(
            "\nMODEL STARTUP ERROR:"
        )

        print(
            type(error).__name__,
            ":",
            error,
        )

        raise


    yield


    # ========================================================
    # SHUTDOWN
    # ========================================================

    model_manager.unload_models()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "Fake News Detection API"
    ),

    description=(
        "Random Forest + BERT "
        "fake-news prediction API"
    ),

    version="1.0.0",

    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

# During development your frontend may run through:
#
# VS Code Live Server:
# http://127.0.0.1:5500
#
# or:
# http://localhost:5500

FRONTEND_ORIGINS = [
    "*"
]


app.add_middleware(

    CORSMiddleware,

    allow_origins=(
        FRONTEND_ORIGINS
    ),

    allow_credentials=False,

    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api")
async def root():
    return {
        "message": "Fake News Detection API is running",
        "docs": "/docs",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
@app.get(
    "/api/health",
    response_model=HealthResponse,
)
async def health_check():
    return HealthResponse(
        **model_manager.get_status()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
@app.post(
    "/api/predict",
    response_model=PredictionResponse,
)
async def predict_api(
    request: NewsRequest,
):
    try:
        return predict_news(
            headline=request.headline,
            article=request.article,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# SERVE FRONTEND STATIC FILES
# ============================================================

from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

