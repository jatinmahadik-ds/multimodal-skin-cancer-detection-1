"""
main.py
================================================================================
Vireo Clinical Doctor Portal — FastAPI Backend

Integrates the 2 verified AI models:
    - Model 1: EfficientNetB0 (ISIC 2019) -> Dermoscopic Image Analysis
    - Model 2: Clinical-Only XGBoost (PAD-UFES-20, 53 dummy features)

ResNet50 Disconnected:
    ResNet50 is not loaded or executed.
    Outputs from the two models are presented independently without an
    unvalidated ensemble combination.

Provides SQLite persistence for patients and reports, and PDF generation.
================================================================================
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import database as db
from model_loader import registry
from pdf_generator import generate_patient_report_pdf
from prediction import (
    ImageValidationError,
    InferenceError,
    run_prediction,
    validate_image_bytes,
)
from schemas import (
    CLINICAL_LABELS,
    DoctorLogin,
    DoctorSignup,
    PatientCreate,
    PatientUpdate,
    ReportCreate,
    ValidationError,
    clinical_options_payload,
    validate_clinical,
)

MAX_UPLOAD_BYTES = int(float(os.environ.get("SKIN_MAX_UPLOAD_MB", "10")) * 1024 * 1024)

DEFAULT_ORIGINS = (
    "http://127.0.0.1:5500,http://localhost:5500,"
    "http://127.0.0.1:5501,http://localhost:5501,"
    "http://127.0.0.1:3000,http://localhost:3000,"
    "http://127.0.0.1:8080,http://localhost:8080,"
    "null"  # local file:// origin
)
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("SKIN_CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]

DISCLAIMER = (
    "This is an experimental research prototype, not a clinical device, and its "
    "output is not a medical diagnosis. Always seek professional assessment of "
    "concerning lesions."
)


def clinical_form_dependency(
    age: Optional[str] = Form(None),
    diameter_1: Optional[str] = Form(None),
    diameter_2: Optional[str] = Form(None),
    fitspatrick: Optional[str] = Form(None),
    smoke: Optional[str] = Form(None),
    drink: Optional[str] = Form(None),
    pesticide: Optional[str] = Form(None),
    skin_cancer_history: Optional[str] = Form(None),
    cancer_history: Optional[str] = Form(None),
    has_piped_water: Optional[str] = Form(None),
    has_sewage_system: Optional[str] = Form(None),
    itch: Optional[str] = Form(None),
    grew: Optional[str] = Form(None),
    hurt: Optional[str] = Form(None),
    changed: Optional[str] = Form(None),
    bleed: Optional[str] = Form(None),
    elevation: Optional[str] = Form(None),
    background_father: Optional[str] = Form(None),
    background_mother: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
) -> dict[str, Optional[str]]:
    return {
        "age": age,
        "diameter_1": diameter_1,
        "diameter_2": diameter_2,
        "fitspatrick": fitspatrick,
        "smoke": smoke,
        "drink": drink,
        "pesticide": pesticide,
        "skin_cancer_history": skin_cancer_history,
        "cancer_history": cancer_history,
        "has_piped_water": has_piped_water,
        "has_sewage_system": has_sewage_system,
        "itch": itch,
        "grew": grew,
        "hurt": hurt,
        "changed": changed,
        "bleed": bleed,
        "elevation": elevation,
        "background_father": background_father,
        "background_mother": background_mother,
        "gender": gender,
        "region": region,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite tables and run migrations
    db.init_database()
    # Load active models (EfficientNetB0 + Clinical XGBoost + Schema)
    registry.load_all()
    yield


app = FastAPI(
    title="Vireo Clinical API",
    version="1.0.0",
    description="Multimodal Skin Cancer Detection Doctor Portal API (EfficientNetB0 + Clinical XGBoost)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_UPLOAD_BYTES + 1_000_000:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."},
                    )
            except ValueError:
                pass
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    for err in exc.errors():
        loc = err.get("loc", ())
        if err.get("type") == "missing" and loc and loc[-1] == "image":
            return JSONResponse(
                status_code=400,
                content={"detail": "Missing required image upload (field name must be 'image')."},
            )
    return JSONResponse(status_code=422, content={"detail": "Request validation failed.", "errors": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak internal stack traces to users
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error while processing the request."},
    )


# ---------------------------------------------------------------------------
# Status and Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Vireo Clinical API",
        "version": "1.0.0",
        "status": "online",
        "pipeline": "dual_model_independent",
        "models": {
            "model_1": "EfficientNetB0 (ISIC 2019) - Dermoscopic Image Analysis",
            "model_2": "Clinical-Only XGBoost (PAD-UFES-20) - 53 Clinical Dummy Features",
        },
        "dual_model_status": "Independent dual-model inference. No unvalidated ensemble applied.",
        "endpoints": [
            "/health", "/clinical-options", "/predict", "/patients",
            "/reports", "/reports/{report_id}/pdf", "/stats"
        ]
    }


@app.get("/health")
def health():
    ready = registry.ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "healthy" if ready else "unhealthy",
            "models": {
                "efficientnet_b0": registry.loaded.get("efficientnet_b0", False),
                "clinical_xgboost": registry.loaded.get("clinical_xgboost", False),
                "feature_schema": registry.loaded.get("feature_schema", False),
            },
            "resnet50_loaded": False,
            "ensemble_validated": False,
            "errors": registry.errors if registry.errors else None,
        }
    )


@app.get("/clinical-options")
def clinical_options():
    if not registry.loaded["feature_schema"]:
        return JSONResponse(status_code=503, content={"detail": "Feature schema is not loaded."})
    return clinical_options_payload(registry)


@app.get("/stats")
def dashboard_stats():
    return db.get_dashboard_stats()


# ---------------------------------------------------------------------------
# AI Prediction Endpoint
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict_endpoint(
    image: UploadFile = File(..., description="Skin lesion image (JPG/PNG)"),
    clinical: dict[str, Optional[str]] = Depends(clinical_form_dependency),
):
    if not registry.ready:
        return JSONResponse(
            status_code=503,
            content={"detail": "AI Models are not ready. Please check /health.", "errors": registry.errors},
        )

    try:
        image_bytes = await image.read()
    except Exception:
        return JSONResponse(status_code=400, content={"detail": "Could not read uploaded image."})

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB size limit."},
        )

    # 1. Image validation
    try:
        validate_image_bytes(image_bytes)
    except ImageValidationError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    # 2. Clinical data validation
    try:
        validated_clinical = validate_clinical(registry, clinical)
    except ValidationError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": "Clinical validation failed.", "errors": exc.errors},
        )

    # 3. 3-Model Independent Inference
    try:
        result = run_prediction(registry, image_bytes, validated_clinical)
    except InferenceError as exc:
        return JSONResponse(status_code=500, content={"detail": f"Inference failed: {exc}"})

    result["disclaimer"] = DISCLAIMER
    return result


# ---------------------------------------------------------------------------
# Patient Management Endpoints
# ---------------------------------------------------------------------------
@app.get("/patients")
def get_patients_list(search: Optional[str] = None, risk: Optional[str] = None, date: Optional[str] = None):
    return db.list_patients(search=search, risk=risk, date=date)


@app.post("/patients")
def create_patient_endpoint(patient_in: PatientCreate):
    data = patient_in.model_dump()
    patient = db.create_patient(data)
    return patient


@app.get("/patients/{patient_id}")
def get_patient_endpoint(patient_id: str):
    p = db.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    reports = db.list_reports(patient_id=patient_id)
    return {"patient": p, "reports": reports}


@app.put("/patients/{patient_id}")
def update_patient_endpoint(patient_id: str, patient_in: PatientUpdate):
    existing = db.get_patient(patient_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Patient not found.")

    data = patient_in.model_dump()
    updated = db.update_patient(patient_id, data)
    return updated


@app.get("/patients/{patient_id}/reports")
def get_patient_reports_endpoint(patient_id: str):
    existing = db.get_patient(patient_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return db.list_reports(patient_id=patient_id)


# ---------------------------------------------------------------------------
# Report Generation, Persistence and Download Endpoints
# ---------------------------------------------------------------------------
@app.post("/reports")
def create_report_endpoint(report_in: ReportCreate):
    payload = report_in.model_dump()
    pid = payload["patient_id"]

    patient = db.get_patient(pid)
    if not patient:
        if payload.get("patient_name") and payload.get("patient_age") is not None:
            patient = db.create_patient({
                "patient_id": pid,
                "name": payload.get("patient_name"),
                "age": payload.get("patient_age"),
                "gender": payload.get("patient_gender") or "Other",
                "phone": payload.get("patient_phone") or "",
                "email": payload.get("patient_email") or "",
            })
        else:
            raise HTTPException(status_code=404, detail=f"Patient '{pid}' does not exist.")

    saved = db.save_report(payload)
    return saved


@app.get("/reports")
def list_all_reports_endpoint():
    return db.list_reports()


@app.get("/reports/{report_id}")
def get_report_endpoint(report_id: str):
    rep = db.get_report(report_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found.")
    return rep


@app.get("/reports/{report_id}/pdf")
def download_report_pdf_endpoint(report_id: str):
    rep = db.get_report(report_id)
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found.")

    pdf_bytes = generate_patient_report_pdf(rep)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=Report_{report_id}.pdf"
        }
    )


# ---------------------------------------------------------------------------
# Doctor Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/login")
def login_endpoint(login_in: DoctorLogin):
    email = login_in.email.strip()
    password = login_in.password

    doc = db.find_doctor_by_email(email)
    if not doc or doc["password"] != password:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    safe_doc = {
        "fullName": doc["full_name"],
        "regNumber": doc["reg_number"],
        "email": doc["email"],
        "phone": doc["phone"],
        "specialization": doc["specialization"],
        "hospital": doc["hospital"],
    }
    return {"status": "success", "doctor": safe_doc}


@app.post("/auth/signup")
def signup_endpoint(signup_in: DoctorSignup):
    body = signup_in.model_dump()
    email = body["email"].strip()

    if db.find_doctor_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    new_doc = db.create_doctor(body)
    return {"status": "success", "message": "Doctor account created."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
