"""
database.py
===========
SQLite persistence for Vireo Clinical Doctor Portal.
Provides thread-safe relational storage for Doctors, Patients, and Reports.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "doctor_portal.db"

_lock = threading.Lock()


def hash_password(password: str) -> str:
    """Hash a demo password with PBKDF2-HMAC-SHA256. Not production-grade auth."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return "pbkdf2_sha256$100000$" + salt.hex() + "$" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored PBKDF2 hash or legacy plaintext."""
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt_hex, digest_hex = stored.split("$", 3)
            digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations))
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (ValueError, TypeError):
            return False
    return password == stored


def get_db_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    with _lock:
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS doctors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT NOT NULL,
                        reg_number TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        phone TEXT NOT NULL,
                        specialization TEXT NOT NULL,
                        hospital TEXT NOT NULL,
                        password TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        patient_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        age INTEGER NOT NULL,
                        gender TEXT NOT NULL,
                        dob TEXT,
                        phone TEXT,
                        email TEXT,
                        address TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_id TEXT UNIQUE NOT NULL,
                        patient_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        efficientnet_score REAL NOT NULL,
                        resnet50_score REAL,
                        clinical_score REAL NOT NULL,
                        ensemble_score REAL,
                        weights_json TEXT,
                        threshold REAL,
                        prediction INTEGER,
                        prediction_label TEXT,
                        risk_level TEXT,
                        clinical_inputs_json TEXT,
                        image_filename TEXT,
                        doctor_notes TEXT,
                        pdf_path TEXT,
                        FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                    )
                """)

                # Non-destructive migration: if reports table has NOT NULL on resnet50_score or ensemble_score, migrate to allow NULL
                try:
                    cur_info = conn.execute("PRAGMA table_info(reports)").fetchall()
                    cols_notnull = {row["name"]: row["notnull"] for row in cur_info}
                    if cols_notnull.get("resnet50_score") == 1 or cols_notnull.get("ensemble_score") == 1:
                        conn.execute("ALTER TABLE reports RENAME TO _reports_legacy")
                        conn.execute("""
                            CREATE TABLE reports (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                report_id TEXT UNIQUE NOT NULL,
                                patient_id TEXT NOT NULL,
                                created_at TEXT NOT NULL,
                                efficientnet_score REAL NOT NULL,
                                resnet50_score REAL,
                                clinical_score REAL NOT NULL,
                                ensemble_score REAL,
                                weights_json TEXT,
                                threshold REAL,
                                prediction INTEGER,
                                prediction_label TEXT,
                                risk_level TEXT,
                                clinical_inputs_json TEXT,
                                image_filename TEXT,
                                doctor_notes TEXT,
                                pdf_path TEXT,
                                FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
                            )
                        """)
                        conn.execute("""
                            INSERT INTO reports (
                                id, report_id, patient_id, created_at, efficientnet_score, resnet50_score,
                                clinical_score, ensemble_score, weights_json, threshold, prediction,
                                prediction_label, risk_level, clinical_inputs_json, image_filename,
                                doctor_notes, pdf_path
                            )
                            SELECT
                                id, report_id, patient_id, created_at, efficientnet_score, resnet50_score,
                                clinical_score, ensemble_score, weights_json, threshold, prediction,
                                prediction_label, risk_level, clinical_inputs_json, image_filename,
                                doctor_notes, pdf_path
                            FROM _reports_legacy
                        """)
                        conn.execute("DROP TABLE _reports_legacy")
                except Exception:
                    pass

                # Seed initial demo doctor if empty
                cur = conn.execute("SELECT COUNT(*) AS c FROM doctors")
                if cur.fetchone()["c"] == 0:
                    conn.execute("""
                        INSERT INTO doctors (full_name, reg_number, email, phone, specialization, hospital, password, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        "Dr. Meera Kulkarni",
                        "MH-DERM-88214",
                        "meera.kulkarni@vireoclinic.in",
                        "+91 98220 11223",
                        "Dermatology & Oncology",
                        "Sunrise Multispecialty Hospital, Aurangabad",
                        hash_password("Demo@1234"),
                        datetime.now(timezone.utc).isoformat()
                    ))

                # Seed initial sample patients if empty
                cur_p = conn.execute("SELECT COUNT(*) AS c FROM patients")
                if cur_p.fetchone()["c"] == 0:
                    sample_patients = [
                        ("PT-10231", "Anil Deshmukh", 58, "Male", "1968-05-12", "+91 98220 54321", "anil.d@email.com", "Pune, Maharashtra", "2026-08-24T10:00:00Z"),
                        ("PT-10232", "Sunita Rao", 34, "Female", "1992-09-20", "+91 98221 65432", "sunita.r@email.com", "Nagpur, Maharashtra", "2026-08-25T11:30:00Z"),
                        ("PT-10233", "Fahad Khan", 45, "Male", "1981-03-15", "+91 98222 76543", "fahad.k@email.com", "Mumbai, Maharashtra", "2026-08-26T09:15:00Z"),
                        ("PT-10234", "Priya Nair", 29, "Female", "1997-11-04", "+91 98223 87654", "priya.n@email.com", "Nashik, Maharashtra", "2026-08-27T14:45:00Z"),
                        ("PT-10235", "Ramesh Patil", 66, "Male", "1960-01-30", "+91 98224 98765", "ramesh.p@email.com", "Kolhapur, Maharashtra", "2026-08-28T16:00:00Z"),
                        ("PT-10236", "Kavita Joshi", 51, "Female", "1975-07-22", "+91 98225 12345", "kavita.j@email.com", "Aurangabad, Maharashtra", "2026-08-29T11:00:00Z"),
                    ]
                    for p in sample_patients:
                        conn.execute("""
                            INSERT INTO patients (patient_id, name, age, gender, dob, phone, email, address, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[8]))

                    # Also seed initial baseline reports for demo patients so report history is populated
                    sample_reports = [
                        ("REP-20260824-001", "PT-10231", "2026-08-24T10:05:00Z", 0.78, 0.71, 0.85, 0.78, json.dumps({"efficientnet": 0.3333, "resnet50": 0.3333, "clinical_xgboost": 0.3333}), 0.50, 1, "Cancer", "high", json.dumps({"age": 58, "gender": "Male", "region": "FACE"}), "Melanoma pattern suspected on facial lesion."),
                        ("REP-20260825-002", "PT-10232", "2026-08-25T11:35:00Z", 0.12, 0.18, 0.22, 0.17, json.dumps({"efficientnet": 0.3333, "resnet50": 0.3333, "clinical_xgboost": 0.3333}), 0.50, 0, "Non-Cancer", "low", json.dumps({"age": 34, "gender": "Female", "region": "ARM"}), "Benign melanocytic nevus."),
                        ("REP-20260826-003", "PT-10233", "2026-08-26T09:20:00Z", 0.45, 0.48, 0.42, 0.45, json.dumps({"efficientnet": 0.3333, "resnet50": 0.3333, "clinical_xgboost": 0.3333}), 0.50, 0, "Non-Cancer", "moderate", json.dumps({"age": 45, "gender": "Male", "region": "BACK"}), "Atypical lesion; close monitoring recommended."),
                        ("REP-20260827-004", "PT-10234", "2026-08-27T14:50:00Z", 0.08, 0.11, 0.15, 0.11, json.dumps({"efficientnet": 0.3333, "resnet50": 0.3333, "clinical_xgboost": 0.3333}), 0.50, 0, "Non-Cancer", "low", json.dumps({"age": 29, "gender": "Female", "region": "CHEST"}), "Seborrheic keratosis pattern."),
                        ("REP-20260828-005", "PT-10235", "2026-08-28T16:05:00Z", 0.82, 0.79, 0.88, 0.83, json.dumps({"efficientnet": 0.3333, "resnet50": 0.3333, "clinical_xgboost": 0.3333}), 0.50, 1, "Cancer", "high", json.dumps({"age": 66, "gender": "Male", "region": "NOSE"}), "Basal cell carcinoma pattern suspected."),
                        ("REP-20260829-006", "PT-10236", "2026-08-29T11:05:00Z", 0.40, 0.52, 0.44, 0.45, json.dumps({"efficientnet": 0.3333, "resnet50": 0.3333, "clinical_xgboost": 0.3333}), 0.50, 0, "Non-Cancer", "moderate", json.dumps({"age": 51, "gender": "Female", "region": "SCALP"}), "Atypical lesion; reassessment advised.")
                    ]
                    for r in sample_reports:
                        conn.execute("""
                            INSERT INTO reports (
                                report_id, patient_id, created_at, efficientnet_score, resnet50_score,
                                clinical_score, ensemble_score, weights_json, threshold, prediction,
                                prediction_label, risk_level, clinical_inputs_json, doctor_notes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, r)
        finally:
            conn.close()


def find_doctor_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT * FROM doctors WHERE LOWER(email) = LOWER(?)", (email.strip(),))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_doctor(doc: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        conn = get_db_connection()
        try:
            with conn:
                conn.execute("""
                    INSERT INTO doctors (full_name, reg_number, email, phone, specialization, hospital, password, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc["fullName"].strip(),
                    doc["regNumber"].strip(),
                    doc["email"].strip().lower(),
                    doc["phone"].strip(),
                    doc["specialization"].strip(),
                    doc["hospital"].strip(),
                    doc["password"],
                    datetime.now(timezone.utc).isoformat()
                ))
            return find_doctor_by_email(doc["email"])
        finally:
            conn.close()


def list_patients(search: Optional[str] = None, risk: Optional[str] = None, date: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        query = """
            SELECT p.*,
                   (SELECT r.risk_level FROM reports r WHERE r.patient_id = p.patient_id ORDER BY r.created_at DESC LIMIT 1) as latest_risk,
                   (SELECT r.prediction_label FROM reports r WHERE r.patient_id = p.patient_id ORDER BY r.created_at DESC LIMIT 1) as latest_prediction,
                   (SELECT r.created_at FROM reports r WHERE r.patient_id = p.patient_id ORDER BY r.created_at DESC LIMIT 1) as latest_report_date
            FROM patients p
            WHERE 1=1
        """
        params = []
        if search:
            query += " AND (LOWER(p.name) LIKE ? OR LOWER(p.patient_id) LIKE ?)"
            s = f"%{search.strip().lower()}%"
            params.extend([s, s])

        query += " ORDER BY p.created_at DESC"
        cur = conn.execute(query, params)
        rows = cur.fetchall()

        results = []
        for r in rows:
            d = dict(r)
            # Apply risk filter in python to match latest risk
            if risk and risk != "all":
                r_val = d.get("latest_risk") or "low"
                if r_val.lower() != risk.lower():
                    continue
            if date:
                # filter by date part of latest report date or created_at
                d_val = (d.get("latest_report_date") or d.get("created_at") or "")[:10]
                if d_val != date:
                    continue
            results.append(d)
        return results
    finally:
        conn.close()


def get_patient(patient_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cur = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def create_patient(p: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        conn = get_db_connection()
        try:
            pid = p.get("patient_id")
            if not pid:
                # Generate new stable patient ID
                import random
                pid = f"PT-{random.randint(10000, 99999)}"
                # Ensure unique
                while conn.execute("SELECT 1 FROM patients WHERE patient_id = ?", (pid,)).fetchone():
                    pid = f"PT-{random.randint(10000, 99999)}"

            now = datetime.now(timezone.utc).isoformat()
            with conn:
                conn.execute("""
                    INSERT INTO patients (patient_id, name, age, gender, dob, phone, email, address, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pid,
                    p["name"].strip(),
                    int(p["age"]),
                    p["gender"].strip(),
                    p.get("dob", ""),
                    str(p.get("phone") or "").strip(),
                    str(p.get("email") or "").strip(),
                    str(p.get("address") or "").strip(),
                    now,
                    now
                ))
            return get_patient(pid)
        finally:
            conn.close()


def update_patient(patient_id: str, p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = get_db_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            with conn:
                conn.execute("""
                    UPDATE patients
                    SET name = ?, age = ?, gender = ?, dob = ?, phone = ?, email = ?, address = ?, updated_at = ?
                    WHERE patient_id = ?
                """, (
                    p.get("name", "").strip(),
                    int(p.get("age", 0)),
                    p.get("gender", "").strip(),
                    p.get("dob", ""),
                    str(p.get("phone") or "").strip(),
                    str(p.get("email") or "").strip(),
                    str(p.get("address") or "").strip(),
                    now,
                    patient_id
                ))
            return get_patient(patient_id)
        finally:
            conn.close()


def list_reports(patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        if patient_id:
            cur = conn.execute("""
                SELECT r.*, p.name as patient_name, p.age as patient_age, p.gender as patient_gender
                FROM reports r
                JOIN patients p ON r.patient_id = p.patient_id
                WHERE r.patient_id = ?
                ORDER BY r.created_at DESC
            """, (patient_id,))
        else:
            cur = conn.execute("""
                SELECT r.*, p.name as patient_name, p.age as patient_age, p.gender as patient_gender
                FROM reports r
                JOIN patients p ON r.patient_id = p.patient_id
                ORDER BY r.created_at DESC
            """)
        rows = [dict(row) for row in cur.fetchall()]
        for d in rows:
            raw_cli = d.get("clinical_inputs_json")
            if raw_cli and isinstance(raw_cli, str):
                try:
                    d["clinical_inputs"] = json.loads(raw_cli)
                except Exception:
                    d["clinical_inputs"] = {}
            elif isinstance(raw_cli, dict):
                d["clinical_inputs"] = raw_cli
            else:
                d["clinical_inputs"] = {}
        return rows
    finally:
        conn.close()


def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cur = conn.execute("""
            SELECT r.*, p.name as patient_name, p.age as patient_age, p.gender as patient_gender,
                   p.phone as patient_phone, p.email as patient_email, p.dob as patient_dob, p.address as patient_address
            FROM reports r
            JOIN patients p ON r.patient_id = p.patient_id
            WHERE r.report_id = ?
        """, (report_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        raw_cli = d.get("clinical_inputs_json")
        if raw_cli and isinstance(raw_cli, str):
            try:
                d["clinical_inputs"] = json.loads(raw_cli)
            except Exception:
                d["clinical_inputs"] = {}
        elif isinstance(raw_cli, dict):
            d["clinical_inputs"] = raw_cli
        else:
            d["clinical_inputs"] = {}
        return d
    finally:
        conn.close()


def save_report(r: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        conn = get_db_connection()
        try:
            rep_id = r.get("report_id")
            if not rep_id:
                import uuid
                today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
                short_uuid = uuid.uuid4().hex[:6].upper()
                rep_id = f"REP-{today_str}-{short_uuid}"

            created_at = r.get("created_at") or datetime.now(timezone.utc).isoformat()
            weights = r.get("weights")
            weights_json = json.dumps(weights) if isinstance(weights, dict) else (str(weights) if weights else None)

            clinical_inputs = r.get("clinical_inputs", {})
            clinical_inputs_json = json.dumps(clinical_inputs) if isinstance(clinical_inputs, dict) else str(clinical_inputs or "{}")

            resnet50_score = r.get("resnet50_score")
            resnet50_val = float(resnet50_score) if resnet50_score is not None else None

            ensemble_score = r.get("ensemble_score")
            ensemble_val = float(ensemble_score) if ensemble_score is not None else None

            threshold = r.get("threshold")
            threshold_val = float(threshold) if threshold is not None else None

            prediction = r.get("prediction")
            prediction_val = int(prediction) if prediction is not None else None

            prediction_label = r.get("prediction_label")
            risk_level = r.get("risk_level")

            with conn:
                conn.execute("""
                    INSERT INTO reports (
                        report_id, patient_id, created_at, efficientnet_score, resnet50_score,
                        clinical_score, ensemble_score, weights_json, threshold, prediction,
                        prediction_label, risk_level, clinical_inputs_json, image_filename,
                        doctor_notes, pdf_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rep_id,
                    r["patient_id"],
                    created_at,
                    float(r["efficientnet_score"]),
                    resnet50_val,
                    float(r["clinical_score"]),
                    ensemble_val,
                    weights_json,
                    threshold_val,
                    prediction_val,
                    prediction_label,
                    risk_level,
                    clinical_inputs_json,
                    r.get("image_filename"),
                    r.get("doctor_notes", ""),
                    r.get("pdf_path")
                ))
            return get_report(rep_id)
        finally:
            conn.close()


def get_dashboard_stats() -> Dict[str, int]:
    conn = get_db_connection()
    try:
        today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        total_patients = conn.execute("SELECT COUNT(*) AS c FROM patients").fetchone()["c"]
        today_patients = conn.execute("SELECT COUNT(*) AS c FROM patients WHERE substr(created_at, 1, 10) = ?", (today_date,)).fetchone()["c"]
        total_predictions = conn.execute("SELECT COUNT(*) AS c FROM reports").fetchone()["c"]
        high_risk_cases = conn.execute(
            "SELECT COUNT(*) AS c FROM reports WHERE risk_level = 'high' OR efficientnet_score >= 0.5 OR clinical_score >= 0.5"
        ).fetchone()["c"]
        return {
            "total_patients": total_patients,
            "today_patients": today_patients,
            "total_predictions": total_predictions,
            "high_risk_cases": high_risk_cases
        }
    finally:
        conn.close()

