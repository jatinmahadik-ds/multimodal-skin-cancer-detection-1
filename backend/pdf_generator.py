"""
pdf_generator.py
================
Pure-Python, zero-dependency PDF 1.4 generator for Vireo Clinical reports.
Produces 100% standard-compliant PDF files readable in all standard viewers (Adobe, Chrome, Edge).

Updated for Dual-Model Pipeline:
  - New reports display independent EfficientNetB0 and Clinical XGBoost assessments.
  - No ResNet50 rows or unvalidated ensemble scores are generated for new assessments.
  - Historical reports containing legacy ResNet50 scores are supported and rendered
    gracefully with an explicit archival notice.
================
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


class SimplePDFBuilder:
    """A lightweight, zero-dependency PDF 1.4 writer."""

    def __init__(self, page_width: float = 595.28, page_height: float = 841.89):
        # A4 paper size in points (72 points/inch)
        self.width = page_width
        self.height = page_height
        self.objects: List[bytes] = []
        self.stream_ops: List[str] = []

    def _add_object(self, content: bytes) -> int:
        self.objects.append(content)
        return len(self.objects)

    def draw_rect(self, x: float, y: float, w: float, h: float, fill_rgb: Optional[tuple] = None, stroke_rgb: Optional[tuple] = None, line_width: float = 1.0):
        ops = []
        ops.append(f"{line_width:.2f} w")
        if fill_rgb:
            ops.append(f"{fill_rgb[0]:.3f} {fill_rgb[1]:.3f} {fill_rgb[2]:.3f} rg")
        if stroke_rgb:
            ops.append(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")

        ops.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
        if fill_rgb and stroke_rgb:
            ops.append("B")
        elif fill_rgb:
            ops.append("f")
        elif stroke_rgb:
            ops.append("S")
        self.stream_ops.append("\n".join(ops))

    def draw_line(self, x1: float, y1: float, x2: float, y2: float, stroke_rgb: tuple = (0.8, 0.8, 0.8), line_width: float = 1.0):
        self.stream_ops.append(
            f"{line_width:.2f} w\n"
            f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG\n"
            f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def draw_text(self, text: str, x: float, y: float, font: str = "F1", size: float = 10.0, rgb: tuple = (0.1, 0.1, 0.1)):
        clean_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.stream_ops.append(
            f"BT\n"
            f"/{font} {size:.2f} Tf\n"
            f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg\n"
            f"{x:.2f} {y:.2f} Td\n"
            f"({clean_text}) Tj\n"
            f"ET"
        )

    def build_bytes(self) -> bytes:
        content_stream = "\n".join(self.stream_ops).encode("latin-1", errors="replace")
        stream_len = len(content_stream)

        # Object 1: Content stream
        obj1 = f"<< /Length {stream_len} >>\nstream\n".encode("latin-1") + content_stream + b"\nendstream"
        # Object 2: Font Helvetica
        obj2 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        # Object 3: Font Helvetica-Bold
        obj3 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        # Object 4: Resources
        obj4 = b"<< /ProcSet [/PDF /Text /ImageC] /Font << /F1 2 0 R /F2 3 0 R >> >>"
        # Object 5: Page
        obj5 = (
            f"<< /Type /Page /Parent 6 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
            f"/Contents 1 0 R /Resources 4 0 R >>"
        ).encode("latin-1")
        # Object 6: Pages
        obj6 = b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>"
        # Object 7: Catalog
        obj7 = b"<< /Type /Catalog /Pages 6 0 R >>"

        objs = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

        buf = bytearray()
        buf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets = [0]
        for i, obj_data in enumerate(objs, start=1):
            offsets.append(len(buf))
            buf.extend(f"{i} 0 obj\n".encode("latin-1"))
            buf.extend(obj_data)
            buf.extend(b"\nendobj\n")

        xref_offset = len(buf)
        buf.extend(f"xref\n0 {len(objs) + 1}\n".encode("latin-1"))
        buf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

        buf.extend(
            f"trailer\n<< /Size {len(objs) + 1} /Root 7 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
        )
        return bytes(buf)


def generate_patient_report_pdf(report: Dict[str, Any]) -> bytes:
    """Generate a clean, high-density clinical report PDF for a patient assessment.
    
    Automatically differentiates between:
      1. New Dual-Model Reports (EfficientNetB0 + Clinical XGBoost, uncombined)
      2. Legacy Historical Reports (gracefully renders archived 3-model records)
    """
    pdf = SimplePDFBuilder()

    # Determine if this is a legacy historical report with ResNet50
    has_legacy_res = (
        report.get("resnet50_score") is not None
        and float(report.get("resnet50_score", 0.0)) > 0.0
        and report.get("ensemble_score") is not None
    )

    # Top Header banner (Teal brand color #0F766E)
    pdf.draw_rect(0, 770, 595.28, 71.89, fill_rgb=(0.059, 0.463, 0.431))
    pdf.draw_text("VIREO CLINICAL", 36, 808, font="F2", size=18, rgb=(1.0, 1.0, 1.0))
    subtitle = "HISTORICAL MULTIMODAL SKIN CANCER REPORT" if has_legacy_res else "INDEPENDENT DUAL-MODEL SKIN CANCER ASSESSMENT"
    pdf.draw_text(subtitle, 36, 792, font="F1", size=9, rgb=(0.85, 0.95, 0.93))
    pdf.draw_text("CONFIDENTIAL CLINICAL ASSESSMENT REPORT", 36, 778, font="F2", size=8, rgb=(0.95, 0.85, 0.40))

    rep_id = str(report.get("report_id", "REP-UNKNOWN"))
    created_at = str(report.get("created_at", datetime.now().isoformat()))[:19].replace("T", " ")
    pdf.draw_text(f"Report ID: {rep_id}", 380, 808, font="F2", size=9, rgb=(1.0, 1.0, 1.0))
    pdf.draw_text(f"Generated: {created_at} UTC", 380, 792, font="F1", size=8, rgb=(0.85, 0.95, 0.93))
    pdf.draw_text("Research Prototype - Not A Final Diagnosis", 380, 778, font="F1", size=8, rgb=(1.0, 0.8, 0.8))

    # Patient Demographic Card
    pdf.draw_rect(36, 680, 523.28, 75, fill_rgb=(0.97, 0.98, 0.99), stroke_rgb=(0.82, 0.86, 0.90), line_width=1.0)
    pdf.draw_text("PATIENT INFORMATION", 48, 738, font="F2", size=10, rgb=(0.1, 0.3, 0.35))
    pdf.draw_line(48, 732, 545, 732, stroke_rgb=(0.82, 0.86, 0.90), line_width=0.75)

    pat_name = str(report.get("patient_name") or report.get("name") or "Unknown Patient")
    pat_id = str(report.get("patient_id", "PT-UNKNOWN"))
    pat_age = str(report.get("patient_age") or report.get("age") or "N/A")
    pat_gender = str(report.get("patient_gender") or report.get("gender") or "N/A")
    pat_phone = str(report.get("patient_phone") or report.get("phone") or "N/A")
    pat_email = str(report.get("patient_email") or report.get("email") or "N/A")

    pdf.draw_text(f"Patient Name:  {pat_name}", 48, 715, font="F2", size=9, rgb=(0.15, 0.15, 0.15))
    pdf.draw_text(f"Patient ID:  {pat_id}", 48, 698, font="F1", size=9, rgb=(0.25, 0.25, 0.25))
    pdf.draw_text(f"Age / Gender:  {pat_age} yrs / {pat_gender}", 240, 715, font="F1", size=9, rgb=(0.25, 0.25, 0.25))
    pdf.draw_text(f"Contact Phone:  {pat_phone}", 240, 698, font="F1", size=9, rgb=(0.25, 0.25, 0.25))
    pdf.draw_text(f"Email:  {pat_email}", 410, 715, font="F1", size=9, rgb=(0.25, 0.25, 0.25))

    eff_score = float(report.get("efficientnet_score", 0.0))
    cli_score = float(report.get("clinical_score", 0.0))

    if not has_legacy_res:
        # =====================================================================
        # DUAL-MODEL ASSESSMENT (NEW REPORTS: INDEPENDENT, NO ENSEMBLE)
        # =====================================================================
        eff_high = eff_score >= 0.50
        cli_high = cli_score >= 0.50

        # Choose primary card theme based on individual model flags
        if eff_high and cli_high:
            banner_fill = (0.99, 0.92, 0.92)  # Soft red
            banner_stroke = (0.85, 0.20, 0.20)
            text_color = (0.75, 0.10, 0.10)
            overall_status = "ELEVATED RISK INDICATED ACROSS IMAGE AND CLINICAL ASSESSMENTS"
        elif eff_high or cli_high:
            banner_fill = (1.00, 0.96, 0.88)  # Soft amber
            banner_stroke = (0.90, 0.60, 0.10)
            text_color = (0.75, 0.45, 0.05)
            overall_status = "DISCORDANT EVALUATION: ONE MODEL INDICATES ELEVATED RISK"
        else:
            banner_fill = (0.91, 0.97, 0.94)  # Soft green
            banner_stroke = (0.15, 0.60, 0.35)
            text_color = (0.10, 0.45, 0.25)
            overall_status = "LOW RISK INDICATED ACROSS BOTH EVALUATIONS"

        pdf.draw_rect(36, 560, 523.28, 105, fill_rgb=banner_fill, stroke_rgb=banner_stroke, line_width=1.5)
        pdf.draw_text("INDEPENDENT DUAL-MODEL ASSESSMENT OVERVIEW", 48, 646, font="F2", size=9.5, rgb=text_color)
        pdf.draw_text(overall_status, 48, 626, font="F2", size=11, rgb=text_color)

        pdf.draw_text(f"Image Model (EfficientNetB0):  {eff_score * 100:.1f}% Cancer Score  [{'Cancer Suspected' if eff_high else 'Benign Pattern'}]", 48, 602, font="F2", size=9.5, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text(f"Clinical Model (XGBoost):      {cli_score * 100:.1f}% Cancer Score  [{'High-Risk Profile' if cli_high else 'Low-Risk Profile'}]", 48, 586, font="F2", size=9.5, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text("Notice: Evaluated as separate models. No combined ensemble score is generated pending paired validation.", 48, 570, font="F1", size=8, rgb=(0.35, 0.35, 0.35))

        # Dual-Model Table
        pdf.draw_rect(36, 400, 523.28, 145, fill_rgb=(1.0, 1.0, 1.0), stroke_rgb=(0.82, 0.86, 0.90), line_width=1.0)
        pdf.draw_text("INDEPENDENT MODEL OUTPUTS", 48, 526, font="F2", size=10, rgb=(0.1, 0.3, 0.35))
        pdf.draw_line(48, 518, 545, 518, stroke_rgb=(0.85, 0.88, 0.92), line_width=0.75)

        # Table Header
        pdf.draw_rect(48, 492, 497.28, 22, fill_rgb=(0.94, 0.96, 0.98))
        pdf.draw_text("Model Name", 56, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))
        pdf.draw_text("Modality & Dataset", 185, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))
        pdf.draw_text("Cancer Score", 345, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))
        pdf.draw_text("Pattern Assessment", 435, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))

        # Row 1: EfficientNetB0
        pdf.draw_line(48, 490, 545, 490, stroke_rgb=(0.90, 0.92, 0.94), line_width=0.5)
        pdf.draw_text("EfficientNetB0 (CNN)", 56, 473, font="F2", size=9, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text("ISIC 2019 (Image 224x224 RGB)", 185, 473, font="F1", size=8.5, rgb=(0.3, 0.3, 0.3))
        pdf.draw_text(f"{eff_score * 100:.2f}%", 345, 473, font="F2", size=9, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text("Cancer Suspected" if eff_high else "Benign Pattern", 435, 473, font="F2", size=8.5, rgb=(0.75, 0.1, 0.1) if eff_high else (0.1, 0.5, 0.25))

        # Row 2: Clinical XGBoost
        pdf.draw_line(48, 460, 545, 460, stroke_rgb=(0.90, 0.92, 0.94), line_width=0.5)
        pdf.draw_text("Clinical-Only XGBoost", 56, 443, font="F2", size=9, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text("PAD-UFES-20 (53 encoded dummy features)", 185, 443, font="F1", size=8.5, rgb=(0.3, 0.3, 0.3))
        pdf.draw_text(f"{cli_score * 100:.2f}%", 345, 443, font="F2", size=9, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text("High-Risk Profile" if cli_high else "Low-Risk Profile", 435, 443, font="F2", size=8.5, rgb=(0.75, 0.1, 0.1) if cli_high else (0.1, 0.5, 0.25))

        # Explanatory Note box
        pdf.draw_line(48, 430, 545, 430, stroke_rgb=(0.85, 0.88, 0.92), line_width=0.75)
        pdf.draw_text("Ensemble Status: No 2-model ensemble weights have been statistically validated.", 56, 415, font="F2", size=8, rgb=(0.2, 0.3, 0.4))
        pdf.draw_text("Image and clinical models operate independently without weight-mixing or output coupling.", 56, 405, font="F1", size=7.5, rgb=(0.4, 0.4, 0.4))

    else:
        # =====================================================================
        # HISTORICAL REPORT COMPATIBILITY (LEGACY 3-MODEL RECORD)
        # =====================================================================
        res_score = float(report.get("resnet50_score", 0.0))
        ens_score = float(report.get("ensemble_score", 0.0))
        threshold = float(report.get("threshold", 0.50))
        pred_label = str(report.get("prediction_label", "Unknown")).upper()
        risk_level = str(report.get("risk_level", "low")).upper()

        banner_fill = (0.97, 0.97, 0.99)
        banner_stroke = (0.70, 0.75, 0.85)
        text_color = (0.25, 0.30, 0.45)

        pdf.draw_rect(36, 560, 523.28, 105, fill_rgb=banner_fill, stroke_rgb=banner_stroke, line_width=1.5)
        pdf.draw_text("HISTORICAL 3-MODEL ARCHIVED CLASSIFICATION", 48, 646, font="F2", size=10, rgb=text_color)
        pdf.draw_text(f"{pred_label}  ({risk_level} RISK - ARCHIVED)", 48, 618, font="F2", size=18, rgb=text_color)

        pdf.draw_text(f"Archived Ensemble Score: {ens_score * 100:.1f}%  |  Threshold: {threshold * 100:.1f}%", 48, 595, font="F2", size=10, rgb=(0.15, 0.15, 0.15))
        pdf.draw_text("Archived Record: Contains historical ResNet50 score recorded prior to pipeline modernization.", 48, 575, font="F1", size=8, rgb=(0.4, 0.4, 0.4))

        # 3-Model Table
        pdf.draw_rect(36, 400, 523.28, 145, fill_rgb=(1.0, 1.0, 1.0), stroke_rgb=(0.82, 0.86, 0.90), line_width=1.0)
        pdf.draw_text("ARCHIVED MODEL CONTRIBUTIONS (HISTORICAL)", 48, 526, font="F2", size=10, rgb=(0.1, 0.3, 0.35))
        pdf.draw_line(48, 518, 545, 518, stroke_rgb=(0.85, 0.88, 0.92), line_width=0.75)

        pdf.draw_rect(48, 492, 497.28, 22, fill_rgb=(0.94, 0.96, 0.98))
        pdf.draw_text("Model Name", 56, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))
        pdf.draw_text("Dataset / Architecture", 185, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))
        pdf.draw_text("Cancer Score", 345, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))
        pdf.draw_text("Archived Weight", 440, 499, font="F2", size=8.5, rgb=(0.2, 0.25, 0.3))

        pdf.draw_line(48, 490, 545, 490, stroke_rgb=(0.90, 0.92, 0.94), line_width=0.5)
        pdf.draw_text("EfficientNetB0 (CNN)", 56, 474, font="F2", size=8.5)
        pdf.draw_text("ISIC 2019 (Image)", 185, 474, font="F1", size=8.5)
        pdf.draw_text(f"{eff_score * 100:.2f}%", 345, 474, font="F2", size=8.5)
        pdf.draw_text("0.3333", 440, 474, font="F1", size=8.5)

        pdf.draw_line(48, 464, 545, 464, stroke_rgb=(0.90, 0.92, 0.94), line_width=0.5)
        pdf.draw_text("ResNet50 (Weighted CNN)", 56, 448, font="F2", size=8.5)
        pdf.draw_text("HAM10000 (Image)", 185, 448, font="F1", size=8.5)
        pdf.draw_text(f"{res_score * 100:.2f}%", 345, 448, font="F2", size=8.5)
        pdf.draw_text("0.3333", 440, 448, font="F1", size=8.5)

        pdf.draw_line(48, 438, 545, 438, stroke_rgb=(0.90, 0.92, 0.94), line_width=0.5)
        pdf.draw_text("Clinical-Only XGBoost", 56, 422, font="F2", size=8.5)
        pdf.draw_text("PAD-UFES-20 (53 features)", 185, 422, font="F1", size=8.5)
        pdf.draw_text(f"{cli_score * 100:.2f}%", 345, 422, font="F2", size=8.5)
        pdf.draw_text("0.3333", 440, 422, font="F1", size=8.5)

        pdf.draw_line(48, 412, 545, 412, stroke_rgb=(0.85, 0.88, 0.92), line_width=0.75)
        pdf.draw_text(f"Archived Ensemble Average: {ens_score * 100:.2f}%", 56, 404, font="F2", size=8, rgb=(0.1, 0.3, 0.4))

    # =========================================================================
    # Clinical Features Input Summary
    # =========================================================================
    pdf.draw_rect(36, 215, 523.28, 170, fill_rgb=(0.98, 0.99, 1.0), stroke_rgb=(0.82, 0.86, 0.90), line_width=1.0)
    pdf.draw_text("RECORDED CLINICAL FEATURES & SYMPTOMS", 48, 368, font="F2", size=10, rgb=(0.1, 0.3, 0.35))
    pdf.draw_line(48, 362, 545, 362, stroke_rgb=(0.85, 0.88, 0.92), line_width=0.75)

    cli_inputs = report.get("clinical_inputs")
    if isinstance(cli_inputs, str):
        try:
            cli_inputs = json.loads(cli_inputs)
        except Exception:
            cli_inputs = {}
    elif not isinstance(cli_inputs, dict):
        cli_inputs = {}

    region = str(cli_inputs.get("region") or cli_inputs.get("lesionLocation") or "Not provided")
    diam1 = str(cli_inputs.get("diameter_1", "N/A"))
    diam2 = str(cli_inputs.get("diameter_2", "N/A"))
    fitz = str(cli_inputs.get("fitspatrick", "N/A"))
    changed = str(cli_inputs.get("changed", "N/A"))
    itch = str(cli_inputs.get("itch", "N/A"))
    bleed = str(cli_inputs.get("bleed", "N/A"))
    hurt = str(cli_inputs.get("hurt", "N/A"))
    grew = str(cli_inputs.get("grew", "N/A"))
    elevation = str(cli_inputs.get("elevation", "N/A"))
    smoke = str(cli_inputs.get("smoke", "N/A"))
    drink = str(cli_inputs.get("drink", "N/A"))
    pesticide = str(cli_inputs.get("pesticide", "N/A"))
    skin_hist = str(cli_inputs.get("skin_cancer_history", "N/A"))
    cancer_hist = str(cli_inputs.get("cancer_history", "N/A"))

    pdf.draw_text(f"Lesion Region:  {region}", 48, 346, font="F1", size=8.5)
    pdf.draw_text(f"Diameters:  {diam1} mm x {diam2} mm", 220, 346, font="F1", size=8.5)
    pdf.draw_text(f"Fitzpatrick Skin Type:  {fitz}", 400, 346, font="F1", size=8.5)

    pdf.draw_text(f"Changed Recently:  {changed}", 48, 328, font="F1", size=8.5)
    pdf.draw_text(f"Itching:  {itch}", 220, 328, font="F1", size=8.5)
    pdf.draw_text(f"Bleeding:  {bleed}", 400, 328, font="F1", size=8.5)

    pdf.draw_text(f"Lesion Grew:  {grew}", 48, 310, font="F1", size=8.5)
    pdf.draw_text(f"Pain:  {hurt}", 220, 310, font="F1", size=8.5)
    pdf.draw_text(f"Elevation:  {elevation}", 400, 310, font="F1", size=8.5)

    pdf.draw_text(f"Smoking:  {smoke}", 48, 292, font="F1", size=8.5)
    pdf.draw_text(f"Alcohol:  {drink}", 220, 292, font="F1", size=8.5)
    pdf.draw_text(f"Pesticide Exposure:  {pesticide}", 400, 292, font="F1", size=8.5)

    pdf.draw_text(f"Personal Skin Cancer History:  {skin_hist}", 48, 274, font="F1", size=8.5)
    pdf.draw_text(f"Other Cancer History:  {cancer_hist}", 300, 274, font="F1", size=8.5)

    notes = str(report.get("doctor_notes") or cli_inputs.get("notes") or "None recorded.")
    if len(notes) > 90:
        notes = notes[:87] + "..."
    pdf.draw_text(f"Doctor / Clinical Observations:  {notes}", 48, 250, font="F2", size=8.5, rgb=(0.1, 0.2, 0.3))
    pdf.draw_text("Pre-processing: 21 clinical fields aligned with feature_schema.json into 53 XGBoost dummy inputs (no CNN leaks).", 48, 230, font="F1", size=7.5, rgb=(0.4, 0.4, 0.4))

    # Medical Disclaimer & Limitations Box
    pdf.draw_rect(36, 95, 523.28, 105, fill_rgb=(0.99, 0.99, 0.99), stroke_rgb=(0.80, 0.80, 0.80), line_width=0.75)
    pdf.draw_text("IMPORTANT MEDICAL & SCIENTIFIC DISCLAIMER", 48, 185, font="F2", size=8.5, rgb=(0.80, 0.20, 0.20))
    disclaimer_lines = [
        "1. Experimental Prototype: This software is an engineering and scientific research project developed for academic evaluation.",
        "   It has not been approved as a certified medical software device (SaMD) by CDSCO, US FDA, or EMA.",
        "2. Non-Calibrated Output: Output scores represent discriminative mathematical model outputs, not calibrated clinical probabilities.",
        "3. Clinical Responsibility: This report must NEVER replace a professional dermatological biopsy, histopathological examination,",
        "   or specialist physician consultation. Clinical decisions remain the sole professional responsibility of the treating physician.",
        "4. Confidentiality: This report contains protected health observations. Unauthorized copying, distribution, or storage is prohibited."
    ]
    curr_y = 172
    for line in disclaimer_lines:
        pdf.draw_text(line, 48, curr_y, font="F1", size=7.2, rgb=(0.35, 0.35, 0.35))
        curr_y -= 11

    # Footer
    pdf.draw_line(36, 75, 559.28, 75, stroke_rgb=(0.85, 0.85, 0.85), line_width=0.5)
    footer_text = (
        "Vireo Clinical - Archived Multimodal Report (EfficientNetB0 + ResNet50 + Clinical XGBoost)"
        if has_legacy_res else
        "Vireo Clinical - Independent Dual-Model Detection System (EfficientNetB0 + Clinical XGBoost)"
    )
    pdf.draw_text(footer_text, 36, 62, font="F1", size=7.5, rgb=(0.5, 0.5, 0.5))
    pdf.draw_text(f"Page 1 of 1   |   Report ID: {rep_id}", 430, 62, font="F1", size=7.5, rgb=(0.5, 0.5, 0.5))

    return pdf.build_bytes()
