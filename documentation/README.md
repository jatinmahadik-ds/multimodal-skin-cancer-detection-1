# Documentation

This folder contains the original project documentation and screenshots that were available at the time of repository preparation.

## Contents

| File / folder | Description | Status |
|---|---|---|
| `project_report.pdf` | PDF rendering of the B.Tech project report. | **Historical / outdated in parts** |
| `project_report.docx` | Original report source used to generate the PDF. | **Historical / outdated in parts** |
| `architecture.png` | Original Draw.io architecture export from `documnets/jijij.drawio.png`. | **Historical / outdated** |
| `screenshots/` | Real screenshots captured from the running portal on 4 November 2026. | Current UI |

## Important accuracy notes

### Project report is historical

The report was written on **26 October 2026** and describes an earlier three-model design:

- EfficientNetB0 (ISIC 2019)
- ResNet50 (HAM10000)
- XGBoost fusion using two CNN probabilities + 21 clinical features (23 inputs)

The current code in this repository does **not** implement that design. ResNet50 has been disconnected, and XGBoost is a **clinical-only** model with **53** encoded clinical features. No validated multimodal ensemble is computed at inference time. See the root `README.md` for the architecture that the code actually implements.

The report is retained for academic provenance only. Its ResNet50 and fusion statements must not be read as a description of the current implementation. It also reports integration checks, not held-out classification metrics.

### Architecture image is historical

`architecture.png` is the original exported diagram. It shows a ResNet50 image branch and a three-model fusion pipeline. It is retained because it is the actual diagram supplied with the project. The current, code-accurate architecture is documented with a Mermaid diagram in the root `README.md`.

### PDF conversion

`project_report.pdf` was generated from `project_report.docx` using a temporary HTML render and Microsoft Edge's print-to-PDF function. Word COM automation was unavailable on the build machine. The content comes directly from the DOCX; no report text, figures, or results were fabricated.

### Screenshots

The screenshots in `screenshots/` were captured from the locally running frontend and FastAPI backend using the seeded **demo** account and **fictional demo patients** created by `backend/database.py`. No real patient data is present.

| Screenshot | Page |
|---|---|
| `01-login.png` | Clinician login |
| `02-dashboard.png` | Dashboard |
| `03-new-assessment.png` | New assessment / prediction form |
| `04-patient-directory.png` | Patient directory |
| `05-reports.png` | Reports archive |
| `06-results.png` | Dual-model assessment result |
