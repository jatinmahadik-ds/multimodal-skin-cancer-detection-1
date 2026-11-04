# Documentation

This folder contains the historical project documentation, the updated technical report, the architecture diagrams and real screenshots of the running portal.

## Contents

| File / folder | Description | Status |
|---|---|---|
| `project_report_final.pdf` | Updated technical report describing the implemented independent dual-model system. | **Current / code-accurate** |
| `project_report_final.docx` | Editable source for the updated report. | **Current / code-accurate** |
| `architecture_final.png` | Code-accurate architecture diagram: two independent branches, no fusion layer. | **Current / code-accurate** |
| `project_report.pdf` | PDF rendering of the original B.Tech project report. | **Historical / outdated in parts** |
| `project_report.docx` | Original report source used to generate the original PDF. | **Historical / outdated in parts** |
| `architecture.png` | Original Draw.io architecture export from `documnets/jijij.drawio.png`. | **Historical / outdated** |
| `screenshots/` | Real screenshots captured from the running portal on 4 November 2026; `01-login.png` was recaptured on 5 November 2026 after the UI copy correction. | Current UI |

## Important accuracy notes

### Updated report supersedes the historical report

`project_report_final.docx` and `project_report_final.pdf` (5 November 2026) describe the code that actually runs. They replace the historical report's architecture, model-inventory, training-methodology and evaluation claims. The updated report covers:

- EfficientNetB0 image classification and its verified held-out metrics.
- Clinical-only XGBoost with 53 encoded features and its verified held-out metrics.
- FastAPI integration, independent model outputs, persistence and PDF export.
- The explicit absence of a validated fusion model and the requirements for one.
- Current limitations, reproducibility steps and claim-to-source traceability.

The historical report and diagram are preserved unchanged for provenance. The updated report does not invent new experiments or metrics.

### Project report is historical

The original report was written on **26 October 2026** and describes an earlier three-model design:

- EfficientNetB0 (ISIC 2019)
- ResNet50 (HAM10000)
- XGBoost fusion using two CNN probabilities + 21 clinical features (23 inputs)

The current code in this repository does **not** implement that design. ResNet50 has been disconnected, and XGBoost is a **clinical-only** model with **53** encoded clinical features. No validated multimodal ensemble is computed at inference time. See the root `README.md` for the architecture that the code actually implements.

The report is retained for academic provenance only. Its ResNet50 and fusion statements must not be read as a description of the current implementation. It also reports integration checks, not held-out classification metrics.

### Architecture image is historical

`architecture.png` is the original exported diagram. It shows a ResNet50 image branch and a three-model fusion pipeline. It is retained because it is the actual diagram supplied with the project.

`architecture_final.png` is the current, code-accurate diagram. It shows two independent branches — EfficientNetB0 image inference and clinical-only XGBoost inference — feeding two separate outputs into a FastAPI response. It explicitly marks the fusion layer as not implemented. The root `README.md` also contains a Mermaid version of the same architecture.

### PDF conversion

Both PDFs were generated with a temporary HTML render and Microsoft Edge's print-to-PDF function. Word COM automation was unavailable on the build machine. The historical PDF comes directly from the original DOCX. The updated PDF is rendered from the same structured content used to generate `project_report_final.docx`, so the two updated files contain the same text, tables and figures. No report text, figures or results were fabricated.

### Screenshots

The screenshots in `screenshots/` were captured from the locally running frontend and FastAPI backend using the seeded **demo** account and **fictional demo patients** created by `backend/database.py`. No real patient data is present.

`01-login.png` was recaptured on 5 November 2026 because the previous capture showed the retired “3 AI Models Fused” / “multimodal fusion model” copy. The other five screenshots remain valid and already show the independent dual-model result view.

| Screenshot | Page |
|---|---|
| `01-login.png` | Clinician login |
| `02-dashboard.png` | Dashboard |
| `03-new-assessment.png` | New assessment / prediction form |
| `04-patient-directory.png` | Patient directory |
| `05-reports.png` | Reports archive |
| `06-results.png` | Dual-model assessment result |
