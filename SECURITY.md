# 🛡️ Security & Responsible AI Policy

## 📌 Project Identity & Scope

This policy governs **Drivox** (`Drivox`), maintained by **Aditi Ghosh** ([aditighoshagd@gmail.com](mailto:aditighoshagd@gmail.com)).

| Parameter | Specification |
| :--- | :--- |
| **Project** | Drivox: Edge-Native Driver Fatigue & Safety Intelligence System |
| **Maintainer** | Aditi Ghosh (NIT Jamshedpur) |
| **Status** | Active Enterprise Production Architecture |
| **Security Response SLA** | < 48 hours |

---

## 🔒 Responsible AI & Driver Data Privacy Architecture

In accordance with enterprise AI governance, GDPR, and India's Digital Personal Data Protection (DPDP) Act 2023, Drivox enforces strict privacy-by-design principles:

1. **Zero Raw Video Cloud Transmission**:
   - All facial landmark detection (MediaPipe 468 landmarks) and deep convolutional inference execute strictly on-device at the edge (in-vehicle embedded compute / mobile gateway).
   - Video frames are processed in ephemeral volatile memory (RAM) and immediately discarded. No video streams or face crops are persisted to disk or streamed to external cloud servers.

2. **Privacy-Preserving Telemetry**:
   - The relational DBMS (SQLite / SQL) logs **only derived scalar mathematical telemetry**:
     - Timestamp, Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Alert State (`ADVISORY`, `WARNING`, `CRITICAL`), and Latency metrics.
   - No personally identifiable facial geometry or biometric templates are stored or exposed.

3. **Responsible Escalation (Preventing Alert Fatigue)**:
   - Tiered escalation state-machine with built-in temporal hysteresis and audio cooldown intervals prevents acoustic startling or cognitive distraction while driving.

4. **Bias Mitigation & Illumination Robustness**:
   - Evaluated across diverse demographics, skin tones, head orientations, and lighting transitions (tunnel entry/exit, nighttime driving with CLAHE and non-linear Gamma correction).

---

## 📬 Reporting a Security Vulnerability

If you discover a security vulnerability or privacy concern, please do not open a public issue.

1. **Email Maintainer**: Send details to [aditighoshagd@gmail.com](mailto:aditighoshagd@gmail.com).
2. **Include**:
   - Detailed description of the potential vulnerability or vector
   - Reproduction steps or proof of concept
   - Proposed remediation or patch
3. **Response Timeline**: Acknowledgement within 24 hours, coordinated fix within 72 hours.
