# MediInsight AI — 

A Streamlit academic project for extracting text from PDF medical reports.

## Important safety and scope
- Educational/research prototype only; not a medical device or diagnostic system.
- It does not calculate disease-risk percentages.
- It does not provide treatment or medication instructions.
- Never test with identifiable patient data unless you have documented permission, consent, and an approved privacy/security process.
- The current parser is deliberately simple and only detects a narrow pattern of laboratory rows.
- Scanned PDFs need OCR; radiology images and ECG waveforms need specialized models and validation.

## 1. Install Python
Use Python 3.11 or 3.12. Confirm:
```bash
python --version
```
On some Windows installations use `py --version`.

## 2. Create and activate a virtual environment

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run locally
```bash
streamlit run app.py
```
The browser should open at `http://localhost:8501`.

## 5. GitHub
Create an empty repository named `medinsight-ai` on GitHub. In this project folder:
```bash
git init
git add .
git commit -m "Initial MediInsight AI prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/medinsight-ai.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username.

## 6. Deploy with Streamlit Community Cloud
1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Choose **Create app**.
4. Select repository `medinsight-ai`, branch `main`, and file `app.py`.
5. Deploy.

Do not upload real patient reports to a public demo. A public Streamlit deployment is not an appropriate place for confidential health records without a proper security, privacy, access-control, and compliance design.

## 7. Development roadmap
1. Improve PDF text extraction and add scanned-PDF OCR.
2. Build report-type-specific structured extraction schemas.
3. Add validation against source page/paragraph and human review.
4. Add evidence-grounded Bangla/English explanations using trusted medical references (RAG).
5. Add a report-grounded question-answer assistant.
6. Add historical comparison only when test, unit, method, and reference ranges are comparable.
7. Evaluate extraction accuracy and explanation safety with expert-reviewed test cases.

## Suggested repository structure as the project grows
```text
medinsight-ai/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── pdf_extractor.py
│   ├── report_classifier.py
│   ├── lab_parser.py
│   ├── explanation_engine.py
│   └── assistant.py
├── data/
│   └── sample_reports/   # synthetic/anonymized only
└── tests/
    └── test_extraction.py
```
