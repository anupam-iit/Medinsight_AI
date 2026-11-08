import re
import streamlit as st
import pymupdf
import fitz
import pytesseract

from PIL import Image
from io import BytesIO

st.set_page_config(page_title="MediInsight AI", page_icon="🩺", layout="wide")

st.title("🩺 MediInsight AI")
st.caption("Medical report understanding assistant — academic prototype")

st.warning(
    "Educational prototype only. It does not diagnose, calculate disease probability, "
    "or replace a qualified healthcare professional. Do not upload identifiable or "
    "confidential patient reports."
)

def extract_pdf_text(pdf_bytes: bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_no, page in enumerate(doc, start=1):
        pages.append(f"\n--- Page {page_no} ---\n{page.get_text('text')}")
    doc.close()
    return "\n".join(pages).strip()

def extract_pdf_content(pdf_file):

    doc = fitz.open(stream=pdf_file, filetype="pdf")

    extracted_pages = []

    for page_number, page in enumerate(doc, start=1):

        # Step 1: Extract selectable text
        text = page.get_text("text").strip()

        # Step 2: Check whether OCR is required
        if len(text) < 50:

            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )

            image = Image.open(
                BytesIO(pix.tobytes("png"))
            )

            # Step 3: Apply OCR
            text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

        extracted_pages.append({
            "page": page_number,
            "text": text.strip()
        })

    doc.close()

    return extracted_pages

def classify_report(text: str) -> str:
    t = text.lower()
    categories = {
        "Laboratory / Pathology": ["hemoglobin", "haemoglobin", "platelet", "wbc", "rbc", "glucose", "cholesterol"],
        "Radiology": ["ultrasound", "sonography", "x-ray", "xray", "ct scan", "mri", "impression", "radiologist"],
        "Cardiology": ["ecg", "ekg", "echocardiogram", "ejection fraction", "心"],
        "Histopathology": ["histopathology", "biopsy", "specimen", "microscopy", "malignant"],
    }
    scores = {name: sum(term in t for term in terms) for name, terms in categories.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General / Unclassified medical report"

def parse_simple_lab_rows(text: str):
    """
    Conservative prototype parser.
    Matches simple lines such as:
    Hemoglobin 13.5 g/dL 12.0 - 16.0
    Real-world reports vary widely; this is not a production parser.
    """
    rows = []
    pattern = re.compile(
        r"(?im)^\s*([A-Za-z][A-Za-z0-9 ()/%.-]{1,45}?)\s+"
        r"([-+]?\d+(?:\.\d+)?)\s*"
        r"([A-Za-zµμ/%^0-9.]+)?\s+"
        r"([-+]?\d+(?:\.\d+)?)\s*(?:-|to)\s*([-+]?\d+(?:\.\d+)?)\s*$"
    )
    for match in pattern.finditer(text):
        name, value, unit, low, high = match.groups()
        value, low, high = float(value), float(low), float(high)
        if low <= value <= high:
            status = "Within stated range"
        elif value < low:
            status = "Below stated range"
        else:
            status = "Above stated range"
        rows.append({
            "Parameter": name.strip(),
            "Result": value,
            "Unit": unit or "",
            "Reference range": f"{low:g} - {high:g}",
            "Prototype status": status
        })
    return rows

def explain_status_bangla(status: str):
    mapping = {
        "Within stated range": "এই মানটি রিপোর্টে দেওয়া রেফারেন্স সীমার মধ্যে আছে।",
        "Below stated range": "এই মানটি রিপোর্টে দেওয়া রেফারেন্স সীমার নিচে। চিকিৎসকের সঙ্গে আলোচনা করুন।",
        "Above stated range": "এই মানটি রিপোর্টে দেওয়া রেফারেন্স সীমার উপরে। চিকিৎসকের সঙ্গে আলোচনা করুন।",
    }
    return mapping.get(status, "এই ফলাফলটি চিকিৎসকের সঙ্গে আলোচনা করুন।")

uploaded = st.file_uploader("Upload a medical report (PDF)", type=["pdf"])

if uploaded:
    pdf_bytes = uploaded.getvalue()
    if len(pdf_bytes) > 15 * 1024 * 1024:
        st.error("For this prototype, please upload a PDF smaller than 15 MB.")
        st.stop()

    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_pdf_text(pdf_bytes)

    if not extracted_text or len(extracted_text) < 30:
        st.error(
            "Little or no selectable text was found. This may be a scanned PDF. "
            "OCR support will be added in the next development phase."
        )
        st.info("For now, test with a text-based PDF.")
        st.stop()

    report_type = classify_report(extracted_text)
    st.success(f"Document processed. Detected category: **{report_type}**")

    tab1, tab2, tab3 = st.tabs(["Overview", "Extracted text", "Structured findings"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.metric("Pages", extracted_text.count("--- Page"))
        c2.metric("Extracted characters", f"{len(extracted_text):,}")
        st.info(
            "This prototype identifies the document category using keywords. "
            "It does not yet interpret radiology images, ECG waveforms, or diagnose conditions."
        )

    with tab2:
        st.text_area("Extracted PDF text", extracted_text, height=420)

    with tab3:
        lab_rows = parse_simple_lab_rows(extracted_text)
        if lab_rows:
            st.subheader("Possible laboratory values")
            st.caption("Prototype extraction only. Verify every value against the original report.")
            st.dataframe(lab_rows, use_container_width=True, hide_index=True)
            for row in lab_rows:
                with st.container(border=True):
                    st.markdown(f"**{row['Parameter']} — {row['Result']} {row['Unit']}**")
                    status = row["Prototype status"]
                    if status == "Within stated range":
                        st.success(status)
                    else:
                        st.warning(status)
                    st.write(explain_status_bangla(status))
        else:
            st.info(
                "No simple laboratory rows were detected. This is expected for many "
                "radiology, cardiology, and narrative reports. A report-specific extraction "
                "module is needed for each document family."
            )

    st.download_button(
        "Download extracted text",
        data=extracted_text,
        file_name="extracted_report_text.txt",
        mime="text/plain"
    )
else:
    st.markdown("### Planned workflow")
    st.write("1. Upload PDF  →  2. Extract text  →  3. Structure findings  →  4. Explain in Bangla/English  →  5. Ask follow-up questions")
    st.info("Start with anonymized sample reports and text-based PDFs.")
