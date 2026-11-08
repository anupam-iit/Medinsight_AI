import re
from io import BytesIO

import streamlit as st
import pymupdf
import pytesseract

from PIL import Image, ImageOps, ImageEnhance


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MediInsight AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediInsight AI")

st.caption(
    "Medical Report Understanding Assistant — Academic Prototype"
)

st.warning(
    "Educational prototype only. It does not diagnose, calculate disease "
    "probability, or replace a qualified healthcare professional. "
    "Do not upload identifiable or confidential patient reports."
)


# --------------------------------------------------
# 2. PDF EXTRACTION WITH OCR
# --------------------------------------------------

def preprocess_image(image):
    """
    Basic image preprocessing to improve OCR readability.
    """

    image = ImageOps.grayscale(image)

    image = ImageEnhance.Contrast(image).enhance(1.5)

    return image


def extract_pdf_content(pdf_bytes: bytes):
    """
    Extract text from every PDF page.

    Method:
    1. Extract selectable text.
    2. If insufficient text is found, apply OCR.
    3. Return page-wise text and extraction details.
    """

    extracted_pages = []

    try:
        doc = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        if doc.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(doc, start=1):

            # Step 1: Extract selectable text
            text = page.get_text("text").strip()

            extraction_method = "Selectable text"
            ocr_used = False

            # Step 2: Apply OCR if selectable text is insufficient
            if len(text) < 80:

                extraction_method = "OCR"
                ocr_used = True

                # Render page at approximately 300 DPI
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(300 / 72, 300 / 72),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                # Step 3: Preprocess image
                image = preprocess_image(image)

                # Step 4: Apply OCR
                ocr_text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--psm 3"
                ).strip()

                # Use OCR result for this page
                if ocr_text:
                    text = ocr_text
                else:
                    text = text or "[No readable text detected on this page]"

            extracted_pages.append({
                "page": page_number,
                "text": text,
                "method": extraction_method,
                "ocr_used": ocr_used,
                "characters": len(text)
            })

        doc.close()

        return extracted_pages

    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")


# --------------------------------------------------
# 3. REPORT CLASSIFICATION
# --------------------------------------------------

def classify_report(text: str) -> str:

    t = text.lower()

    categories = {
        "Laboratory / Pathology": [
            "hemoglobin",
            "haemoglobin",
            "platelet",
            "wbc",
            "rbc",
            "glucose",
            "cholesterol",
            "creatinine",
            "bilirubin"
        ],

        "Radiology": [
            "ultrasound",
            "sonography",
            "x-ray",
            "xray",
            "ct scan",
            "mri",
            "impression",
            "radiologist"
        ],

        "Cardiology": [
            "ecg",
            "ekg",
            "echocardiogram",
            "ejection fraction",
            "cardiac"
        ],

        "Histopathology": [
            "histopathology",
            "biopsy",
            "specimen",
            "microscopy",
            "malignant"
        ]
    }

    scores = {
        name: sum(term in t for term in terms)
        for name, terms in categories.items()
    }

    best = max(scores, key=scores.get)

    if scores[best] > 0:
        return best

    return "General / Unclassified medical report"


# --------------------------------------------------
# 4. LABORATORY VALUE EXTRACTION
# --------------------------------------------------

def parse_simple_lab_rows(text: str):
    """
    Conservative prototype parser.

    Example:
    Hemoglobin 13.5 g/dL 12.0 - 16.0

    This parser supports only simple text formats.
    """

    rows = []

    pattern = re.compile(
        r"(?im)^\s*([A-Za-z][A-Za-z0-9 ()/%.-]{1,45}?)\s+"
        r"([-+]?\d+(?:\.\d+)?)\s*"
        r"([A-Za-zµμ/%^0-9.]+)?\s+"
        r"([-+]?\d+(?:\.\d+)?)\s*(?:-|to)\s*"
        r"([-+]?\d+(?:\.\d+)?)\s*$"
    )

    for match in pattern.finditer(text):

        name, value, unit, low, high = match.groups()

        value = float(value)
        low = float(low)
        high = float(high)

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


# --------------------------------------------------
# 5. BANGLA EXPLANATION
# --------------------------------------------------

def explain_status_bangla(status: str):

    mapping = {

        "Within stated range":
            "এই মানটি রিপোর্টে দেওয়া রেফারেন্স সীমার মধ্যে আছে।",

        "Below stated range":
            "এই মানটি রিপোর্টে দেওয়া রেফারেন্স সীমার নিচে। "
            "চিকিৎসকের সঙ্গে আলোচনা করুন।",

        "Above stated range":
            "এই মানটি রিপোর্টে দেওয়া রেফারেন্স সীমার উপরে। "
            "চিকিৎসকের সঙ্গে আলোচনা করুন।"
    }

    return mapping.get(
        status,
        "এই ফলাফলটি চিকিৎসকের সঙ্গে আলোচনা করুন।"
    )


# --------------------------------------------------
# 6. PDF UPLOAD
# --------------------------------------------------

uploaded = st.file_uploader(
    "Upload a medical report (PDF)",
    type=["pdf"]
)


if uploaded:

    pdf_bytes = uploaded.getvalue()

    # File size validation
    if len(pdf_bytes) > 15 * 1024 * 1024:

        st.error(
            "For this prototype, please upload a PDF smaller than 15 MB."
        )

        st.stop()

    # --------------------------------------------------
    # 7. PROCESS ALL PDF PAGES
    # --------------------------------------------------

    try:

        with st.spinner(
            "Processing PDF pages. OCR may take some time..."
        ):

            pages = extract_pdf_content(pdf_bytes)

        # Combine text from all pages
        extracted_text = "\n\n".join(
            f"--- Page {page['page']} ---\n{page['text']}"
            for page in pages
        )

    except Exception as e:

        st.error(str(e))
        st.stop()

    # --------------------------------------------------
    # 8. EXTRACTION SUMMARY
    # --------------------------------------------------

    total_pages = len(pages)

    total_characters = sum(
        page["characters"] for page in pages
    )

    ocr_pages = sum(
        1 for page in pages if page["ocr_used"]
    )

    st.success(
        f"PDF processing completed. "
        f"Detected category: {classify_report(extracted_text)}"
    )

    # --------------------------------------------------
    # 9. TABS
    # --------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Extracted Text",
        "Page Diagnostics",
        "Structured Findings"
    ])

    # --------------------------------------------------
    # TAB 1: OVERVIEW
    # --------------------------------------------------

    with tab1:

        report_type = classify_report(extracted_text)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Pages",
            total_pages
        )

        col2.metric(
            "Extracted Characters",
            f"{total_characters:,}"
        )

        col3.metric(
            "OCR Pages",
            ocr_pages
        )

        st.subheader("Report Information")

        st.write(
            f"**Detected Report Category:** {report_type}"
        )

        st.write(
            f"**Total Pages Processed:** {total_pages}"
        )

        st.write(
            f"**Pages Processed Using OCR:** {ocr_pages}"
        )

        st.info(
            "This prototype identifies report categories using keywords "
            "and extracts basic text. It does not yet interpret medical "
            "images, ECG waveforms, or diagnose conditions."
        )

    # --------------------------------------------------
    # TAB 2: EXTRACTED TEXT
    # --------------------------------------------------

    with tab2:

        st.subheader("Complete Extracted Report")

        st.text_area(
            "Extracted PDF Text",
            extracted_text,
            height=600
        )

        st.download_button(
            label="Download Extracted Text",
            data=extracted_text,
            file_name="extracted_medical_report.txt",
            mime="text/plain"
        )

    # --------------------------------------------------
    # TAB 3: PAGE DIAGNOSTICS
    # --------------------------------------------------

    with tab3:

        st.subheader("Page-by-Page Extraction Status")

        for page in pages:

            with st.expander(
                f"Page {page['page']} — "
                f"{page['method']} — "
                f"{page['characters']} characters"
            ):

                st.write(
                    f"**Extraction Method:** {page['method']}"
                )

                st.write(
                    f"**Extracted Characters:** {page['characters']}"
                )

                if page["ocr_used"]:
                    st.info("OCR was applied to this page.")

                else:
                    st.success(
                        "Selectable PDF text was extracted."
                    )

                if page["characters"] < 30:

                    st.warning(
                        "Very little readable text was detected. "
                        "Please verify this page against the original PDF."
                    )

                st.text_area(
                    f"Page {page['page']} content",
                    page["text"],
                    height=200,
                    key=f"page_text_{page['page']}"
                )

    # --------------------------------------------------
    # TAB 4: STRUCTURED FINDINGS
    # --------------------------------------------------

    with tab4:

        lab_rows = parse_simple_lab_rows(extracted_text)

        if lab_rows:

            st.subheader("Possible Laboratory Values")

            st.caption(
                "Experimental extraction only. "
                "Verify every value against the original report."
            )

            st.dataframe(
                lab_rows,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Result Overview")

            for index, row in enumerate(lab_rows):

                with st.container(border=True):

                    st.markdown(
                        f"**{row['Parameter']} — "
                        f"{row['Result']} {row['Unit']}**"
                    )

                    st.write(
                        f"Reference Range: {row['Reference range']}"
                    )

                    status = row["Prototype status"]

                    if status == "Within stated range":

                        st.success(status)

                    else:

                        st.warning(status)

                    st.write(
                        explain_status_bangla(status)
                    )

        else:

            st.info(
                "No simple laboratory values were detected. "
                "This is expected for many radiology, cardiology, "
                "and narrative reports. A report-specific extraction "
                "module is needed for each document family."
            )


# --------------------------------------------------
# 10. INITIAL SCREEN
# --------------------------------------------------

else:

    st.markdown("### Planned Workflow")

    st.write(
        "1. Upload PDF"
        "  →  2. Extract Text"
        "  →  3. Structure Findings"
        "  →  4. Explain in Bangla / English"
        "  →  5. Ask Follow-up Questions"
    )

    st.info(
        "Start with anonymized sample reports. "
        "Both text-based and scanned PDFs are supported by the prototype."
    )
