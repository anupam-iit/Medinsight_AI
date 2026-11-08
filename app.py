import streamlit as st
import fitz
import requests
import json
import textwrap
import io

from PIL import Image

# Optional OCR dependency
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MediInsight AI",
    page_icon="🩺",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_CHUNK_SIZE = 7000


# =========================================================
# 2. APPLICATION DESIGN
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Hind Siliguri', sans-serif;
}

.main {
    background-color: #f5f8fa;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

.header-box {
    background: linear-gradient(120deg, #063b46, #087e83);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.header-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.header-subtitle {
    font-size: 16px;
    opacity: 0.9;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.info-note {
    background: #e8f3f5;
    border-left: 5px solid #087e83;
    padding: 14px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HEADER
# =========================================================

st.markdown("""
<div class="header-box">

    <div class="header-title">
        🩺 MediInsight AI
    </div>

    <div class="header-subtitle">
        Your Medical Report Understanding Assistant
        <br>
        Medical reports explained in simple Bangla
    </div>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="info-note">

<b>Privacy:</b> Your uploaded report is processed by the
locally running AI model on this computer. This application
does not send report content to Gemini or OpenAI.

<b>Important:</b> AI-generated explanations are informational,
not medical diagnoses. Consult a qualified healthcare
professional for medical decisions.

</div>
""", unsafe_allow_html=True)


# =========================================================
# 4. PDF EXTRACTION
# =========================================================

def extract_pdf_content(pdf_bytes, run_ocr=False):

    extracted_pages = []

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page_number, page in enumerate(pdf):

        native_text = page.get_text("text").strip()

        extracted_text = native_text
        extraction_method = "PDF Text"

        should_run_ocr = (
            OCR_AVAILABLE and
            (len(native_text) < 80 or run_ocr)
        )

        if should_run_ocr:

            try:

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                ocr_text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--psm 3"
                ).strip()

                if len(ocr_text) > len(native_text):

                    extracted_text = ocr_text
                    extraction_method = "OCR"

            except Exception:
                pass

        extracted_pages.append({

            "page": page_number + 1,

            "text": extracted_text,

            "method": extraction_method

        })

    pdf.close()

    return extracted_pages


# =========================================================
# 5. TEXT CHUNKING
# =========================================================

def split_text_into_chunks(text, max_chars=MAX_CHUNK_SIZE):

    lines = text.splitlines()

    chunks = []
    current_chunk = ""

    for line in lines:

        if len(current_chunk) + len(line) > max_chars:

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = line

        else:

            current_chunk += "\n" + line

    if current_chunk.strip():

        chunks.append(current_chunk.strip())

    return chunks


# =========================================================
# 6. AI RESPONSE STRUCTURE
# =========================================================

OUTPUT_STRUCTURE = {

    "report_type": "",

    "report_purpose_bn": "",

    "good_findings": [

        {
            "finding": "",
            "result": "",
            "reference_range": "",
            "explanation_bn": ""
        }

    ],

    "attention_findings": [

        {
            "finding": "",
            "result": "",
            "reference_range": "",
            "explanation_bn": "",
            "discussion_bn": ""
        }

    ],

    "overall_summary_bn": "",

    "important_observations_bn": [],

    "patient_guidance_bn": [],

    "limitations_bn": []

}


# =========================================================
# 7. OLLAMA API CONNECTION
# =========================================================

def call_ollama(prompt, model):

    url = f"{OLLAMA_URL}/api/chat"

    payload = {

        "model": model,

        "messages": [

            {
                "role": "user",
                "content": prompt
            }

        ],

        "stream": False,

        "format": "json",

        "keep_alive": "10m",

        "options": {

            "temperature": 0.1,

            "num_ctx": 4096,

            "num_predict": 1200

        }

    }

    response = requests.post(

        url,

        json=payload,

        timeout=(10, 600)

    )

    response.raise_for_status()

    response_data = response.json()

    ai_text = response_data["message"]["content"]

    return json.loads(ai_text)


# =========================================================
# 8. MEDICAL REPORT AI PROMPT
# =========================================================

def build_analysis_prompt(report_text, final_analysis=False):

    if final_analysis:

        task_instruction = """

        You are now analyzing the combined results of
        multiple sections of the same medical report.

        Consolidate all findings into one coherent analysis.

        Remove duplicate findings.

        Consider relationships between different findings.

        Explain the overall report in simple Bangla.

        Do not invent additional findings.

        Do not treat individual findings as confirmed
        diagnoses.

        """

    else:

        task_instruction = """

        Analyze this section of a medical report.

        Identify all clearly recognizable findings.

        Do not assume missing test names, values,
        units or reference ranges.

        Do not diagnose diseases.

        """

    prompt = f"""

You are the specialized medical report understanding
engine of MediInsight AI.

Your task is to understand medical reports and explain
their contents in simple, understandable Bangla.

{task_instruction}


IMPORTANT ANALYSIS INSTRUCTIONS:

1. REPORT IDENTIFICATION

Identify the type and purpose of the medical report.

2. GOOD FINDINGS

Identify results that are explicitly within their
provided reference ranges.

A finding should only be classified as good when
the result and its reference range can be understood.

Do not assume that a result is normal when no
reference range or clear interpretation is provided.

3. ATTENTION REQUIRED

Identify results that are:

- Above the provided reference range.
- Below the provided reference range.
- Explicitly flagged as H, L, High or Low.
- Described as abnormal in the report.

Explain what the finding means.

An abnormal result does not automatically mean
the patient has a disease.

4. OVERALL REPORT ANALYSIS

Consider the report as a whole.

Explain:

- What the report is about.
- What the overall findings suggest.
- Which observations are important.
- Whether related findings should be considered together.

5. GENERAL PATIENT GUIDANCE

Provide safe, general information about what the
patient may discuss with a qualified healthcare
professional.

Do not prescribe medicines.

Do not recommend starting or stopping treatment.

Do not provide a definitive diagnosis.

6. ACCURACY

Never invent:

- Test results.
- Reference ranges.
- Patient information.
- Diagnoses.
- Medical history.

If OCR text is unclear, acknowledge the uncertainty.

If the report contains radiology or other imaging
findings, explain only what is stated in its extracted
written report. Do not claim to have examined the
original medical image.

7. LANGUAGE

Write explanations in simple Bangla.

Keep medical test names and technical terms in English
where necessary.

Avoid unnecessarily complicated medical language.


RETURN VALID JSON USING THIS EXACT STRUCTURE:

{json.dumps(OUTPUT_STRUCTURE, ensure_ascii=False)}


MEDICAL REPORT CONTENT:

{report_text}

"""

    return prompt


# =========================================================
# 9. ANALYZE INDIVIDUAL REPORT SECTIONS
# =========================================================

def analyze_report_chunks(pages, model):

    all_chunks = []

    for page in pages:

        chunks = split_text_into_chunks(
            page["text"]
        )

        for chunk in chunks:

            all_chunks.append({

                "page": page["page"],

                "text": chunk

            })

    if not all_chunks:

        raise ValueError(
            "No readable text found in the PDF."
        )

    partial_results = []

    progress = st.progress(0)

    status = st.empty()

    total = len(all_chunks)

    for index, chunk in enumerate(all_chunks):

        status.info(

            f"AI analyzing section {index + 1} of {total}..."

        )

        prompt = build_analysis_prompt(

            f"Page {chunk['page']}:\n{chunk['text']}",

            final_analysis=False

        )

        result = call_ollama(prompt, model)

        partial_results.append(result)

        progress.progress(
            (index + 1) / total
        )

    progress.empty()

    status.empty()

    return partial_results


# =========================================================
# 10. CONSOLIDATE ANALYSIS
# =========================================================

def consolidate_analysis(partial_results, model):

    results = partial_results

    # Reduce large collections in groups.
    while len(results) > 3:

        reduced_results = []

        for start in range(0, len(results), 3):

            group = results[start:start + 3]

            prompt = build_analysis_prompt(

                json.dumps(
                    group,
                    ensure_ascii=False
                ),

                final_analysis=True

            )

            merged = call_ollama(prompt, model)

            reduced_results.append(merged)

        results = reduced_results

    final_prompt = build_analysis_prompt(

        json.dumps(
            results,
            ensure_ascii=False
        ),

        final_analysis=True

    )

    return call_ollama(
        final_prompt,
        model
    )


# =========================================================
# 11. COMPLETE REPORT ANALYSIS
# =========================================================

def generate_ai_analysis(pages, model):

    partial_results = analyze_report_chunks(

        pages,

        model

    )

    final_result = consolidate_analysis(

        partial_results,

        model

    )

    return final_result


# =========================================================
# 12. DISPLAY GOOD FINDINGS
# =========================================================

def display_good_findings(findings):

    st.subheader("🟢 Good Findings")

    st.caption(
        "Results identified as within their stated "
        "reference ranges."
    )

    if not findings:

        st.info(
            "No findings could be confidently classified "
            "as within the stated reference ranges."
        )

        return

    for item in findings:

        with st.container(border=True):

            st.markdown(
                f"**{item.get('finding', 'Finding')}**"
            )

            st.write(
                f"Result: {item.get('result', 'Not available')}"
            )

            st.write(
                f"Reference range: "
                f"{item.get('reference_range', 'Not available')}"
            )

            st.write(
                item.get(
                    "explanation_bn",
                    "ব্যাখ্যা পাওয়া যায়নি।"
                )
            )


# =========================================================
# 13. DISPLAY ATTENTION FINDINGS
# =========================================================

def display_attention_findings(findings):

    st.subheader("🟠 Attention Required")

    st.caption(
        "Results identified as outside their stated "
        "reference ranges or explicitly flagged."
    )

    if not findings:

        st.info(
            "No clearly out-of-range findings were "
            "identified by the AI. This does not mean "
            "that the entire report is normal."
        )

        return

    for item in findings:

        with st.container(border=True):

            st.markdown(
                f"**{item.get('finding', 'Finding')}**"
            )

            st.write(
                f"Result: {item.get('result', 'Not available')}"
            )

            st.write(
                f"Reference range: "
                f"{item.get('reference_range', 'Not available')}"
            )

            st.write(
                item.get(
                    "explanation_bn",
                    "ব্যাখ্যা পাওয়া যায়নি।"
                )
            )

            discussion = item.get("discussion_bn", "")

            if discussion:

                st.info(
                    f"চিকিৎসকের সঙ্গে আলোচনা: {discussion}"
                )


# =========================================================
# 14. DISPLAY OVERALL ANALYSIS
# =========================================================

def display_overall_analysis(analysis):

    st.subheader("🔵 Overall Report Analysis")

    with st.container(border=True):

        st.markdown("### Report Identification")

        st.write(
            analysis.get(
                "report_type",
                "Not identified"
            )
        )

        st.write(
            analysis.get(
                "report_purpose_bn",
                ""
            )
        )

        st.divider()

        st.markdown("### সামগ্রিক রিপোর্ট বিশ্লেষণ")

        st.write(
            analysis.get(
                "overall_summary_bn",
                "সারাংশ পাওয়া যায়নি।"
            )
        )

        st.divider()

        st.markdown("### গুরুত্বপূর্ণ পর্যবেক্ষণ")

        observations = analysis.get(
            "important_observations_bn",
            []
        )

        for observation in observations:

            st.markdown(f"- {observation}")

        st.divider()

        st.markdown("### সাধারণ পরামর্শ")

        guidance = analysis.get(
            "patient_guidance_bn",
            []
        )

        for advice in guidance:

            st.markdown(f"- {advice}")

        st.divider()

        st.markdown("### সীমাবদ্ধতা ও সতর্কতা")

        limitations = analysis.get(
            "limitations_bn",
            []
        )

        for limitation in limitations:

            st.markdown(f"- {limitation}")


# =========================================================
# 15. SIDEBAR SETTINGS
# =========================================================

with st.sidebar:

    st.header("⚙️ AI Settings")

    model = st.text_input(

        "Ollama Model",

        value=DEFAULT_MODEL

    )

    st.caption(
        "The model must already be downloaded "
        "and available in Ollama."
    )

    st.divider()

    st.markdown("### System Status")

    st.write("AI Runtime: Ollama")

    st.write("Processing: Local")

    st.write("Language: Bangla + English")

    st.write("AI Model:", model)

    if OCR_AVAILABLE:

        st.success("OCR Python package available")

    else:

        st.warning(
            "OCR Python package unavailable. "
            "Scanned PDFs may not work."
        )


# =========================================================
# 16. PDF UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">Upload Medical Report</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(

    "Select your medical report (PDF)",

    type=["pdf"]

)

run_ocr = st.checkbox(

    "Run OCR on every page (useful for scanned or "
    "image-based reports)",

    value=False,

    disabled=not OCR_AVAILABLE

)


# =========================================================
# 17. PROCESS REPORT
# =========================================================

if uploaded_file is not None:

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "File size exceeds the 20 MB limit."
        )

    else:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "🩺 Analyze Medical Report",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "Extracting medical report..."
                ):

                    pdf_bytes = uploaded_file.getvalue()

                    extracted_pages = extract_pdf_content(

                        pdf_bytes,

                        run_ocr

                    )

                full_text = "\n".join(

                    page["text"]

                    for page in extracted_pages

                )

                if not full_text.strip():

                    st.error(
                        "No readable text could be extracted "
                        "from this PDF."
                    )

                    st.stop()

                # Show extracted text for verification.
                with st.expander(
                    "View Extracted Report Text"
                ):

                    st.text(full_text)

                st.info(
                    f"Extracted {len(extracted_pages)} pages."
                )

                with st.spinner(
                    "Checking connection with Ollama..."
                ):

                    requests.get(
                        f"{OLLAMA_URL}/api/tags",
                        timeout=10
                    ).raise_for_status()

                with st.spinner(
                    "MediInsight AI is analyzing your complete report. "
                    "This may take some time..."
                ):

                    analysis = generate_ai_analysis(

                        extracted_pages,

                        model

                    )

                st.session_state["analysis"] = analysis

                st.success(
                    "Medical report analysis completed!"
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to Ollama. "
                    "Please ensure Ollama is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "AI processing timed out. "
                    "Try again with a shorter report."
                )

            except requests.exceptions.HTTPError as error:

                st.error(
                    f"Ollama API error: {error}"
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )


# =========================================================
# 18. DISPLAY AI RESULTS
# =========================================================

if "analysis" in st.session_state:

    analysis = st.session_state["analysis"]

    st.divider()

    st.markdown(
        '<div class="section-title">'
        'Medical Report Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    # Two finding grids
    col1, col2 = st.columns(2)

    with col1:

        display_good_findings(

            analysis.get(
                "good_findings",
                []
            )

        )

    with col2:

        display_attention_findings(

            analysis.get(
                "attention_findings",
                []
            )

        )

    st.divider()

    # Overall analysis
    display_overall_analysis(analysis)


# =========================================================
# 19. FOOTER
# =========================================================

st.markdown("""
<div class="footer">

    <b>MediInsight AI</b><br>

    Developed by: Anupam Roy<br>

    A Student of PMIT<br>

    Medical Report Understanding Assistant<br>

    © 2026 MediInsight AI. All Rights Reserved.<br><br>

    <small>
    This application provides educational information
    and does not replace professional medical advice.
    </small>

</div>
""", unsafe_allow_html=True)
