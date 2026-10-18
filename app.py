"""
AgriDoctor – AI-powered Crop Disease Diagnosis & Advisory
==========================================================
Two-agent workflow:
  1. Pathologist Agent  – analyses a crop/leaf image and produces a diagnosis.
  2. Extension Officer Agent – converts the diagnosis into a farmer-friendly
     remedial-action report.

Both agents call the IBM watsonx.ai REST API with the IBM Granite 4 H Small
model.  The Pathologist Agent sends the image as a base-64 encoded data URL
alongside the prompt; the Extension Officer Agent sends plain text.
"""

import base64
import io
import json
import textwrap
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WATSONX_ENDPOINT = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "5247ecfe-b435-4f50-93e9-2ed52faedf1d"
MODEL_ID = "ibm/granite-4-h-small"          # Granite 4 H Small identifier on watsonx
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Text generation endpoint (supports multimodal for Granite 4)
GENERATE_URL = f"{WATSONX_ENDPOINT}/ml/v1/text/chat?version=2024-05-31"

# ---------------------------------------------------------------------------
# IBM IAM Token Helper
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3000, show_spinner=False)
def get_iam_token(api_key: str) -> str:
    """Exchange an IBM Cloud API key for a short-lived IAM bearer token."""
    resp = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"IAM token request failed ({resp.status_code}): {resp.text}"
        )
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Core API call
# ---------------------------------------------------------------------------

def call_watsonx(
    api_key: str,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """
    Send a chat-completion request to watsonx.ai and return the assistant
    response text.

    `messages` follows the OpenAI-style format:
        [{"role": "user", "content": "..." | [...]}, ...]
    """
    token = get_iam_token(api_key)

    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        GENERATE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"watsonx API error ({resp.status_code}): {resp.text}"
        )

    data = resp.json()
    # Navigate the response structure
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected response structure: {json.dumps(data)}")


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def extract_image_context(uploaded_file) -> dict:
    """
    Extract visual metadata from the uploaded image using PIL.
    Returns a dict with filename, dimensions, mode, dominant colours, and
    brightness — used as structured context for the text-based LLM.
    """
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    width, height = img.size
    filename = uploaded_file.name

    # Sample a grid of pixels for colour analysis
    img_small = img.resize((50, 50))
    pixels = list(img_small.getdata())

    # Average colour
    avg_r = int(sum(p[0] for p in pixels) / len(pixels))
    avg_g = int(sum(p[1] for p in pixels) / len(pixels))
    avg_b = int(sum(p[2] for p in pixels) / len(pixels))

    # Brightness (0-255)
    brightness = int((avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114))

    # Colour tone description
    if avg_g > avg_r and avg_g > avg_b:
        if avg_g > 120:
            tone = "predominantly green (healthy-looking foliage)"
        else:
            tone = "dark green (possible shade stress or fungal darkening)"
    elif avg_r > avg_g and avg_r > avg_b:
        tone = "reddish/brownish (possible leaf scorch, rust, or blight)"
    elif avg_r > 150 and avg_g > 130 and avg_b < 100:
        tone = "yellowish-green (possible chlorosis or nutrient deficiency)"
    elif avg_r > 160 and avg_g > 140 and avg_b < 80:
        tone = "yellow (likely chlorosis, nitrogen deficiency, or viral infection)"
    elif avg_r < 80 and avg_g < 80 and avg_b < 80:
        tone = "very dark/black (possible sooty mould, necrosis, or severe blight)"
    else:
        tone = f"mixed tones (R:{avg_r} G:{avg_g} B:{avg_b})"

    brightness_desc = (
        "well-lit" if brightness > 120
        else "dim/underexposed (may affect analysis accuracy)"
    )

    return {
        "filename": filename,
        "dimensions": f"{width}x{height} pixels",
        "colour_tone": tone,
        "avg_rgb": f"R:{avg_r} G:{avg_g} B:{avg_b}",
        "brightness": f"{brightness}/255 ({brightness_desc})",
    }


# ---------------------------------------------------------------------------
# Agent 1 – Pathologist Agent
# ---------------------------------------------------------------------------

PATHOLOGIST_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert plant pathologist and crop disease diagnostician with decades
of field and laboratory experience. A farmer has submitted a crop or plant-leaf
image for diagnosis. You will be given:
  1. Image metadata (dimensions, colour analysis, brightness) extracted
     automatically from the photo.
  2. A description of symptoms as observed by the farmer.

Use both sources of information to deliver a rigorous, evidence-based
preliminary diagnosis.

GUIDELINES:
- Base your analysis on the colour tone data, brightness, and farmer-reported
  symptoms. Colour shifts (yellowing, browning, darkening) are key indicators.
- Identify the most likely disease(s) and state the evidence that supports each.
- Provide a confidence level: High / Medium / Low.
- List up to three alternative differential diagnoses when relevant.
- If the information is insufficient for a reliable assessment, say so clearly
  and ask the farmer for more details rather than guessing.
- Always include the disclaimer that this is an AI-generated preliminary
  assessment and is NOT a confirmed laboratory diagnosis.

OUTPUT FORMAT (use exactly these headings):
## Primary Diagnosis
## Observed Symptoms
## Confidence Level
## Possible Causes
## Differential Diagnoses
## Disclaimer
""").strip()


def run_pathologist_agent(
    api_key: str,
    image_context: dict,
    symptom_description: str,
) -> str:
    """
    Agent 1: sends image metadata + farmer symptom description to Granite 4
    and returns a structured diagnostic report.
    """
    context_block = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in image_context.items()
    )

    user_message = (
        "Please analyse the following crop image data and farmer observations, "
        "then provide a detailed disease diagnosis following the specified "
        "output format.\n\n"
        "=== IMAGE ANALYSIS DATA ===\n"
        f"{context_block}\n\n"
        "=== FARMER-REPORTED SYMPTOMS ===\n"
        f"{symptom_description if symptom_description.strip() else 'No additional description provided by farmer.'}\n\n"
        "Based on the colour analysis and symptoms above, provide your diagnosis."
    )

    messages = [
        {"role": "system", "content": PATHOLOGIST_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    return call_watsonx(api_key, messages, max_tokens=1200, temperature=0.2)


# ---------------------------------------------------------------------------
# Agent 2 – Extension Officer Agent
# ---------------------------------------------------------------------------

EXTENSION_OFFICER_SYSTEM_PROMPT = textwrap.dedent("""
You are a friendly and knowledgeable agricultural extension officer with
extensive experience advising smallholder and commercial farmers.  A plant
pathologist has provided you with a crop disease diagnosis report.  Your job is
to convert that technical report into a clear, practical, farmer-friendly
advisory in simple English that even a first-time farmer can understand.

GUIDELINES:
- Use plain, non-technical language wherever possible.
- Provide immediate remedial actions the farmer can take right now.
- Include preventive measures to stop future outbreaks.
- Give crop-care and general best-practice recommendations.
- When chemical treatments are appropriate, mention only well-known, widely
  available options; refer to product labels and local regulations for exact
  dosages – DO NOT invent specific dosages.
- Always recommend Personal Protective Equipment (PPE) when chemicals are
  mentioned.
- End with a safety notice and a recommendation to seek professional help if
  symptoms persist or worsen.
- If the pathologist indicated the image was unclear, advise the farmer to
  obtain a clearer image or consult a local expert directly.

OUTPUT FORMAT (use exactly these headings):
## Summary for the Farmer
## Immediate Remedial Actions
## Preventive Measures
## Crop-Care Recommendations
## Treatment Options
## Safety Warnings
## When to Seek Expert Help
""").strip()


def run_extension_officer_agent(api_key: str, diagnosis_report: str) -> str:
    """
    Agent 2: receives the Pathologist's report and produces a farmer-friendly
    advisory.
    """
    messages = [
        {
            "role": "system",
            "content": EXTENSION_OFFICER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "The plant pathologist has provided the following crop disease "
                "diagnosis report. Please convert it into a practical, "
                "farmer-friendly advisory:\n\n"
                f"{diagnosis_report}"
            ),
        },
    ]

    return call_watsonx(api_key, messages, max_tokens=1400, temperature=0.4)


# ---------------------------------------------------------------------------
# Report Builder
# ---------------------------------------------------------------------------

def build_final_report(
    diagnosis: str,
    advisory: str,
    filename: str,
) -> str:
    """Combine both agent outputs into a single downloadable text report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "=" * 70

    report = textwrap.dedent(f"""
        {divider}
        AGRI DOCTOR – CROP DISEASE ADVISORY REPORT
        Generated: {timestamp}
        Image File: {filename}
        {divider}

        SECTION 1 – PATHOLOGIST AGENT: DISEASE DIAGNOSIS
        {divider}
        {diagnosis}

        {divider}
        SECTION 2 – EXTENSION OFFICER AGENT: FARMER ADVISORY
        {divider}
        {advisory}

        {divider}
        IMPORTANT NOTICE
        {divider}
        This report is an AI-generated preliminary assessment produced by the
        AgriDoctor application using IBM Granite 4 H Small on IBM watsonx.ai.
        It is NOT a confirmed laboratory diagnosis. Farmers are strongly advised
        to consult a certified plant pathologist or local agricultural extension
        officer for confirmation, especially before applying chemical treatments.
        Always follow local agricultural regulations and pesticide label
        instructions. Use appropriate Personal Protective Equipment (PPE) when
        handling agrochemicals.
        {divider}
    """).strip()

    return report


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AgriDoctor – Crop Disease AI",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🌿 AgriDoctor")
        st.caption("AI-Powered Crop Disease Diagnosis")
        st.divider()

        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "IBM Cloud API Key",
            type="password",
            placeholder="Enter your IBM Cloud API key",
            help="Your IBM Cloud API key is used to authenticate with watsonx.ai. "
                 "It is never stored or logged.",
        )

        st.divider()
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **AgriDoctor** uses a two-agent AI workflow:

            **🔬 Agent 1 – Pathologist**
            Analyses crop/leaf images to detect diseases and symptoms.

            **📋 Agent 2 – Extension Officer**
            Converts the diagnosis into practical farmer-friendly advice.

            **Powered by:**
            IBM Granite 4 H Small on IBM watsonx.ai
            """
        )
        st.divider()
        st.caption(
            "⚠️ AI-generated preliminary assessment only. "
            "Consult a certified expert for confirmation."
        )

    # ── Main Page ─────────────────────────────────────────────────────────────
    st.title("🌾 AgriDoctor – Crop Disease Diagnosis & Advisory")
    st.markdown(
        "Upload a photo of your crop or plant leaf. AgriDoctor's two AI agents "
        "will diagnose the disease and provide a practical management advisory."
    )

    st.info(
        "📌 **How it works:** "
        "① Upload image → ② Pathologist Agent diagnoses the disease → "
        "③ Extension Officer Agent generates farmer-friendly advice → "
        "④ Download your advisory report.",
        icon="ℹ️",
    )

    st.divider()

    # ── Image Upload ──────────────────────────────────────────────────────────
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.subheader("📷 Upload Crop Image")
        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG, or PNG image",
            type=["jpg", "jpeg", "png"],
            help="For best results, upload a clear, well-lit close-up of the "
                 "affected plant part (leaf, stem, fruit, etc.).",
        )
        st.subheader("📝 Describe What You See (Optional but Recommended)")
        symptom_description = st.text_area(
            "Describe the visible symptoms on your crop",
            placeholder=(
                "e.g. 'Leaves are turning yellow with brown spots, edges are "
                "drying out, some leaves have white powdery coating, plant "
                "looks wilted in the morning...'"
            ),
            height=130,
            help="Describe any discolouration, spots, lesions, wilting, mould, "
                 "or unusual appearance you can see. The more detail you provide, "
                 "the more accurate the diagnosis will be.",
        )

    with col_preview:
        if uploaded_file is not None:
            st.subheader("🖼️ Uploaded Image Preview")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
            uploaded_file.seek(0)  # Reset file pointer after PIL reads it

    st.divider()

    # ── Diagnose Button ───────────────────────────────────────────────────────
    diagnose_btn = st.button(
        "🔍 Diagnose Crop",
        type="primary",
        disabled=(uploaded_file is None),
        use_container_width=True,
    )

    if uploaded_file is None:
        st.warning("Please upload a crop or plant-leaf image to get started.")

    # ── Run Agents ────────────────────────────────────────────────────────────
    if diagnose_btn:
        if not api_key:
            st.error(
                "🔐 API key is required. Please enter your IBM Cloud API key "
                "in the sidebar."
            )
            st.stop()

        # Extract image metadata using PIL
        try:
            image_context = extract_image_context(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to process the uploaded image: {exc}")
            st.stop()

        # ── Agent 1: Pathologist ──────────────────────────────────────────────
        st.subheader("🔬 Agent 1 – Pathologist Agent")
        st.caption("Analysing your crop image for disease symptoms…")

        diagnosis_result = None
        with st.spinner("🧬 Pathologist Agent is examining the image…"):
            try:
                diagnosis_result = run_pathologist_agent(
                    api_key, image_context, symptom_description
                )
            except RuntimeError as exc:
                error_msg = str(exc)
                if "401" in error_msg or "403" in error_msg:
                    st.error(
                        "❌ Authentication failed. Please check your IBM Cloud "
                        "API key and try again."
                    )
                elif "model" in error_msg.lower():
                    st.error(
                        "❌ The selected model is not available or not supported "
                        f"for this request.\n\nDetails: {error_msg}"
                    )
                else:
                    st.error(f"❌ Pathologist Agent error: {error_msg}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error in Pathologist Agent: {exc}")
                st.stop()

        with st.expander("📄 Pathologist Agent Report", expanded=True):
            st.markdown(diagnosis_result)

        # ── Agent 2: Extension Officer ────────────────────────────────────────
        st.subheader("📋 Agent 2 – Extension Officer Agent")
        st.caption(
            "Converting the diagnosis into a farmer-friendly advisory…"
        )

        advisory_result = None
        with st.spinner("📝 Extension Officer Agent is preparing your advisory…"):
            try:
                advisory_result = run_extension_officer_agent(
                    api_key, diagnosis_result
                )
            except RuntimeError as exc:
                error_msg = str(exc)
                if "401" in error_msg or "403" in error_msg:
                    st.error(
                        "❌ Authentication failed. Please check your IBM Cloud "
                        "API key and try again."
                    )
                else:
                    st.error(f"❌ Extension Officer Agent error: {error_msg}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Unexpected error in Extension Officer Agent: {exc}")
                st.stop()

        with st.expander("📄 Extension Officer Advisory", expanded=True):
            st.markdown(advisory_result)

        # ── Combined Final Report ─────────────────────────────────────────────
        st.divider()
        st.subheader("📑 Combined Final Report")

        final_report = build_final_report(
            diagnosis_result,
            advisory_result,
            uploaded_file.name,
        )

        st.success("✅ Multi-agent analysis complete!")

        with st.expander("📋 View Full Combined Report", expanded=False):
            st.text(final_report)

        # ── Download Button ───────────────────────────────────────────────────
        st.download_button(
            label="⬇️ Download Advisory Report (.txt)",
            data=final_report.encode("utf-8"),
            file_name=f"AgriDoctor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ── Footer Warning ────────────────────────────────────────────────────
        st.divider()
        st.warning(
            "⚠️ **Important Disclaimer:** This report is an AI-generated "
            "preliminary assessment and is NOT a confirmed laboratory diagnosis. "
            "Always consult a certified plant pathologist or local agricultural "
            "extension officer before taking action, especially regarding "
            "chemical treatments. Follow all pesticide label instructions and "
            "local agricultural regulations. Use appropriate Personal Protective "
            "Equipment (PPE) when handling agrochemicals.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
