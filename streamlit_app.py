# app.py — Streamlit Cloud–ready, safe version (Brevo SMTP)

import os
import html
import re
from math import radians, sin, cos, asin, sqrt

import streamlit as st
from fpdf import FPDF
from groq import Groq
import pandas as pd
import requests

# -------------------- Page Config (must be first st.* call) -------------------- #
st.set_page_config(page_title="HealthCare AI Assistant", page_icon="🩺", layout="wide")

# -------------------- Secrets/env helper -------------------- #
def get_secret(name: str, required: bool = False, default: str | None = None) -> str | None:
    """
    Reads from st.secrets first, then environment. 
    If required and missing, shows an error and stops the app.
    """
    val = None
    try:
        # Streamlit Cloud
        val = st.secrets.get(name)
    except Exception:
        pass
    if val is None:
        # Local dev or other hosts
        val = os.getenv(name, default)
    if required and not val:
        st.error(f"Missing required secret: {name}")
        st.stop()
    return val

# Not strictly required to render the app; features will be disabled if missing.
GROQ_API_KEY = get_secret("GROQ_API_KEY", required=False)
GEOAPIFY_KEY = get_secret("GEOAPIFY_KEY", required=False)

# -------------------- Brevo SMTP Secrets -------------------- #
BREVO_SMTP_HOST = get_secret("BREVO_SMTP_HOST", required=False) or "smtp-relay.brevo.com"
BREVO_SMTP_PORT = int(get_secret("BREVO_SMTP_PORT", required=False) or 587)
BREVO_SMTP_LOGIN = get_secret("BREVO_SMTP_LOGIN", required=False)  # e.g. 9388b3001@smtp-brevo.com
BREVO_SMTP_PASSWORD = get_secret("BREVO_SMTP_PASSWORD", required=False)

# Optional: preferred From / Reply-To (use verified address if available)
BREVO_FROM_EMAIL = get_secret("BREVO_FROM_EMAIL", required=False) or BREVO_SMTP_LOGIN
BREVO_FROM_NAME = get_secret("BREVO_FROM_NAME", required=False) or "HealthCare AI Assistant"
BREVO_REPLY_TO_EMAIL = get_secret("BREVO_REPLY_TO_EMAIL", required=False)
BREVO_REPLY_TO_NAME = get_secret("BREVO_REPLY_TO_NAME", required=False)

# -------------------- Theme State -------------------- #
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme = st.session_state.theme
if theme == "dark":
    bg_color = "#121212"
    text_color = "#FFFFFF"
    header_color = "#5B0C86"
    card_color = "#1e1e26"
    result_color = "#2A2A2D"
    button_color = "#ffffff"
    button_text = "#6a1b9a"
    shadow = "0 4px 6px rgba(0, 0, 0, 0.1)"
else:
    bg_color = "#f5f5f5"
    text_color = "#000000"
    header_color = "#e1bee7"
    result_color = "#f7eeee"
    card_color = "#ffffff"
    button_color = "#ce93d8"
    button_text = "#000000"
    shadow = "0 4px 6px rgba(0, 0, 0, 0.1)"

# -------------------- Safe CSS -------------------- #
st.markdown(f"""
    <style>
        html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        .block-container {{ padding-top: 6rem !important; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        .custom-header {{
            position: fixed; top: 0; left: 0; right: 0; height: 60px;
            background-color: {header_color}; color: {text_color};
            padding: 0 2rem; z-index: 9999;
            display: flex; align-items: center; justify-content: space-between;
        }}
        .custom-header-title {{ font-size: 1.4rem; font-weight: bold; }}
        .stButton>button, .stDownloadButton>button {{
            background-color: {button_color} !important;
            color: {button_text} !important;
            font-weight: bold; border: none; padding: 8px 16px;
            border-radius: 6px; box-shadow: {shadow};
            transition: all 0.3s ease-in-out;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
        }}
        .result-container {{
            background-color: {result_color};
            padding: 15px; border-radius: 10px; margin-top: 10px;
            box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
            word-wrap: break-word; white-space: pre-wrap;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------- Header + Theme toggle -------------------- #
st.markdown("""
    <div class="custom-header">
        <div class="custom-header-title">🩺 HealthCare AI Assistant</div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    toggle_label = "🌞 Light Mode" if theme == "dark" else "🌙 Dark Mode"
    if st.button(toggle_label):
        st.session_state.theme = "light" if theme == "dark" else "dark"
        st.rerun()

# -------------------- Optional Images -------------------- #
def get_base64_image(path: str) -> str | None:
    try:
        if os.path.exists(path):
            import base64
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None

gif_base64 = get_base64_image("Online Doctor.gif")
if gif_base64:
    st.markdown(f"""
        <div style="text-align:center; margin-top:20px;">
            <img src="data:image/gif;base64,{gif_base64}" style="max-width:280px;"/>
        </div>
    """, unsafe_allow_html=True)

# -------------------- Groq Client -------------------- #
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.warning(f"LLM not available: {e}")

def ask_groq(user_input: str) -> str:
    if not client:
        return "LLM is not configured. Please add GROQ_API_KEY in Streamlit secrets."
    try:
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.2,
            max_tokens=800,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"LLM error: {e}"

# -------------------- Unicode PDF -------------------- #
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_page()
        # Make sure DejaVuSans.ttf is in repo root
        self.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        self.set_font("DejaVu", size=12)

    def add_unicode_text(self, text: str):
        self.multi_cell(0, 8, text)

# -------------------- Main Card -------------------- #
st.markdown(f"<div style='background-color:{card_color}; color:{text_color}; padding:2rem; border-radius:20px;'>",
            unsafe_allow_html=True)

st.subheader("Enter Symptoms or Health Concern")
st.caption("**Disclaimer:** This tool provides general information and is **not** a medical diagnosis. "
           "For emergencies or worsening symptoms, seek professional care immediately.")

symptoms = st.text_area(
    "Describe your issue",
    placeholder="e.g. skin rash, fatigue, fever...",
    height=150
)

if st.button("Analyze Symptoms"):
    if not symptoms.strip():
        st.warning("⚠️ Please describe your symptoms.")
    else:
        with st.spinner("Analyzing symptoms..."):
            llm_response = ask_groq(symptoms.strip())
        st.session_state.llm_response = llm_response
        st.session_state.pdf_generated = False
        st.success("✅ Analysis complete!")

# Render AI response (escaped to avoid HTML injection)
if "llm_response" in st.session_state:
    escaped = html.escape(st.session_state.llm_response or "")
    st.markdown(
        f'<div class="result-container"><p><strong>AI Response:</strong></p><div>{escaped}</div></div>',
        unsafe_allow_html=True
    )

    if st.button("📝 Generate PDF"):
        try:
            pdf = PDF()
            pdf.add_unicode_text(st.session_state.llm_response or "")
            pdf_path = "healthcare_report.pdf"
            pdf.output(pdf_path)
            st.session_state.pdf_generated = True
            st.session_state.pdf_path = pdf_path
            st.success("✅ PDF generated successfully!")
        except Exception as e:
            st.error(f"❌ Failed to generate PDF: {e}")

if st.session_state.get("pdf_generated", False):
    try:
        with open(st.session_state.pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="healthcare_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Could not open generated PDF: {e}")

    # -------------------- Email sending (optional) — Brevo SMTP -------------------- #
    st.markdown("---")
    st.subheader("📧 Send the PDF to your email (optional)")
    email = st.text_input("Recipient email", placeholder="you@example.com")

    EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    # Check SMTP config presence
    can_send_email = bool(BREVO_SMTP_HOST and BREVO_SMTP_PORT and BREVO_SMTP_LOGIN and BREVO_SMTP_PASSWORD)
    if not can_send_email:
        st.info("Email sending is not configured. Add **BREVO_SMTP_HOST**, **BREVO_SMTP_PORT**, "
                "**BREVO_SMTP_LOGIN**, and **BREVO_SMTP_PASSWORD** to Secrets to enable.")

    if st.button("📤 Send PDF to Email"):
        if not EMAIL_RE.match(email or ""):
            st.warning("Please enter a valid email address.")
        elif not can_send_email:
            st.error("Email sending not configured. Set Brevo SMTP secrets first.")
        else:
            try:
                with open(st.session_state.pdf_path, 'rb') as f:
                    pdf_binary_data = f.read()

                # Build MIME email
                from email.mime.multipart import MIMEMultipart
                from email.mime.text import MIMEText
                from email.mime.base import MIMEBase
                from email import encoders
                import smtplib

                msg = MIMEMultipart()
                msg['From'] = f"{BREVO_FROM_NAME} <{BREVO_FROM_EMAIL}>"
                msg['To'] = email.strip()
                msg['Subject'] = "Your Healthcare Report"

                # Optional Reply-To header
                if BREVO_REPLY_TO_EMAIL:
                    if BREVO_REPLY_TO_NAME:
                        msg.add_header('Reply-To', f"{BREVO_REPLY_TO_NAME} <{BREVO_REPLY_TO_EMAIL}>")
                    else:
                        msg.add_header('Reply-To', BREVO_REPLY_TO_EMAIL)

                msg.attach(MIMEText("Attached is your AI-generated healthcare report.", 'plain'))

                part = MIMEBase('application', 'octet-stream')
                part.set_payload(pdf_binary_data)
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment; filename=healthcare_report.pdf')
                msg.attach(part)

                # Send via Brevo SMTP (TLS on 587)
                with smtplib.SMTP(BREVO_SMTP_HOST, BREVO_SMTP_PORT, timeout=30) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(BREVO_SMTP_LOGIN, BREVO_SMTP_PASSWORD)
                    server.sendmail(BREVO_FROM_EMAIL or BREVO_SMTP_LOGIN, email.strip(), msg.as_string())

                st.success(f"✅ PDF sent to {email.strip()} successfully via Brevo SMTP!")
            except FileNotFoundError:
                st.error("❌ PDF file not found. Please generate the PDF first.")
            except Exception as e:
                st.error(f"❌ Failed to send email via Brevo SMTP: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Hospital Finder -------------------- #
st.markdown(f"<div style='background-color:{card_color}; color:{text_color}; padding:1.5rem; border-radius:15px; margin-top:30px;'>",
            unsafe_allow_html=True)

st.subheader("🏥 Find Nearby Hospitals (Optional)")

if not GEOAPIFY_KEY:
    st.info("Hospital search is disabled. Add **GEOAPIFY_KEY** to Secrets to enable.")
else:
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        location_input = st.text_input("Enter your city or address:", placeholder="e.g. Mughalpura, Lahore, Pakistan")
    with col_h2:
        st.markdown("<div style='margin-top: 26px;'>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search Hospitals")
        st.markdown("</div>", unsafe_allow_html=True)

    # Cached helpers
    @st.cache_data(ttl=3600)
    def geocode_address(address: str, api_key: str):
        try:
            r = requests.get(
                "https://api.geoapify.com/v1/geocode/search",
                params={"text": address, "apiKey": api_key},
                timeout=20
            )
            if r.ok:
                data = r.json()
                if data.get("features"):
                    p = data["features"][0]["properties"]
                    return p.get("lat"), p.get("lon")
        except Exception:
            return None, None
        return None, None

    @st.cache_data(ttl=3600)
    def find_nearby_hospitals(lat: float, lon: float, api_key: str):
        try:
            r = requests.get(
                "https://api.geoapify.com/v2/places",
                params={
                    "categories": "healthcare.hospital",
                    "filter": f"circle:{lon},{lat},35000",  # lon,lat
                    "limit": 10,
                    "apiKey": api_key,
                },
                timeout=20
            )
            return r.json() if r.ok else None
        except Exception:
            return None

    @st.cache_data(ttl=3600)
    def get_route_distance_km(start_lat, start_lon, end_lat, end_lon, api_key: str):
        """Driving distance in km using Geoapify routing (NOTE: waypoints are lon,lat)."""
        try:
            r = requests.get(
                "https://api.geoapify.com/v1/routing",
                params={
                    "waypoints": f"{start_lon},{start_lat}|{end_lon},{end_lat}",
                    "mode": "drive",
                    "details": "route_details",
                    "apiKey": api_key,
                },
                timeout=20
            )
            if r.ok:
                data = r.json()
                if data.get("features"):
                    return data["features"][0]["properties"]["distance"] / 1000
        except Exception:
            pass
        return None

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * asin(sqrt(a))

    if 'search_clicked' not in st.session_state:
        st.session_state.search_clicked = False

    if search_clicked and location_input:
        st.session_state.search_clicked = True
        with st.spinner("Locating and searching hospitals..."):
            lat, lon = geocode_address(location_input.strip(), GEOAPIFY_KEY)
            if lat and lon:
                hospitals = find_nearby_hospitals(lat, lon, GEOAPIFY_KEY)
                if hospitals and hospitals.get("features"):
                    data_rows = []
                    for hospital in hospitals["features"]:
                        props = hospital.get("properties", {})
                        name = props.get("name", "Unknown")
                        address = props.get("formatted", "No address available")
                        h_lat, h_lon = props.get("lat"), props.get("lon")
                        # Fast approximate distance for all results
                        approx_km = haversine_km(lat, lon, h_lat, h_lon) if (h_lat and h_lon) else None
                        data_rows.append({
                            "Name": name,
                            "Address": address,
                            "Approx. Distance (km)": round(approx_km, 2) if approx_km is not None else "N/A",
                            "Coordinates": f"({h_lat}, {h_lon})",
                            "lat": h_lat,
                            "lon": h_lon
                        })

                    df = pd.DataFrame(data_rows)
                    st.subheader("Results")
                    st.dataframe(df[["Name", "Address", "Approx. Distance (km)", "Coordinates"]], use_container_width=True)

                    # Optional: On-demand driving distance to save quota/time
                    with st.expander("Compute driving distances (slower; uses extra API calls)"):
                        top_n = st.slider("Compute for top N by approximate distance:", min_value=1, max_value=min(10, len(df)), value=5)
                        if st.button("Calculate driving distances"):
                            with st.spinner("Calculating driving distances..."):
                                df_sorted = df.sort_values(by="Approx. Distance (km)", key=lambda s: pd.to_numeric(s, errors="coerce"))
                                subset = df_sorted.head(top_n).copy()
                                dists = []
                                for _, row in subset.iterrows():
                                    d = get_route_distance_km(lat, lon, float(row["lat"]), float(row["lon"]), GEOAPIFY_KEY)
                                    dists.append(round(d, 2) if d is not None else None)
                                subset["Driving Distance (km)"] = dists
                                st.dataframe(subset[["Name", "Address", "Approx. Distance (km)", "Driving Distance (km)", "Coordinates"]],
                                             use_container_width=True)
                else:
                    st.warning("⚠️ No hospitals found or API returned no features.")
            else:
                st.error("❌ Invalid address or geocoding failed.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Footer -------------------- #
st.markdown(f"""
    <p style='text-align:center;color:{text_color};margin-top:40px;'>
        © 2025 <strong>Team HealthGenix</strong> • For information only, not a medical diagnosis.
    </p>
""", unsafe_allow_html=True)
