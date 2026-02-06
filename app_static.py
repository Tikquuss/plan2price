"""
AcierUnited - O2 Prototype (hard-coded output)
---------------------------------------------
Démo client crédible:
- Upload PDF (obligatoire avant affichage)
- Latence simulée (spinner/progress) à chaque upload/re-upload
- Toujours retourne le même JSON hard-coded (simule O1)
- Prix (CAD/kg ou CAD/tonne) -> coût mis à jour instantanément
- Exports CSV/JSON nommés selon le PDF uploadé
- Bouton Reset / Retour accueil visible

Run:
    pip install streamlit pandas
    streamlit run app.py
"""

import json
import io
import os
import time
import hashlib
from copy import deepcopy

import pandas as pd
import streamlit as st


# ============================================================
# 1) HARD-CODED "GROUND TRUTH" RESULT (ALWAYS RETURN THIS)
# ============================================================

GROUND_TRUTH = {
    "prototype_mode": {
        "always_return_this": True,
        "source": "manual_ground_truth_from_images",
        "notes": "Ignore uploaded PDF content for prototype. Always return this same result."
    },
    "units": {
        "length_display": "ft-in",
        "length_calc": "m",
        "weight": "kg",
        "price_input": ["CAD_per_kg", "CAD_per_tonne"]
    },
    "constants": {
        "kg_per_m": {"10M": 0.785, "15M": 1.57},
        "conversions": {"ft_to_m": 0.3048, "in_to_m": 0.0254}
    },
    "line_items": [
        {
            "id": "stirrup_A",
            "family": "etrie",
            "bar": "10M",
            "description": "Étrier 10M de 3'-0 1/2\" x 4'-8\" @ 12\" c/c",
            "dimensions_ft_in": {"a": "3'-0 1/2\"", "b": "4'-8\""},
            "spacing": "12\" c/c",
            "qty": None,
            "is_dummy_qty": True,
            "unit_length_m": 4.9022,
            "length_method": "perimeter_2*(a+b)",
            "total_length_m": None,
            "is_dummy_length_total": True,
            "kg_per_m": 0.785,
            "total_kg": None,
            "is_dummy_weight": True
        },
        {
            "id": "stirrup_B",
            "family": "etrie",
            "bar": "10M",
            "description": "Étrier 10M de 11'-4\" x 3'-0 1/2\" @ 12\" c/c",
            "dimensions_ft_in": {"a": "11'-4\"", "b": "3'-0 1/2\""},
            "spacing": "12\" c/c",
            "qty": None,
            "is_dummy_qty": True,
            "unit_length_m": 8.7376,
            "length_method": "perimeter_2*(a+b)",
            "total_length_m": None,
            "is_dummy_length_total": True,
            "kg_per_m": 0.785,
            "total_kg": None,
            "is_dummy_weight": True
        },
        {
            "id": "L_shape_12",
            "family": "L_shape",
            "bar": "15M",
            "description": "12 - 15M en forme de \"L\" de 7'-0\" x 1'-4\"",
            "dimensions_ft_in": {"a": "7'-0\"", "b": "1'-4\""},
            "spacing": None,
            "qty": 12,
            "is_dummy_qty": False,
            "unit_length_m": 2.54,
            "length_method": "a+b",
            "total_length_m": 30.48,
            "is_dummy_length_total": False,
            "kg_per_m": 1.57,
            "total_kg": 47.8536,
            "is_dummy_weight": False
        },
        {
            "id": "L_shape_24",
            "family": "L_shape",
            "bar": "15M",
            "description": "24 - 15M en forme de \"L\" de 7'-0\" x 1'-4\"",
            "dimensions_ft_in": {"a": "7'-0\"", "b": "1'-4\""},
            "spacing": None,
            "qty": 24,
            "is_dummy_qty": False,
            "unit_length_m": 2.54,
            "length_method": "a+b",
            "total_length_m": 60.96,
            "is_dummy_length_total": False,
            "kg_per_m": 1.57,
            "total_kg": 95.7072,
            "is_dummy_weight": False
        },
        {
            "id": "longitudinal_15M",
            "family": "longitudinal",
            "bar": "15M",
            "description": "15M @ 12\" c/c (longueur totale non donnée sur la vue)",
            "dimensions_ft_in": None,
            "spacing": "12\" c/c",
            "qty": None,
            "is_dummy_qty": True,
            "unit_length_m": None,
            "length_method": "unknown_total_length",
            "total_length_m": None,
            "is_dummy_length_total": True,
            "kg_per_m": 1.57,
            "total_kg": None,
            "is_dummy_weight": True
        },
        {
            "id": "anchorage_8",
            "family": "anchorage",
            "bar": None,
            "description": "8 Ancrages à béton",
            "dimensions_ft_in": None,
            "spacing": None,
            "qty": 8,
            "is_dummy_qty": False,
            "unit_length_m": None,
            "length_method": "not_applicable",
            "total_length_m": None,
            "is_dummy_length_total": True,
            "kg_per_m": None,
            "total_kg": None,
            "is_dummy_weight": True
        }
    ],
    "totals": {
        "known_total_length_m": 91.44,
        "known_total_kg": 143.5608,
        "unknown_items": ["stirrup_A", "stirrup_B", "longitudinal_15M", "anchorage_8"],
        "cost_formula": {
            "if_price_is_CAD_per_kg": "total_cost = known_total_kg * price_per_kg",
            "if_price_is_CAD_per_tonne": "total_cost = (known_total_kg/1000) * price_per_tonne"
        }
    }
}


# ============================================================
# 2) HELPERS
# ============================================================

def ground_truth_payload() -> dict:
    """Return a deep copy so UI manipulations never mutate the constant."""
    return deepcopy(GROUND_TRUTH)


def to_dataframe(line_items: list) -> pd.DataFrame:
    """Flatten line_items to a user-friendly table."""
    rows = []
    for it in line_items:
        dims = it.get("dimensions_ft_in") or {}
        rows.append({
            "id": it.get("id"),
            "type": it.get("family"),
            "bar": it.get("bar"),
            "description": it.get("description"),
            "a (ft-in)": dims.get("a"),
            "b (ft-in)": dims.get("b"),
            "spacing": it.get("spacing"),
            "qty": it.get("qty"),
            "unit_length_m": it.get("unit_length_m"),
            "total_length_m": it.get("total_length_m"),
            "kg_per_m": it.get("kg_per_m"),
            "total_kg": it.get("total_kg"),
            "dummy?": (
                bool(it.get("is_dummy_qty"))
                or bool(it.get("is_dummy_length_total"))
                or bool(it.get("is_dummy_weight"))
            )
        })
    return pd.DataFrame(rows)


def compute_cost(known_total_kg: float, price_value: float, price_unit: str) -> float:
    """Compute cost from known_total_kg."""
    if price_unit == "CAD_per_kg":
        return known_total_kg * price_value
    if price_unit == "CAD_per_tonne":
        return (known_total_kg / 1000.0) * price_value
    raise ValueError(f"Unknown price_unit: {price_unit}")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")


def json_to_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def is_probably_pdf(uploaded_file) -> tuple[bool, str]:
    """
    Checks extension + mime type when available.
    Streamlit filters by extension, but we keep a guard for robustness.
    """
    name = (uploaded_file.name or "").lower()
    ext_ok = name.endswith(".pdf")
    mime = getattr(uploaded_file, "type", None)  # often "application/pdf"
    mime_ok = (mime == "application/pdf") if mime is not None else True
    ok = ext_ok and mime_ok
    reason = []
    if not ext_ok:
        reason.append("Extension != .pdf")
    if not mime_ok:
        reason.append(f"MIME inattendu: {mime}")
    return ok, " | ".join(reason)


def base_name_without_pdf(filename: str) -> str:
    """Returns filename without trailing .pdf (case-insensitive)."""
    base = os.path.basename(filename)
    if base.lower().endswith(".pdf"):
        return base[:-4]
    return os.path.splitext(base)[0]


def file_signature(uploaded_file) -> str:
    """
    Create a lightweight signature for 'new file uploaded' detection.
    We combine name + size + first bytes hash.
    """
    # Beware: uploaded_file is a SpooledTemporaryFile-like object
    # We'll read a small prefix then reset pointer.
    pos = uploaded_file.tell()
    uploaded_file.seek(0)
    prefix = uploaded_file.read(4096)
    uploaded_file.seek(pos)
    h = hashlib.sha256(prefix).hexdigest()[:12]
    size = getattr(uploaded_file, "size", None)
    return f"{uploaded_file.name}|{size}|{h}"


def simulate_analysis(min_seconds: float = 1.2, max_seconds: float = 2.2) -> None:
    """
    Simulate a believable analysis delay with progress bar.
    Deterministic-ish duration based on time to feel real, but stable enough.
    """
    # simple fixed duration (you can randomize if you want)
    duration = (min_seconds + max_seconds) / 2.0
    steps = 30
    progress = st.progress(0, text="Analyse en cours…")
    for i in range(steps):
        time.sleep(duration / steps)
        progress.progress(int((i + 1) * 100 / steps), text="Analyse en cours…")
    progress.empty()


def reset_app():
    """Reset session state keys used by the app."""
    for k in ["analysis_ready", "last_file_sig"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# ============================================================
# 3) STREAMLIT APP
# ============================================================

st.set_page_config(page_title="AcierUnited - Prototype O2", layout="wide")

# Minimal header
st.title("AcierUnited — Prototype client (O2)")
st.caption("Prototype") # résultats simulés (O1 hard-codé), UX réaliste pour la démo.

# Initialize session state
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "last_file_sig" not in st.session_state:
    st.session_state.last_file_sig = None

# Top actions
top_left, top_right = st.columns([3, 1])
with top_right:
    st.button("🏠 Retour accueil / Reset", on_click=reset_app, use_container_width=True)

st.divider()

# 1) Upload required
uploaded = st.file_uploader("1) Uploader un fichier PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload un PDF pour lancer l’analyse et afficher les résultats.")
    st.stop()

ok_pdf, reason = is_probably_pdf(uploaded)
if not ok_pdf:
    st.error(f"Le fichier uploadé ne semble pas être un PDF valide. {reason}".strip())
    st.stop()

# Detect new upload (re-upload or different file)
sig = file_signature(uploaded)
new_file = (st.session_state.last_file_sig != sig)
if new_file:
    st.session_state.last_file_sig = sig
    st.session_state.analysis_ready = False

# Optional: show file info
st.success(f"PDF reçu : {uploaded.name}  |  Taille: {getattr(uploaded, 'size', 'unknown')} bytes")

# 2) Simulate analysis when needed
if not st.session_state.analysis_ready:
    with st.spinner("Analyse du plan…"):
        simulate_analysis()
    st.session_state.analysis_ready = True
    st.rerun()

# From here: analysis ready
export_base = base_name_without_pdf(uploaded.name)
payload = ground_truth_payload()
df = to_dataframe(payload["line_items"])

# Layout
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("2) Prix du fer")

    # These widgets will trigger rerun automatically, so cost updates instantly.
    price_unit = st.selectbox(
        "Unité de prix",
        options=["CAD_per_kg", "CAD_per_tonne"],
        index=0,
        key="price_unit"
    )
    price_value = st.number_input(
        "Valeur",
        min_value=0.0,
        value=2.50,
        step=0.10,
        key="price_value",
        help="Ex: 2.50 CAD/kg ou 2500 CAD/tonne"
    )

    known_total_kg = float(payload["totals"]["known_total_kg"])
    known_total_length_m = float(payload["totals"]["known_total_length_m"])
    total_cost = compute_cost(known_total_kg, float(price_value), price_unit)

    st.subheader("3) Totaux (connus en prototype)")
    st.metric("Poids total connu (kg)", f"{known_total_kg:.4f}")
    st.metric("Longueur totale connue (m)", f"{known_total_length_m:.2f}")
    st.metric("Coût estimé (sur poids connu)", f"{total_cost:,.2f} CAD")

    st.caption("Le coût se met à jour automatiquement quand tu changes le prix.")

    st.info(
        "⚠️ Items incomplets en prototype (dummy/unknown): "
        + ", ".join(payload["totals"]["unknown_items"])
    )

with col_right:
    st.subheader("4) Détails des armatures (table)")
    st.dataframe(df, use_container_width=True)

    st.subheader("5) Exports")
    csv_bytes = df_to_csv_bytes(df)
    json_bytes = json_to_bytes(payload)

    st.download_button(
        label="Télécharger CSV (line_items)",
        data=csv_bytes,
        file_name=f"{export_base}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        label="Télécharger JSON (payload complet)",
        data=json_bytes,
        file_name=f"{export_base}.json",
        mime="application/json",
        use_container_width=True
    )

    with st.expander("Voir le JSON (debug)"):
        st.json(payload)
