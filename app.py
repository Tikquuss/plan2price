"""
AcierUnited — Estimateur du prix d’armatures à partir d’un plan (Prototype)
--------------------------------------------------------------------------

- Upload d’un PDF CIBLE (obligatoire)
- Affichage (preview images) du PDF EXEMPLE + PDF CIBLE (expanders)
- Génération automatique du prompt LONG (USER_PROMPT(candidats)) à partir du PDF CIBLE
- Possibilité de modifier/override le prompt (expander)
- Appel IA (O1) via ai.pdf_to_report_AI(...)
- Table O1 éditable (corrections manuelles)
- Calcul O2 (longueurs, poids, coût) via report_to_cost.compute_o2(...)
- Export CSV/JSON pour O1 et O2

Run:
    pip install streamlit pandas pymupdf openai pydantic
    streamlit run app.py
"""

import os
import io
import json
import time
import hashlib
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

# =========================
# Imports projet
# =========================
# ai.py doit être dans le même dossier (ou dans PYTHONPATH)
from ai import (
    EXAMPLE_PDF,
    USER_PROMPT,
    extract_rebar_text_candidates,
    pdf_to_report_AI,
)

# report_to_cost.py doit être dans le même dossier (ou dans PYTHONPATH)
from report_to_cost import compute_o2, round_for_display


# =========================
# Helpers UI / IO
# =========================

def file_signature(uploaded_file) -> str:
    """
    Lightweight signature for 'new upload' detection:
    name + size + hash(first 4KB)
    """
    pos = uploaded_file.tell()
    uploaded_file.seek(0)
    prefix = uploaded_file.read(4096)
    uploaded_file.seek(pos)
    h = hashlib.sha256(prefix).hexdigest()[:12]
    size = getattr(uploaded_file, "size", None)
    return f"{uploaded_file.name}|{size}|{h}"


def simulate_analysis(seconds: float = 1.8) -> None:
    """Small progress bar to feel like a real analysis."""
    steps = 24
    progress = st.progress(0, text="Analyse en cours…")
    for i in range(steps):
        time.sleep(seconds / steps)
        progress.progress(int((i + 1) * 100 / steps), text="Analyse en cours…")
    progress.empty()


def base_name_without_pdf(filename: str) -> str:
    """Return filename without trailing .pdf (case-insensitive)."""
    base = os.path.basename(filename)
    if base.lower().endswith(".pdf"):
        return base[:-4]
    return os.path.splitext(base)[0]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")


def json_to_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def render_pdf_preview(pdf_bytes: bytes, max_pages: int = 2, zoom: float = 1.8) -> None:
    """
    Render PDF pages as images (avoids Chrome 'blocked' iframe issue).
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n = min(len(doc), max_pages)
        for i in range(n):
            page = doc[i]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            st.image(pix.tobytes("png"), caption=f"Page {i+1}", use_container_width=True)
        if len(doc) > max_pages:
            st.info(f"Aperçu limité à {max_pages} pages. Télécharge le PDF pour voir le reste.")
    except Exception as e:
        st.error(f"Impossible d’afficher l’aperçu PDF: {e}")


# =========================
# O1 <-> DataFrame helpers
# =========================

O1_EDITABLE_COLUMNS = [
    "item_id",
    "diameter",
    "shape",
    "count",
    "legs_ft_in",
    "stirrup_dims_ft_in",
    "explicit_length_ft_in",
    "spacing_raw",
    "dimensions_raw",
    "notes",
]

def o1_items_to_df(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert report['items'] list into a DataFrame for editing.
    We store list-like fields as JSON strings to edit nicely.
    """
    rows = []
    for it in report.get("items", []):
        legs = it.get("legs_ft_in")
        stir = it.get("stirrup_dims_ft_in")

        rows.append({
            "item_id": it.get("item_id"),
            "diameter": it.get("diameter"),
            "shape": it.get("shape"),
            "count": it.get("count"),

            # store lists as compact JSON strings for edit
            "legs_ft_in": json.dumps(legs, ensure_ascii=False) if isinstance(legs, list) else "",
            "stirrup_dims_ft_in": json.dumps(stir, ensure_ascii=False) if isinstance(stir, list) else "",
            "explicit_length_ft_in": it.get("explicit_length_ft_in") or "",

            "spacing_raw": it.get("spacing_raw") or "",
            "dimensions_raw": it.get("dimensions_raw") or "",
            "notes": it.get("notes") or "",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=O1_EDITABLE_COLUMNS)

    # IMPORTANT: make 'count' an object to allow "NAN" or blank, and avoid Streamlit dtype crash
    if "count" in df.columns:
        df["count"] = df["count"].apply(lambda x: "" if x == "NAN" else x).astype("object")

    return df[O1_EDITABLE_COLUMNS]


def parse_json_list_field(s: str) -> Optional[List[str]]:
    """
    Parse a JSON list string like ["7'-0\"", "1'-4\""].
    Returns None if empty.
    """
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        return None
    return None


def df_to_o1_items(df: pd.DataFrame, original_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply user edits from df back into items.
    We only overwrite editable fields.
    """
    out_items: List[Dict[str, Any]] = []
    # map by item_id for safety
    orig_by_id = {it.get("item_id"): it for it in original_items}

    for _, row in df.iterrows():
        item_id = int(row["item_id"]) if str(row["item_id"]).strip() else None
        if item_id is None or item_id not in orig_by_id:
            continue

        it = dict(orig_by_id[item_id])  # copy original full item

        # diameter/shape
        it["diameter"] = str(row.get("diameter", "")).strip() or it.get("diameter", "other")
        it["shape"] = str(row.get("shape", "")).strip() or it.get("shape", "other")

        # count (allow NAN)
        c = str(row.get("count", "")).strip()
        if c == "" or c.upper() == "NAN":
            it["count"] = "NAN"
        else:
            try:
                it["count"] = int(float(c))
            except Exception:
                it["count"] = "NAN"

        # list fields
        legs = parse_json_list_field(row.get("legs_ft_in", ""))
        stir = parse_json_list_field(row.get("stirrup_dims_ft_in", ""))
        it["legs_ft_in"] = legs
        it["stirrup_dims_ft_in"] = stir

        # explicit length
        el = str(row.get("explicit_length_ft_in", "")).strip()
        it["explicit_length_ft_in"] = el if el else None

        # raw strings
        it["spacing_raw"] = str(row.get("spacing_raw", "")).strip() or None
        it["dimensions_raw"] = str(row.get("dimensions_raw", "")).strip() or None
        it["notes"] = str(row.get("notes", "")).strip() or it.get("notes", "")

        out_items.append(it)

    # keep order by item_id
    out_items.sort(key=lambda x: x.get("item_id", 10**9))
    return out_items


# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="AcierUnited — Estimateur", layout="wide")

st.title("Estimateur du prix d’armatures à partir d’un plan")
st.caption("Prototype")

# session init
if "last_sig" not in st.session_state:
    st.session_state["last_sig"] = None
if "report" not in st.session_state:
    st.session_state["report"] = None
if "o2_result" not in st.session_state:
    st.session_state["o2_result"] = None
if "generated_prompt" not in st.session_state:
    st.session_state["generated_prompt"] = None
if "user_prompt_override" not in st.session_state:
    st.session_state["user_prompt_override"] = ""
if "target_path" not in st.session_state:
    st.session_state["target_path"] = None

# top actions
top_left, top_right = st.columns([3, 1])
with top_right:
    if st.button("🏠 Retour accueil / Reset", use_container_width=True):
        for k in [
            "last_sig",
            "report",
            "o2_result",
            "generated_prompt",
            "user_prompt_override",
            "target_path",
        ]:
            st.session_state.pop(k, None)
        st.rerun()

st.divider()

uploaded = st.file_uploader("Uploader un fichier PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload un PDF pour lancer l’analyse et afficher les résultats.")
    st.stop()

sig = file_signature(uploaded)
new_file = (st.session_state.get("last_sig") != sig)

if new_file:
    st.session_state["last_sig"] = sig
    st.session_state.pop("report", None)
    st.session_state.pop("o2_result", None)
    st.session_state.pop("generated_prompt", None)
    # optionnel: reset l’override quand on change de PDF
    st.session_state["user_prompt_override"] = ""
    st.session_state["target_path"] = None

st.success(f"PDF reçu : {uploaded.name} | Taille: {getattr(uploaded, 'size', 'unknown')} bytes")

# Read bytes (Streamlit has no stable disk path)
uploaded.seek(0)
target_pdf_bytes = uploaded.read()

# ============================================================
# Persist target PDF to disk path (needed by extract + ai)
# (we do it early so expanders/prompt can work)
# ============================================================
tmp_dir = ".streamlit_tmp"
os.makedirs(tmp_dir, exist_ok=True)
target_path = os.path.join(tmp_dir, uploaded.name)
with open(target_path, "wb") as f:
    f.write(target_pdf_bytes)
st.session_state["target_path"] = target_path

# ============================================================
# Build generated prompt (once per file)
# ============================================================
if st.session_state.get("generated_prompt") is None:
    try:
        target_candidates = extract_rebar_text_candidates(target_path)
        st.session_state["generated_prompt"] = USER_PROMPT(target_candidates)
    except Exception as e:
        st.error("Impossible de générer le prompt automatiquement (candidats textuels).")
        st.exception(e)
        st.stop()

generated_prompt = st.session_state["generated_prompt"]

# ============================================================
# Expanders (MUST be before any potential crash)
# ============================================================

with st.expander("📄 Voir le PDF EXEMPLE (in-context)", expanded=False):
    try:
        with open(EXAMPLE_PDF, "rb") as f:
            example_bytes = f.read()
        st.caption(f"PDF EXEMPLE : {os.path.basename(EXAMPLE_PDF)}")
        render_pdf_preview(example_bytes, max_pages=2)
        st.download_button(
            "Télécharger le PDF EXEMPLE",
            data=example_bytes,
            file_name=os.path.basename(EXAMPLE_PDF),
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Impossible de charger EXAMPLE_PDF: {e}")

with st.expander("📄 Voir le PDF CIBLE uploadé", expanded=False):
    st.caption(f"PDF CIBLE : {uploaded.name}")
    render_pdf_preview(target_pdf_bytes, max_pages=2)
    st.download_button(
        "Télécharger le PDF CIBLE",
        data=target_pdf_bytes,
        file_name=uploaded.name,
        mime="application/pdf",
        use_container_width=True,
    )

with st.expander("🧠 Voir / modifier le prompt (USER_PROMPT)", expanded=False):
    st.caption(
        "Par défaut, l’app utilise le prompt LONG auto-généré par ai.USER_PROMPT(candidats) "
        "(candidats extraits du PDF CIBLE). "
        "Tu peux le modifier ici : si tu écris, ton texte remplace le prompt généré."
    )

    # display generated prompt (read-only)
    st.text_area(
        "Prompt auto-généré (lecture seule)",
        value=generated_prompt,
        height=220,
        disabled=True,
    )

    user_prompt_override = st.text_area(
        "Override (optionnel) — ce texte remplace le prompt auto-généré (soyez prudent quant à l’usage de cette fonctionnalité)",
        value=st.session_state.get("user_prompt_override", ""),
        height=220,
    )
    st.session_state["user_prompt_override"] = user_prompt_override

st.divider()

# ============================================================
# O1 — Call AI
# ============================================================

if st.session_state.get("report") is None:
    with st.spinner("Analyse du plan…"):
        simulate_analysis(1.8)

        override = (st.session_state.get("user_prompt_override") or "").strip()
        prompt_to_use = override or generated_prompt

        try:
            report = pdf_to_report_AI(
                target_pdf=target_path,
                user_prompt=prompt_to_use,
            )
            st.session_state["report"] = report
        except Exception as e:
            st.error("Erreur pendant l’appel IA (O1).")
            st.exception(e)
            st.stop()

report: Dict[str, Any] = st.session_state["report"]

# Handle doc_status
doc_status = report.get("doc_status", "ok")
if doc_status == "doc_not_plan":
    st.error("Le document ne ressemble pas à un plan de structure (doc_not_plan).")
    with st.expander("Voir le JSON (debug)", expanded=False):
        st.json(report)
    st.stop()

if doc_status == "plan_no_rebar":
    st.warning("Plan détecté, mais aucune armature n’a été trouvée (plan_no_rebar).")
    with st.expander("Voir le JSON (debug)", expanded=False):
        st.json(report)
    # User can still export
    st.download_button(
        "Télécharger JSON — O1",
        data=json_to_bytes(report),
        file_name=f"{base_name_without_pdf(uploaded.name)}_O1.json",
        mime="application/json",
        use_container_width=True,
    )
    st.stop()

# ============================================================
# O1 — Display + editable table
# ============================================================

st.subheader("O1 — Armatures détectées (corrections possibles)")

items = report.get("items", [])
o1_df = o1_items_to_df(report)

# Forcer les "" à "NAN" pour la colonne count
#o1_df["count"] = o1_df["count"].replace({"": "NAN"})

# ✅ Force count to string to avoid Streamlit dtype mismatch
if "count" in o1_df.columns:
    def _count_to_str(x):
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, (int, float)):
            # ex: 12.0 -> "12"
            try:
                return str(int(x))
            except Exception:
                return str(x)
        return str(x)

    o1_df["count"] = o1_df["count"].map(_count_to_str).astype("string")

# column config
colcfg = {
    "item_id": st.column_config.NumberColumn("item_id", disabled=True),
    "diameter": st.column_config.SelectboxColumn("diameter", options=["10M", "15M", "20M", "25M", "other"]),
    "shape": st.column_config.SelectboxColumn("shape", options=["L", "stirrup", "straight", "other"]),
    # count is text to allow NAN / blank without dtype crash
    "count": st.column_config.TextColumn("count (int ou NAN)", help="Ex: 12, 6, ou NAN"),
    "legs_ft_in": st.column_config.TextColumn("legs_ft_in (JSON liste)", help='Ex: ["7\'-0\\"", "1\'-4\\""]'),
    "stirrup_dims_ft_in": st.column_config.TextColumn("stirrup_dims_ft_in (JSON liste)", help='Ex: ["3\'-0 1/2\\"", "4\'-8\\""]'),
    "explicit_length_ft_in": st.column_config.TextColumn("explicit_length_ft_in"),
    "spacing_raw": st.column_config.TextColumn("spacing_raw"),
    "dimensions_raw": st.column_config.TextColumn("dimensions_raw"),
    "notes": st.column_config.TextColumn("notes"),
}

edited_df = st.data_editor(
    o1_df,
    use_container_width=True,
    hide_index=True,
    column_config=colcfg,
    disabled=["item_id"],  # keep ids stable
    num_rows="fixed",
    key="o1_editor",
)

# apply edits back to report for O2
if st.button("✅ Appliquer les corrections O1 (prend le tableau édité par l’utilisateur et le réinjecte dans le report IA)", use_container_width=False):
    try:
        corrected_items = df_to_o1_items(edited_df, original_items=items)
        report["items"] = corrected_items
        st.session_state["report"] = report
        st.success("Corrections appliquées. Tu peux maintenant estimer le coût (O2).")
        # clear any previous O2 result
        st.session_state.pop("o2_result", None)
        st.rerun()
    except Exception as e:
        st.error("Impossible d’appliquer les corrections (format invalide dans une colonne).")
        st.exception(e)

# Exports O1
o1_export_base = base_name_without_pdf(uploaded.name)

with st.expander("Voir le JSON O1 (debug)", expanded=False):
    st.json(report)

o1_csv = df_to_csv_bytes(edited_df)
st.download_button(
    "Télécharger CSV — O1",
    data=o1_csv,
    file_name=f"{o1_export_base}_O1.csv",
    mime="text/csv",
    use_container_width=True,
)
st.download_button(
    "Télécharger JSON — O1",
    data=json_to_bytes(report),
    file_name=f"{o1_export_base}_O1.json",
    mime="application/json",
    use_container_width=True,
)

st.divider()

# ============================================================
# O2 — Pricing + compute button
# ============================================================

st.subheader("O2 — Estimation du coût")

left, right = st.columns([1, 2], gap="large")

with left:
    st.write("**Unité de prix**")
    price_unit = st.selectbox("Unité", options=["CAD_per_kg", "CAD_per_tonne"], index=0)

    price_value = st.number_input(
        "Valeur",
        min_value=0.0,
        value=2.50,
        step=0.10,
        help="Ex: 2.50 CAD/kg ou 2500 CAD/tonne",
    )

    if st.button("Estimer le coût", use_container_width=True):
        # build price_per_kg dict for all diameters found
        # if tonne: convert to /kg
        unit_price_per_kg = float(price_value) if price_unit == "CAD_per_kg" else float(price_value) / 1000.0

        # apply same price to all known diameters; 'other' left absent => NAN costs there
        price_per_kg = {d: unit_price_per_kg for d in ["10M", "15M", "20M", "25M"]}

        try:
            o2_result = compute_o2(report, price_per_kg=price_per_kg)
            o2_result = round_for_display(o2_result, nd_m=2, nd_kg=2, nd_cost=2)
            st.session_state["o2_result"] = o2_result
            st.success("O2 calculé.")
        except Exception as e:
            st.error("Erreur pendant le calcul O2.")
            st.exception(e)

with right:
    st.write("**Détails O2 (par item)**")
    o2 = st.session_state.get("o2_result")

    if o2 is None:
        st.info("Clique sur **Estimer le coût** pour générer O2.")
    else:
        # table O2
        o2_table = o2.get("o2_table", [])
        o2_df = pd.DataFrame(o2_table)

        if not o2_df.empty:
            st.dataframe(o2_df, use_container_width=True, hide_index=True)
        else:
            st.warning("O2 table vide.")

        # totals
        totals_by_d = o2.get("o2_totals_by_diameter", {})
        gt = o2.get("o2_grand_totals", {})

        st.write("**Totaux par diamètre**")
        totals_df = pd.DataFrame([
            {"diameter": d, **vals} for d, vals in totals_by_d.items()
        ])
        if not totals_df.empty:
            st.dataframe(totals_df, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            ### Total général
            - **Longueur totale (m)** : {gt.get("total_m")}
            - **Poids total (kg)** : {gt.get("weight_kg")}
            - **Coût total** : {gt.get("cost")}
            """
        )

        # Exports O2
        st.subheader("Exports O2")
        st.download_button(
            "Télécharger CSV — O2 (table)",
            data=df_to_csv_bytes(o2_df) if not o2_df.empty else b"",
            file_name=f"{o1_export_base}_O2_table.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Télécharger JSON — O2 (payload complet)",
            data=json_to_bytes(o2),
            file_name=f"{o1_export_base}_O2.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("Voir le JSON O2 (debug)", expanded=False):
            st.json(o2)

# show warnings if any
warnings = report.get("warnings", [])
if warnings:
    with st.expander("⚠️ Warnings", expanded=False):
        for w in warnings:
            st.warning(w)
