"""
AcierUnited — Estimateur du prix d’armatures à partir d’un plan
--------------------------------------------------------------
Prototype

Run:
    pip install streamlit pandas
    streamlit run app.py
"""

import os
import io
import json
import time
import base64
import hashlib
from copy import deepcopy
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

import fitz  # PyMuPDF

# --- tes modules ---
from ai import pdf_to_report_AI, EXAMPLE_PDF, USER_PROMPT, extract_rebar_text_candidates
from report_to_cost import compute_o2, round_for_display



# ============================================================
# Helpers UI
# ============================================================

def simulate_analysis(seconds: float = 1.8) -> None:
    steps = 30
    progress = st.progress(0, text="Analyse en cours…")
    for i in range(steps):
        time.sleep(seconds / steps)
        progress.progress(int((i + 1) * 100 / steps), text="Analyse en cours…")
    progress.empty()

def reset_app():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def base_name_without_pdf(filename: str) -> str:
    base = os.path.basename(filename)
    if base.lower().endswith(".pdf"):
        return base[:-4]
    return os.path.splitext(base)[0]

def file_signature(uploaded_file) -> str:
    pos = uploaded_file.tell()
    uploaded_file.seek(0)
    prefix = uploaded_file.read(4096)
    uploaded_file.seek(pos)
    h = hashlib.sha256(prefix).hexdigest()[:12]
    size = getattr(uploaded_file, "size", None)
    return f"{uploaded_file.name}|{size}|{h}"

# def render_pdf_bytes(pdf_bytes: bytes, height: int = 650) -> None:
#     """Affiche un PDF via iframe base64 (souvent le plus compatible)."""
#     if not pdf_bytes:
#         st.warning("PDF vide.")
#         return
#     b64 = base64.b64encode(pdf_bytes).decode("utf-8")
#     html = f"""
#     <iframe
#         src="data:application/pdf;base64,{b64}"
#         width="100%"
#         height="{height}"
#         style="border: 1px solid #ddd; border-radius: 8px;"
#         type="application/pdf"
#     ></iframe>
#     """
#     st.components.v1.html(html, height=height, scrolling=True)

# def safe_json_download(data: Any, filename: str, label: str):
#     st.download_button(
#         label=label,
#         data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
#         file_name=filename,
#         mime="application/json",
#         use_container_width=True,
#     )

def render_pdf_preview(pdf_bytes: bytes, max_pages: int = 2, zoom: float = 1.5) -> None:
    """
    Preview stable (Chrome-safe): rend les premières pages en images.
    Nécessite pymupdf: pip install pymupdf
    """
    if not pdf_bytes:
        st.warning("PDF vide.")
        return

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = min(len(doc), max_pages)
    st.caption(f"Aperçu : {n} page(s) sur {len(doc)}")

    for i in range(n):
        page = doc[i]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, caption=f"Page {i+1}", use_container_width=True)

def df_downloads(df: pd.DataFrame, base: str, prefix: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Télécharger CSV — {prefix}",
        data=csv,
        file_name=f"{base}_{prefix}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# Convert report <-> editable dataframe (Fix A)
# ============================================================

EDITABLE_COLS = [
    "item_id", "diameter", "shape", "count",
    "legs_ft_in", "stirrup_dims_ft_in", "explicit_length_ft_in",
    "spacing_raw", "dimensions_raw",
    "notes",
]

def items_to_editable_df(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append({
            "item_id": it.get("item_id"),
            "diameter": it.get("diameter"),
            "shape": it.get("shape"),
            "count": None if it.get("count") == "NAN" else it.get("count"),
            "legs_ft_in": json.dumps(it.get("legs_ft_in"), ensure_ascii=False) if it.get("legs_ft_in") else "",
            "stirrup_dims_ft_in": json.dumps(it.get("stirrup_dims_ft_in"), ensure_ascii=False) if it.get("stirrup_dims_ft_in") else "",
            "explicit_length_ft_in": it.get("explicit_length_ft_in") or "",
            "spacing_raw": it.get("spacing_raw") or "",
            "dimensions_raw": it.get("dimensions_raw") or "",
            "notes": it.get("notes") or "",
        })

    df = pd.DataFrame(rows)
    # Important: count nullable integer => compatible avec NumberColumn
    if "count" in df.columns:
        df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
    return df[EDITABLE_COLS]

def editable_df_to_items(df: pd.DataFrame, original_items: list[dict]) -> list[dict]:
    """
    Reconstruit items en conservant les champs non-édités.
    """
    out = []
    orig_by_id = {it.get("item_id"): deepcopy(it) for it in original_items}

    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        it = orig_by_id.get(item_id, {"item_id": item_id})

        it["diameter"] = row["diameter"]
        it["shape"] = row["shape"]

        # count: Int64 nullable -> "NAN" si NA
        if pd.isna(row["count"]):
            it["count"] = "NAN"
        else:
            it["count"] = int(row["count"])

        # listes en JSON (si l’utilisateur modifie)
        def parse_json_list(s: str):
            s = (s or "").strip()
            if not s:
                return None
            try:
                v = json.loads(s)
                return v
            except Exception:
                # si l’utilisateur écrit "7'-0\", 1'-4\"" → fallback simple
                return [x.strip() for x in s.split(",") if x.strip()]

        it["legs_ft_in"] = parse_json_list(row["legs_ft_in"]) if row["shape"] == "L" else None
        it["stirrup_dims_ft_in"] = parse_json_list(row["stirrup_dims_ft_in"]) if row["shape"] == "stirrup" else None
        it["explicit_length_ft_in"] = (row["explicit_length_ft_in"] or "").strip() or None

        it["spacing_raw"] = (row["spacing_raw"] or "").strip() or None
        it["dimensions_raw"] = (row["dimensions_raw"] or "").strip() or None
        it["notes"] = row["notes"] or ""

        out.append(it)

    return out


# ============================================================
# Streamlit app
# ============================================================

st.set_page_config(page_title="AcierUnited", layout="wide")

st.title("Estimateur du prix d’armatures à partir d’un plan")
st.caption("Prototype")

top_left, top_right = st.columns([3, 1])
with top_right:
    st.button("🏠 Retour accueil / Reset", on_click=reset_app, use_container_width=True)

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

st.success(f"PDF reçu : {uploaded.name} | Taille: {getattr(uploaded, 'size', 'unknown')} bytes")

# Lire bytes du PDF uploadé
uploaded.seek(0)
target_pdf_bytes = uploaded.read()

# --- Expanders (doivent être AVANT tout crash) ---
with st.expander("📄 Voir le PDF EXEMPLE (in-context)", expanded=False):
    try:
        with open(EXAMPLE_PDF, "rb") as f:
            example_bytes = f.read()
        st.caption(f"PDF EXEMPLE : {os.path.basename(EXAMPLE_PDF)}")
        #render_pdf_bytes(example_bytes, height=650)
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
    #render_pdf_bytes(target_pdf_bytes, height=650)
    render_pdf_preview(target_pdf_bytes, max_pages=2)
    st.download_button(
        "Télécharger le PDF CIBLE",
        data=target_pdf_bytes,
        file_name=uploaded.name,
        mime="application/pdf",
        use_container_width=True,
    )

# Prompt éditable (USER_PROMPT optionnel)
with st.expander("🧠 Voir / modifier le prompt (USER_PROMPT)", expanded=False):
    st.caption("Par défaut, l’app utilise le prompt LONG auto-généré par ai.USER_PROMPT(candidats). "
               "Si tu écris ici, ton texte remplace ce prompt (tu peux coller au complet).")
    user_prompt_override = st.text_area(
        "USER_PROMPT (FR) — optionnel",
        value=st.session_state.get("user_prompt_override", ""),
        height=260,
    )
    st.session_state["user_prompt_override"] = user_prompt_override

st.divider()

# ============================================================
# Appel IA (O1)
# ============================================================

if st.session_state.get("report") is None:
    with st.spinner("Analyse du plan…"):
        simulate_analysis(1.8)

        # IMPORTANT: pdf_to_report_AI attend un chemin, donc on écrit le upload dans un temp file
        # (Streamlit ne donne pas un chemin disque stable)
        tmp_dir = ".streamlit_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        target_path = os.path.join(tmp_dir, uploaded.name)
        with open(target_path, "wb") as f:
            f.write(target_pdf_bytes)

        try:
            report = pdf_to_report_AI(
                target_pdf=target_path,
                user_prompt=(user_prompt_override.strip() or None),
            )
            st.session_state["report"] = report
        except Exception as e:
            st.error("Erreur pendant l’appel IA (O1).")
            st.exception(e)
            st.stop()

report = st.session_state["report"]

# Cas doc_not_plan / plan_no_rebar
doc_status = report.get("doc_status")
if doc_status == "doc_not_plan":
    st.error("Le document ne ressemble pas à un plan de structure (doc_not_plan).")
    with st.expander("Voir le JSON (debug)"):
        st.json(report)
    st.stop()

if doc_status == "plan_no_rebar":
    st.warning("Plan détecté, mais aucune armature n’a été trouvée (plan_no_rebar).")
    with st.expander("Voir le JSON (debug)"):
        st.json(report)
    st.stop()

# ============================================================
# O1 — affichage + corrections
# ============================================================

st.subheader("O1 — Armatures détectées (corrections possibles)")

items = report.get("items", [])
if not items:
    st.info("Aucune armature détectée (items vide).")
else:
    df_o1 = items_to_editable_df(items)

    # ✅ Fix de ton erreur: count = NumberColumn (pas TextColumn)
    edited_df = st.data_editor(
        df_o1,
        use_container_width=True,
        hide_index=True,
        column_config={
            "item_id": st.column_config.NumberColumn("item_id", disabled=True),
            "count": st.column_config.NumberColumn("count", step=1, min_value=0),
            "legs_ft_in": st.column_config.TextColumn("legs_ft_in (JSON liste)"),
            "stirrup_dims_ft_in": st.column_config.TextColumn("stirrup_dims_ft_in (JSON liste)"),
            "notes": st.column_config.TextColumn("notes", width="large"),
        },
        disabled=["item_id"],  # item_id non modifiable
    )

    # reconstruire items corrigés
    corrected_items = editable_df_to_items(edited_df, items)
    report_corrected = deepcopy(report)
    report_corrected["items"] = corrected_items

    with st.expander("Voir le JSON O1 (debug)"):
        st.json(report_corrected)

    safe_json_download(report_corrected, f"{base_name_without_pdf(uploaded.name)}_O1.json", "Télécharger JSON — O1")

# ============================================================
# O2 — prix + calcul
# ============================================================

st.divider()
st.subheader("O2 — Estimation du coût")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    price_unit = st.selectbox("Unité de prix", ["CAD_per_kg", "CAD_per_tonne"], index=0)
    price_value = st.number_input("Valeur", min_value=0.0, value=2.50, step=0.10)

    if st.button("Estimer le coût", type="primary", use_container_width=True):
        # Construire price_per_kg uniforme
        if price_unit == "CAD_per_kg":
            price_per_kg = {"10M": float(price_value), "15M": float(price_value), "20M": float(price_value), "25M": float(price_value)}
        else:
            # CAD/tonne -> CAD/kg
            price_per_kg_val = float(price_value) / 1000.0
            price_per_kg = {"10M": price_per_kg_val, "15M": price_per_kg_val, "20M": price_per_kg_val, "25M": price_per_kg_val}

        o2 = compute_o2(report_corrected if items else report, price_per_kg=price_per_kg)
        o2 = round_for_display(o2, nd_m=2, nd_kg=2, nd_cost=2)
        st.session_state["o2_result"] = o2

with col_right:
    o2 = st.session_state.get("o2_result")
    if o2 is None:
        st.info("Entre un prix puis clique sur « Estimer le coût ».")
    else:
        # tableaux O2
        df_o2 = pd.DataFrame(o2.get("o2_table", []))
        st.write("### Détails O2 (par item)")
        st.dataframe(df_o2, use_container_width=True, hide_index=True)

        # Totaux par diamètre
        totals = o2.get("o2_totals_by_diameter", {})
        df_totals = pd.DataFrame([
            {"diameter": k, **v} for k, v in totals.items()
        ])
        st.write("### Totaux par diamètre")
        st.dataframe(df_totals, use_container_width=True, hide_index=True)

        # Grand total
        gt = o2.get("o2_grand_totals", {})
        st.write("### Total général")
        big = gt.get("cost", "NAN")
        st.metric("Coût total", f"{big} CAD" if big != "NAN" else "NAN")

        # exports
        base = base_name_without_pdf(uploaded.name)
        st.write("### Exports")
        df_downloads(df_o2, base, "O2_table")
        df_downloads(df_totals, base, "O2_totals_by_diameter")
        safe_json_download(o2, f"{base}_O2_full.json", "Télécharger JSON — O2 complet")

        with st.expander("Voir le JSON O2 (debug)"):
            st.json(o2)
