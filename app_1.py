"""
AcierUnited — Estimateur du prix d’armatures à partir d’un plan (Prototype)
---------------------------------------------------------------------------
Objectif:
- Upload PDF (obligatoire)
- Latence réaliste (spinner/progress) à chaque upload/re-upload
- Afficher le PDF EXEMPLE + le USER_PROMPT (modifiable) en expander
- Appeler une IA (placeholder): report = pdf_to_report_AI(target_pdf, user_prompt)
- Afficher le report (modifiable minimalement)
- Entrée prix (CAD/kg ou CAD/tonne) + bouton "Estimer le coût"
- Calcul O2 via compute_o2(report, price_per_kg={...})
- Afficher un tableau récapitulatif + total en grande police
- Exports CSV/JSON (au choix) + JSON debug
- Bouton Reset / Retour accueil

Run:
    pip install streamlit pandas pymupdf
    streamlit run app.py
"""

import io
import os
import re
import json
import time
import base64
import hashlib
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai import pdf_to_report_AI  


# ============================================================
# 0) CONFIG
# ============================================================

# IMPORTANT:
# - EXAMPLE_PDF: chemin local (dans ton repo / serveur / machine)
# - En prod, tu le mettras dans un dossier static.
EXAMPLE_PDF = "Plan_Structure_Base_St-Roch.pdf"  # <-- adapte le chemin si besoin

DEFAULT_USER_PROMPT_FR = """\
Vous êtes un estimateur professionnel spécialisé dans la lecture de plans de structure
et le calcul des armatures en acier (rebar).

Votre tâche consiste à analyser un plan de structure (PDF) et à produire un tableau
des armatures contenant, pour chaque groupe :
- le diamètre,
- la forme,
- la quantité,
- la longueur unitaire (définie par ses dimensions et sa formule de calcul).

Vous recevez :
- un PDF d’exemple (Plan_Structure_Base_St-Roch), utilisé strictement comme démonstration in-context,
- une description détaillée de la méthode appliquée sur cet exemple,
- un PDF cible, sur lequel vous devez appliquer exactement la même méthode.

La sortie doit concerner UNIQUEMENT le document cible.
Aucun texte hors du JSON demandé.

====================================================
MÉTHODE GÉNÉRALE (OBLIGATOIRE)
====================================================

1. Ouvrir le document et l’analyser page par page.

2. Sur chaque page, identifier UNIQUEMENT les plans d’armatures.
   Ces plans sont généralement caractérisés par :
   - des titres contenant des termes comme « armatures », « détail des armatures »,
   - la présence de notations telles que : 10M, 15M, 20M, « @ 12" c/c », « Étrier », etc.

3. Ignorer les vues qui ne contiennent aucune information d’armature
   (par exemple : géométrie du béton sans ferraillage).

4. Pour chaque plan d’armatures :
   - identifier les différentes vues (plan, élévation, profil),
   - repérer les groupes d’armatures,
   - extraire le diamètre (10M / 15M / …),
   - identifier la forme :
        • barre droite,
        • barre en L,
        • étrier,
   - extraire les dimensions écrites lorsqu’elles existent,
   - déterminer la formule de calcul de la longueur unitaire,
   - déterminer la quantité selon les règles ci-dessous.

5. Éviter absolument les doubles comptages :
   - un même groupe d’armatures peut apparaître sur plusieurs vues (plan / élévation / profil),
   - s’il s’agit du même groupe physique, il ne doit être compté qu’une seule fois,
   - compter séparément uniquement lorsque la géométrie indique clairement
     des groupes distincts.

====================================================
RÈGLES DE CALCUL DES LONGUEURS
====================================================

IMPORTANT :
Le calcul arithmétique final des longueurs NE DOIT PAS être effectué.
Vous devez uniquement extraire les dimensions et indiquer la formule.
Le calcul numérique sera effectué par l’application.

Conventions à appliquer :

1. Barres en forme de L :
   - longueur unitaire = somme des deux jambes.
   - fournir legs_ft_in = [jambre1, jambe2]
   - formula_hint = "a + b"

2. Étriers :
   - longueur unitaire = 2 × (dimension A + dimension B).
   - fournir stirrup_dims_ft_in = [dimA, dimB]
   - formula_hint = "2*(a + b)"

3. Barres droites :
   - si la longueur est explicitement indiquée, fournir explicit_length_ft_in,
   - sinon, déduire la longueur à partir de la géométrie visible sur d’autres vues (sans calcul final),
     et expliquer la déduction dans notes (ex: "4'-8 + 2*1'-4", etc.).

====================================================
RÈGLES DE DÉTERMINATION DES QUANTITÉS
====================================================

1. Si la quantité est explicitement indiquée dans le libellé
   (exemple : « 12-15M ») :
   - utiliser directement cette valeur.

2. Si la quantité n’est PAS indiquée :
   a. tenter un comptage visuel :
      - compter les barres, points ou répétitions visibles sur les vues,
   b. si le comptage visuel n’est pas fiable :
      - estimer à partir de l’espacement :
        nombre ≈ L / E + 1
        où :
        - E est l’espacement (ex. 12" c/c),
        - L est une dimension pertinente déduite de la géométrie,
   c. si aucune méthode fiable n’est possible :
      - indiquer la quantité comme "NAN" et expliquer la raison.

====================================================
CAS PARTICULIERS
====================================================

- Si le document analysé ne correspond pas à un plan de structure :
  retourner doc_status = "doc_not_plan".

- Si le document est un plan de structure mais ne contient aucune armature :
  retourner doc_status = "plan_no_rebar" et une liste items vide.

====================================================
DÉMONSTRATION IN-CONTEXT (Plan_Structure_Base_St-Roch)
====================================================

Le document d’exemple Plan_Structure_Base_St-Roch contient une seule page.

Sur cette page :
- la partie gauche contient des vues de la base de béton sans armatures,
- la partie droite contient trois vues représentant le même objet :
    (3) vue en plan du détail des armatures,
    (4) vue en élévation du détail des armatures,
    (5) vue de profil du détail des armatures.

Ces trois vues décrivent le même système d’armatures et doivent être analysées conjointement.

La méthode suivante a été appliquée :

(A) BARRES 15M EN FORME DE "L"
Libellé :
"12-15M en forme de 'L' de 7'-0\" x 1'-4\""
- Nombre = 12 (écrit)
- Longueur unitaire = 7'-0\" + 1'-4\" (somme des jambes)
- Ce groupe apparaît sur les vues (3), (4) et (5) -> ne pas recompter
- Il y a deux occurrences distinctes -> deux items séparés

(B) ÉTRIERS 10M (PETITS)
Libellé :
"Étrier en 10M de 3'-0 1/2\" x 4'8\" @ 12\" c/c"
- Longueur unitaire = 2*(a+b)
- Nombre non écrit -> comptage visuel sur (4) : 6
- Estimation alternative H/E+1 avec H=7'-0\" et E=12\" -> ~8
- Deux groupes identiques -> deux items

(C) ÉTRIERS 10M (GRANDS)
Libellé :
"Étrier en 10M de 11'-4\" x 3'-0 1/2\" @ 12\" c/c"
- Longueur unitaire = 2*(a+b)
- Nombre compté sur (4) : 6

(D) BARRES 15M @ 12\" c/c (VUE 4)
Libellé :
"15M @ 12\" c/c"
- Longueur déduite : 4'-8\" + 2*1'-4\" = 7'-4\" (sans calcul final dans la sortie)
- Nombre compté : 14
- Estimation alternative : L/E+1 avec L≈14'-0\" -> ~15

(E) BARRES 15M @ 12\" c/c (VUE 5)
- Longueur déduite : 11'-4\" + 2*1'-4\" = 14'-0\"
- Nombre compté : 8
- Estimation alternative : l/E+1 avec l≈7'-4\" -> ~8

TABLEAU FINAL (résumé):
1) 15M, 12, 8'-4"
2) 15M, 12, 8'-4"
3) 10M, 6, 15'-5"
4) 10M, 6, 15'-5"
5) 10M, 6, 28'-9"
6) 15M, 14, 7'-4"
7) 15M, 8, 14'-0"

====================================================
TÂCHE SUR LE DOCUMENT CIBLE
====================================================

Appliquer exactement la même méthode au document cible.
Retourner UNIQUEMENT le JSON final correspondant au document cible.
"""


# ============================================================
# 1) HELPERS UI
# ============================================================

def is_probably_pdf(uploaded_file) -> Tuple[bool, str]:
    name = (uploaded_file.name or "").lower()
    ext_ok = name.endswith(".pdf")
    mime = getattr(uploaded_file, "type", None)
    mime_ok = (mime == "application/pdf") if mime is not None else True
    ok = ext_ok and mime_ok
    reason = []
    if not ext_ok:
        reason.append("Extension != .pdf")
    if not mime_ok:
        reason.append(f"MIME inattendu: {mime}")
    return ok, " | ".join(reason)

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

def simulate_analysis(min_seconds: float = 1.2, max_seconds: float = 2.2) -> None:
    duration = (min_seconds + max_seconds) / 2.0
    steps = 30
    progress = st.progress(0, text="Analyse en cours…")
    for i in range(steps):
        time.sleep(duration / steps)
        progress.progress(int((i + 1) * 100 / steps), text="Analyse en cours…")
    progress.empty()

def reset_app():
    for k in [
        "analysis_ready", "last_file_sig",
        "report", "report_edit_df",
        "o2_result", "o2_result_rounded",
        "price_unit", "price_value",
        "user_prompt",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def render_pdf_in_expander(pdf_bytes: bytes, title: str, expander_label: str):
    """
    Affiche un PDF dans un expander (embed HTML).
    """
    with st.expander(expander_label, expanded=False):
        st.caption(title)
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        html = f"""
        <iframe
            src="data:application/pdf;base64,{b64}"
            width="100%"
            height="600"
            style="border: 1px solid #ddd; border-radius: 8px;"
        ></iframe>
        """
        st.components.v1.html(html, height=620, scrolling=False)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buff = io.StringIO()
    df.to_csv(buff, index=False)
    return buff.getvalue().encode("utf-8")

def json_to_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ============================================================
# 2) O2 — calculs (reprend l’esprit de ton notebook)
# ============================================================

KG_PER_M = {"10M": 0.785, "15M": 1.57, "20M": 2.355, "25M": 3.925, "other": None}
INCH_TO_M = 0.0254

def parse_ft_in_to_inches(s: str) -> Optional[float]:
    if s is None:
        return None
    s0 = s.strip().replace("’", "'").replace("″", '"').replace("”", '"').replace("“", '"')
    s0 = re.sub(r"\s+", " ", s0)
    s0 = re.sub(r"(\d+)'\s*(\d+)\s*\"", r"\1'-\2\"", s0)  # 4'8" -> 4'-8"

    m = re.match(r"^\s*(\d+)\s*'\s*-\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?\s*\"\s*$", s0)
    if not m:
        m2 = re.match(r"^\s*(\d+)\s*'\s*-\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?\s*$", s0.replace('"', ''))
        if not m2:
            return None
        ft = int(m2.group(1)); inch = int(m2.group(2))
        frac = m2.group(3); den = m2.group(4)
    else:
        ft = int(m.group(1)); inch = int(m.group(2))
        frac = m.group(3); den = m.group(4)

    frac_val = (int(frac) / int(den)) if (frac and den) else 0.0
    return ft * 12.0 + inch + frac_val

def inches_to_ft_in_str(inches: float) -> str:
    if inches is None:
        return "NAN"
    inches_rounded = round(inches * 2.0) / 2.0
    ft = int(inches_rounded // 12)
    rem = inches_rounded - 12 * ft
    inch_int = int(rem // 1)
    frac = rem - inch_int
    if abs(frac) < 1e-9:
        return f"{ft}'-{inch_int}\""
    if abs(frac - 0.5) < 1e-9:
        return f"{ft}'-{inch_int} 1/2\""
    return f"{ft}'-{rem:.2f}\""

def compute_length_from_notes(notes: str) -> Optional[float]:
    if not notes:
        return None
    t = notes.replace("×", "*").replace("x", "*")
    pat = re.compile(
        r"(?P<A>\d+\s*'\s*-\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\")\s*\+\s*2\s*\*\s*(?P<B>\d+\s*'\s*-\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\")"
    )
    m = pat.search(t)
    if m:
        A = parse_ft_in_to_inches(m.group("A"))
        B = parse_ft_in_to_inches(m.group("B"))
        if A is not None and B is not None:
            return A + 2.0 * B
    # fallback: A + B
    terms = re.findall(r"\d+\s*'\s*-\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\"", t)
    if len(terms) >= 2 and "+" in t:
        A = parse_ft_in_to_inches(terms[0])
        B = parse_ft_in_to_inches(terms[1])
        if A is not None and B is not None:
            return A + B
    return None

def get_unit_length_inches(item: Dict[str, Any]) -> Optional[float]:
    # 1) override direct (si user corrige)
    override = item.get("unit_length_ft_in_override")
    if override and override != "NAN":
        v = parse_ft_in_to_inches(override)
        if v is not None:
            return v

    # 2) si déjà calculé par pipeline
    v = item.get("unit_length_in_final")
    if isinstance(v, (int, float)):
        return float(v)

    shape = item.get("shape", "other")

    if shape == "L":
        legs = item.get("legs_ft_in") or []
        if len(legs) >= 2:
            a = parse_ft_in_to_inches(legs[0])
            b = parse_ft_in_to_inches(legs[1])
            if a is not None and b is not None:
                return a + b

    if shape == "stirrup":
        dims = item.get("stirrup_dims_ft_in") or []
        if len(dims) >= 2:
            a = parse_ft_in_to_inches(dims[0])
            b = parse_ft_in_to_inches(dims[1])
            if a is not None and b is not None:
                return 2.0 * (a + b)

    if shape == "straight":
        L = item.get("explicit_length_ft_in")
        if L:
            return parse_ft_in_to_inches(L)
        return compute_length_from_notes(item.get("notes", ""))

    return compute_length_from_notes(item.get("notes", ""))

def compute_o2(report: Dict[str, Any], price_per_kg: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    out = json.loads(json.dumps(report))  # deep copy

    if out.get("doc_status") != "ok":
        out["o2_summary"] = {"status": "skipped", "reason": out.get("doc_status")}
        return out

    summary_items = []
    totals_by_diameter: Dict[str, Dict[str, Any]] = {}

    for item in out.get("items", []):
        diameter = item.get("diameter", "other")
        count = item.get("count")

        count_int = count if isinstance(count, int) else None

        unit_in = get_unit_length_inches(item)
        unit_ft_in = inches_to_ft_in_str(unit_in) if unit_in is not None else "NAN"
        unit_m = unit_in * INCH_TO_M if unit_in is not None else None

        total_m = (count_int * unit_m) if (count_int is not None and unit_m is not None) else None

        kg_per_m = KG_PER_M.get(diameter)
        weight_kg = (total_m * kg_per_m) if (kg_per_m is not None and total_m is not None) else None

        cost = None
        unit_price = None
        if price_per_kg and weight_kg is not None:
            unit_price = price_per_kg.get(diameter)
            if unit_price is not None:
                cost = weight_kg * unit_price

        item["unit_length_ft_in_o2"] = unit_ft_in
        item["unit_length_m_o2"] = unit_m if unit_m is not None else "NAN"
        item["total_length_m_o2"] = total_m if total_m is not None else "NAN"
        item["kg_per_m_o2"] = kg_per_m if kg_per_m is not None else "NAN"
        item["weight_kg_o2"] = weight_kg if weight_kg is not None else "NAN"
        item["unit_price_per_kg_o2"] = unit_price if unit_price is not None else "NAN"
        item["cost_o2"] = cost if cost is not None else "NAN"

        if diameter not in totals_by_diameter:
            totals_by_diameter[diameter] = {"total_m": 0.0, "weight_kg": 0.0, "cost": 0.0, "has_cost": False}
        if total_m is not None:
            totals_by_diameter[diameter]["total_m"] += total_m
        if weight_kg is not None:
            totals_by_diameter[diameter]["weight_kg"] += weight_kg
        if cost is not None:
            totals_by_diameter[diameter]["cost"] += cost
            totals_by_diameter[diameter]["has_cost"] = True

        summary_items.append({
            "item_id": item.get("item_id"),
            "diameter": diameter,
            "shape": item.get("shape"),
            "count": count,
            "unit_length_ft_in_o2": unit_ft_in,
            "total_length_m_o2": total_m if total_m is not None else "NAN",
            "weight_kg_o2": weight_kg if weight_kg is not None else "NAN",
            "unit_price_per_kg_o2": unit_price if unit_price is not None else "NAN",
            "cost_o2": cost if cost is not None else "NAN",
        })

    grand_total_m = sum(v["total_m"] for v in totals_by_diameter.values())
    grand_weight_kg = sum(v["weight_kg"] for v in totals_by_diameter.values())
    has_any_cost = any(v["has_cost"] for v in totals_by_diameter.values())
    grand_cost = sum(v["cost"] for v in totals_by_diameter.values()) if has_any_cost else "NAN"

    out["o2_table"] = summary_items
    out["o2_totals_by_diameter"] = totals_by_diameter
    out["o2_grand_totals"] = {"total_m": grand_total_m, "weight_kg": grand_weight_kg, "cost": grand_cost}
    return out

def round_for_display(result: dict, nd_m: int = 2, nd_kg: int = 2, nd_cost: int = 2) -> dict:
    def r(x, nd):
        if isinstance(x, (int, float)):
            return round(float(x), nd)
        return x

    for row in result.get("o2_table", []):
        row["total_length_m_o2"] = r(row.get("total_length_m_o2"), nd_m)
        row["weight_kg_o2"] = r(row.get("weight_kg_o2"), nd_kg)
        row["cost_o2"] = r(row.get("cost_o2"), nd_cost)

    for _, t in result.get("o2_totals_by_diameter", {}).items():
        t["total_m"] = r(t.get("total_m"), nd_m)
        t["weight_kg"] = r(t.get("weight_kg"), nd_kg)
        t["cost"] = r(t.get("cost"), nd_cost)

    gt = result.get("o2_grand_totals", {})
    gt["total_m"] = r(gt.get("total_m"), nd_m)
    gt["weight_kg"] = r(gt.get("weight_kg"), nd_kg)
    gt["cost"] = r(gt.get("cost"), nd_cost)
    return result


# ============================================================
# 3) STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Estimateur armatures — Prototype", layout="wide")

st.title("Estimateur du prix d’armatures à partir d’un plan")
st.caption("Prototype")

# Session state init
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "last_file_sig" not in st.session_state:
    st.session_state.last_file_sig = None
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_USER_PROMPT_FR

top_left, top_right = st.columns([3, 1])
with top_right:
    st.button("🏠 Retour accueil / Reset", on_click=reset_app, use_container_width=True)

st.divider()

# ---- Accueil (upload obligatoire) ----
uploaded = st.file_uploader("1) Uploader un fichier PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload un PDF pour lancer l’analyse et afficher les résultats.")
    st.stop()

ok_pdf, reason = is_probably_pdf(uploaded)
if not ok_pdf:
    st.error(f"Le fichier uploadé ne semble pas être un PDF valide. {reason}".strip())
    st.stop()

sig = file_signature(uploaded)
new_file = (st.session_state.last_file_sig != sig)
if new_file:
    st.session_state.last_file_sig = sig
    st.session_state.analysis_ready = False
    # reset downstream results
    for k in ["report", "report_edit_df", "o2_result", "o2_result_rounded"]:
        if k in st.session_state:
            del st.session_state[k]

st.success(f"PDF reçu : {uploaded.name}  |  Taille: {getattr(uploaded, 'size', 'unknown')} bytes")

# ---- Simulation analyse (UX identique) ----
if not st.session_state.analysis_ready:
    with st.spinner("Analyse du plan…"):
        simulate_analysis()
    st.session_state.analysis_ready = True
    st.rerun()

# ---- Lire bytes du pdf cible ----
uploaded.seek(0)
target_pdf_bytes = uploaded.read()
export_base = base_name_without_pdf(uploaded.name)

# ---- Expanders: PDF EXEMPLE + USER_PROMPT ----
# PDF EXEMPLE
if os.path.exists(EXAMPLE_PDF):
    with open(EXAMPLE_PDF, "rb") as f:
        example_pdf_bytes = f.read()
    render_pdf_in_expander(
        example_pdf_bytes,
        title=f"PDF EXEMPLE : {os.path.basename(EXAMPLE_PDF)}",
        expander_label="📄 Voir le PDF EXEMPLE (in-context)"
    )
else:
    with st.expander("📄 Voir le PDF EXEMPLE (in-context)", expanded=False):
        st.warning(f"EXAMPLE_PDF introuvable: {EXAMPLE_PDF} (adapte le chemin).")

# Prompt (modifiable: USER_PROMPT seulement)
with st.expander("🧠 Voir / modifier le prompt (USER_PROMPT)", expanded=False):
    st.caption("Le prompt est modifiable. Les modifications s’appliquent au prochain calcul IA.")
    st.session_state.user_prompt = st.text_area(
        "USER_PROMPT (FR)",
        value=st.session_state.user_prompt,
        height=420
    )

# (optionnel) afficher aussi le PDF CIBLE en expander
render_pdf_in_expander(
    target_pdf_bytes,
    title=f"PDF CIBLE : {uploaded.name}",
    expander_label="📄 Voir le PDF CIBLE uploadé"
)

st.divider()

# ============================================================
# 5) Appel IA + affichage report
# ============================================================

# On appelle l'IA une seule fois par upload (et pas à chaque rerun UI)
if "report" not in st.session_state:
    try:
        with st.spinner("Appel IA (extraction armatures)…"):
            # IMPORTANT: tu as dit qu'on suppose cette méthode existe.
            report = pdf_to_report_AI(target_pdf_bytes, st.session_state.user_prompt)
        st.session_state.report = report
    except NotImplementedError as e:
        st.session_state.report = {
            "doc_status": "doc_not_plan",
            "document_name": export_base,
            "items": [],
            "warnings": [
                "pdf_to_report_AI() n'est pas encore branchée.",
                "Branche ton pipeline VLM + postprocess_report() dans pdf_to_report_AI().",
                str(e),
            ],
        }
    except Exception as e:
        st.session_state.report = {
            "doc_status": "doc_not_plan",
            "document_name": export_base,
            "items": [],
            "warnings": [f"Erreur IA: {type(e).__name__}: {e}"],
        }

report = st.session_state.report

# ---- Cas doc_not_plan / plan_no_rebar ----
doc_status = report.get("doc_status", "doc_not_plan")

if doc_status == "doc_not_plan":
    st.error("Le document ne ressemble pas à un plan exploitable pour l’estimation d’armatures.")
    with st.expander("Détails / raisons", expanded=True):
        st.json(report)
    st.stop()

if doc_status == "plan_no_rebar":
    st.warning("Plan détecté, mais aucune armature n’a été trouvée (items vide).")
    with st.expander("Rapport (JSON)", expanded=False):
        st.json(report)
    st.stop()

# ---- doc_status == ok ----
st.subheader("Résultat IA — Armatures détectées (modifiable)")

# Construire un DataFrame éditable minimal :
# - count (quantité)
# - diameter
# - unit_length_ft_in_override (si tu veux corriger rapidement une longueur)
# Le reste reste visible en JSON.
items = report.get("items", [])

def items_to_edit_df(items_list):
    rows = []
    for it in items_list:
        rows.append({
            "item_id": it.get("item_id"),
            "diameter": it.get("diameter"),
            "shape": it.get("shape"),
            "count": it.get("count"),
            "unit_length_ft_in_override": it.get("unit_length_ft_in_override", "NAN"),
            "raw_label": it.get("raw_label"),
            "page": it.get("page"),
            "notes": it.get("notes"),
        })
    return pd.DataFrame(rows)

if "report_edit_df" not in st.session_state:
    st.session_state.report_edit_df = items_to_edit_df(items)

edit_df = st.data_editor(
    st.session_state.report_edit_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "item_id": st.column_config.NumberColumn("item_id", disabled=True),
        "diameter": st.column_config.SelectboxColumn("diameter", options=["10M", "15M", "20M", "25M", "other"]),
        "shape": st.column_config.SelectboxColumn("shape", options=["L", "stirrup", "straight", "other"]),
        "count": st.column_config.TextColumn("count"),
        "unit_length_ft_in_override": st.column_config.TextColumn("override longueur (ft-in)", help="Optionnel. Ex: 7'-4\""),
        "raw_label": st.column_config.TextColumn("raw_label", disabled=True),
        "page": st.column_config.TextColumn("page", disabled=True),
        "notes": st.column_config.TextColumn("notes"),
    },
)

# Appliquer l’édition au report (sans casser le reste)
def apply_edits_to_report(report_dict: dict, edited_df: pd.DataFrame) -> dict:
    out = deepcopy(report_dict)
    # map item_id -> row
    rows = {int(r["item_id"]): r for _, r in edited_df.iterrows() if str(r.get("item_id", "")).strip() != ""}
    new_items = []
    for it in out.get("items", []):
        iid = it.get("item_id")
        if iid in rows:
            r = rows[iid]
            # count: autoriser int ou "NAN"
            cnt_raw = r.get("count")
            if isinstance(cnt_raw, str):
                cnt_raw = cnt_raw.strip()
                if cnt_raw.upper() == "NAN" or cnt_raw == "":
                    it["count"] = "NAN"
                else:
                    try:
                        it["count"] = int(float(cnt_raw))
                    except:
                        it["count"] = "NAN"
            elif isinstance(cnt_raw, (int, float)):
                it["count"] = int(cnt_raw)

            it["diameter"] = r.get("diameter", it.get("diameter"))
            it["shape"] = r.get("shape", it.get("shape"))
            it["notes"] = r.get("notes", it.get("notes"))

            ov = r.get("unit_length_ft_in_override")
            if isinstance(ov, str):
                ov = ov.strip()
                it["unit_length_ft_in_override"] = ov if ov else "NAN"
            else:
                it["unit_length_ft_in_override"] = "NAN"
        new_items.append(it)
    out["items"] = new_items
    return out

report_edited = apply_edits_to_report(report, edit_df)

with st.expander("Voir le rapport IA complet (JSON)", expanded=False):
    st.json(report_edited)

if report_edited.get("warnings"):
    st.warning("Warnings IA")
    with st.expander("Voir warnings", expanded=False):
        st.write(report_edited.get("warnings", []))

st.divider()

# ============================================================
# 6) Prix + bouton Estimer le coût + résultats O2
# ============================================================

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Prix du fer")

    price_unit = st.selectbox(
        "Unité de prix",
        options=["CAD_per_kg", "CAD_per_tonne"],
        index=0,
        key="price_unit",
    )
    price_value = st.number_input(
        "Valeur",
        min_value=0.0,
        value=2.50,
        step=0.10,
        key="price_value",
        help='Ex: 2.50 CAD/kg ou 2500 CAD/tonne',
    )

    # Conversion: on crée un dict price_per_kg identique pour tous diamètres
    # Si l'utilisateur saisit CAD/tonne -> /1000 pour CAD/kg
    if price_unit == "CAD_per_tonne":
        price_per_kg_value = float(price_value) / 1000.0
    else:
        price_per_kg_value = float(price_value)

    # On applique le même prix aux diamètres connus (tu peux étendre la liste)
    price_per_kg = {
        "10M": price_per_kg_value,
        "15M": price_per_kg_value,
        "20M": price_per_kg_value,
        "25M": price_per_kg_value,
        "other": price_per_kg_value,
    }

    st.caption(f"Prix utilisé en interne: {price_per_kg_value:.4f} CAD/kg")

    estimate = st.button("💰 Estimer le coût", use_container_width=True)

with col_right:
    st.subheader("Calcul (O2)")
    st.caption("Le calcul s’applique au tableau (potentiellement corrigé) ci-dessus.")

    if estimate:
        with st.spinner("Calcul O2 (longueurs, poids, coût)…"):
            o2_result = compute_o2(report_edited, price_per_kg=price_per_kg)
            o2_result_round = round_for_display(deepcopy(o2_result))
        st.session_state.o2_result = o2_result
        st.session_state.o2_result_rounded = o2_result_round
        st.success("Calcul terminé.")

    if "o2_result_rounded" in st.session_state:
        o2r = st.session_state.o2_result_rounded

        # Tableau O2 principal
        o2_df = pd.DataFrame(o2r.get("o2_table", []))
        st.dataframe(o2_df, use_container_width=True)

        # Totaux par diamètre
        st.subheader("Totaux par diamètre")
        totals_rows = []
        for d, t in o2r.get("o2_totals_by_diameter", {}).items():
            totals_rows.append({
                "diameter": d,
                "total_m": t.get("total_m"),
                "weight_kg": t.get("weight_kg"),
                "cost": t.get("cost") if t.get("has_cost") else "NAN",
            })
        totals_df = pd.DataFrame(totals_rows)
        st.dataframe(totals_df, use_container_width=True)

        # Total global en gros
        gt = o2r.get("o2_grand_totals", {})
        total_cost = gt.get("cost", "NAN")
        st.markdown("---")
        st.markdown(
            f"""
            <div style="padding:16px;border:1px solid #e6e6e6;border-radius:12px;background:#fafafa;">
              <div style="font-size:14px;color:#555;">Total estimé</div>
              <div style="font-size:34px;font-weight:800;">
                {total_cost if isinstance(total_cost, str) else f"{total_cost:,.2f}"} CAD
              </div>
              <div style="font-size:13px;color:#666;margin-top:6px;">
                Longueur totale: {gt.get("total_m", "NAN")} m · Poids total: {gt.get("weight_kg", "NAN")} kg
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Exports (choix: CSV/JSON) pour chaque tableau
        st.subheader("Exports")
        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            st.download_button(
                "⬇️ CSV — O2 (lignes)",
                data=df_to_csv_bytes(o2_df),
                file_name=f"{export_base}__o2_table.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                "⬇️ JSON — O2 (complet)",
                data=json_to_bytes(o2r),
                file_name=f"{export_base}__o2_full.json",
                mime="application/json",
                use_container_width=True
            )

        with export_col2:
            st.download_button(
                "⬇️ CSV — Totaux par diamètre",
                data=df_to_csv_bytes(totals_df),
                file_name=f"{export_base}__o2_totals_by_diameter.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                "⬇️ JSON — Report corrigé (entrée O2)",
                data=json_to_bytes(report_edited),
                file_name=f"{export_base}__report_edited.json",
                mime="application/json",
                use_container_width=True
            )

        with export_col3:
            # petit JSON grand totals
            st.download_button(
                "⬇️ JSON — Grand totals",
                data=json_to_bytes(gt),
                file_name=f"{export_base}__o2_grand_totals.json",
                mime="application/json",
                use_container_width=True
            )

        with st.expander("Voir JSON O2 (debug)", expanded=False):
            st.json(o2r)
    else:
        st.info("Clique sur « Estimer le coût » après avoir vérifié/corrigé le tableau IA.")

