# ============================================================
# AcierUnited — Extraction armatures (VLM) + calcul longueurs en Python
#
# - PDF EXEMPLE (in-context): Plan_Structure_Base_St-Roch.pdf
# - PDF CIBLE (target): n'importe quel PDF de plan
# - Prompt LONG en FR (production-ready, non-conversationnel)
# - Le modèle NE FAIT PAS l'arithmétique finale: il fournit dimensions + formule.
# - Le code Python calcule les longueurs (robuste aux erreurs 15-4 vs 15-5, etc.)
#
# Dépendances:
#   pip install -U openai pydantic pymupdf
# ============================================================

__all__ = ["pdf_to_report_AI", "EXAMPLE_PDF", "USER_PROMPT", "extract_rebar_text_candidates"]


import os
import re
import json
from typing import List, Literal, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
import fitz  # PyMuPDF
from openai import OpenAI
import streamlit as st

# ------------------ CONFIG ------------------
# IMPORTANT: ne jamais hardcoder la clé. Utilise un secret / variable d'env.
# Colab: os.environ["OPENAI_API_KEY"] = "sk-..." (dans une cellule privée)
#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
# if not OPENAI_API_KEY:
#     st.error("OPENAI_API_KEY manquante. Ajoute-la dans Streamlit Secrets ou dans l'environnement local.")
#     st.stop()

MODEL = "gpt-5"  # ou "gpt-4.1"

pdf_dir = "static/samples"  # dossier contenant les PDFs d'exemple et cible
EXAMPLE_PDF = f"{pdf_dir}/Plan_Structure_Base_St-Roch.pdf"

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
Vous êtes un estimateur professionnel spécialisé dans la lecture de plans de structure et le calcul des armatures.

Contraintes strictes :
- Retourner UNIQUEMENT un JSON conforme au schéma demandé (aucun texte hors JSON).
- Le JSON final doit concerner UNIQUEMENT le document cible.
- raw_label et evidence_text_raw doivent provenir du document cible.
- Ne pas effectuer le calcul arithmétique final des longueurs : extraire dimensions + formule uniquement.
- Éviter les doubles comptages entre vues (plan/élévation/profil), sauf occurrences réellement distinctes.
"""

# Prompt LONG, FR, production-grade (in-context complet) + ajout de "Pass 0" candidats
USER_PROMPT  = lambda target_candidates : f"""
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

Aucune longueur numérique finale ne doit être calculée ici.


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

Toute estimation ou hypothèse doit être explicitement justifiée dans les notes.


====================================================
CAS PARTICULIERS
====================================================

- Si le document analysé ne correspond pas à un plan de structure :
  retourner doc_status = "doc_not_plan".

- Si le document est un plan de structure mais ne contient aucune armature :
  retourner doc_status = "plan_no_rebar" et une liste items vide.


====================================================
DÉMONSTRATION IN-CONTEXT
====================================================

Le document d’exemple Plan_Structure_Base_St-Roch contient une seule page.

Sur cette page :
- la partie gauche contient des vues de la base de béton sans armatures,
- la partie droite contient trois vues représentant le même objet
  (deux poteaux sur une semelle jumelée) :
    (3) vue en plan du détail des armatures,
    (4) vue en élévation du détail des armatures,
    (5) vue de profil du détail des armatures.

Ces trois vues décrivent le même système d’armatures
et doivent être analysées conjointement.

La méthode suivante a été appliquée :

----------------------------------------------------
Groupe 1 — Barres en L (vue en plan, haut droite)
----------------------------------------------------
Libellé :
"12-15M en forme de 'L' de 7'-0\\" x 1'-4\\""

- diamètre : 15M
- forme : L
- quantité : 12 (indiquée dans le libellé)
- jambes : 7'-0\\", 1'-4\\"
- formula_hint : "a + b"

Ce groupe apparaît sur plusieurs vues,
mais correspond à un seul groupe physique.

----------------------------------------------------
Groupe 2 — Barres en L (vue en plan, haut gauche)
----------------------------------------------------
Même libellé et mêmes dimensions que le groupe 1.
Il s’agit d’un groupe physique distinct
et il doit être compté séparément.

----------------------------------------------------
Groupe 3 — Étriers 10M (petits)
----------------------------------------------------
Libellé :
"Étrier en 10M de 3'-0 1/2\\" x 4'-8\\" @ 12\\" c/c"

- diamètre : 10M
- forme : étrier
- dimensions : 3'-0 1/2\\", 4'-8\\"
- formula_hint : "2*(a + b)"
- quantité non indiquée

Détermination de la quantité :
- comptage visuel sur la vue en élévation : 6
- estimation alternative :
  espacement E = 12\\", hauteur H ≈ 7'-0\\",
  H/E + 1 ≈ 8

----------------------------------------------------
Groupe 4 — Étriers 10M identiques
----------------------------------------------------
Même libellé et mêmes dimensions que le groupe 3.
Il s’agit d’un second groupe physique distinct,
compté séparément.

----------------------------------------------------
Groupe 5 — Étriers 10M (grands)
----------------------------------------------------
Libellé :
"Étrier en 10M de 11'-4\\" x 3'-0 1/2\\" @ 12\\" c/c"

- diamètre : 10M
- forme : étrier
- dimensions : 11'-4\\", 3'-0 1/2\\"
- formula_hint : "2*(a + b)"
- quantité comptée visuellement sur l’élévation : 6

----------------------------------------------------
Groupe 6 — Barres 15M @ 12\\" c/c (élévation)
----------------------------------------------------
Libellé :
"15M @ 12\\" c/c"

- diamètre : 15M
- forme : barre droite
- espacement : 12\\"
- longueur déduite de la géométrie :
  largeur 4'-8\\" + 2 × 1'-4\\" (pattes des barres en L)
- quantité comptée visuellement sur l’élévation : 14

----------------------------------------------------
Groupe 7 — Barres 15M @ 12\\" c/c (profil)
----------------------------------------------------
Même libellé que le groupe 6,
mais géométrie différente.

- longueur déduite :
  11'-4\\" + 2 × 1'-4\\"
- quantité comptée visuellement sur le profil : 8

Ce groupe est distinct du groupe 6
et doit être compté séparément.


====================================================
TÂCHE SUR LE DOCUMENT CIBLE
====================================================

Appliquer exactement la même méthode au document cible.

Exigences de preuve :
- Pour chaque item, evidence_text_raw doit contenir au moins un extrait EXACT présent dans le document cible
  (même si le texte est bruité).
- raw_label doit provenir du document cible.
- Si l’information manque, utiliser "NAN" et expliquer dans notes, et/ou ajouter un warning.

Aide (candidats textuels extraits automatiquement du document cible) :
{target_candidates}

Retourner UNIQUEMENT le JSON final correspondant au document cible.
"""

# ============================================================
# PASS 0 — Extraction de candidats textuels (vector PDF)
# ============================================================

REBAR_PATTERNS = [
    r"\b10M\b", r"\b15M\b", r"\b20M\b", r"\b25M\b",
    r"@ *\d+\"? *c\/c", r"c\/c",
    r"\b[EÉ]trier\b", r"\barmatures?\b", r"\bbarres?\b",
]
CANDIDATE_RE = re.compile("|".join(REBAR_PATTERNS), re.IGNORECASE)

def extract_rebar_text_candidates(pdf_path: str, max_lines_per_page: int = 200) -> List[dict]:
    """
    Extract likely rebar-related text lines from a vector PDF.
    Returns list of dict: {page: int, lines: [str]}
    """
    doc = fitz.open(pdf_path)
    out: List[dict] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text") or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        hits = []
        for ln in lines:
            if CANDIDATE_RE.search(ln):
                clean = re.sub(r"\s+", " ", ln).strip()
                hits.append(clean)

        uniq, seen = [], set()
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)

        if uniq:
            out.append({"page": page_index + 1, "lines": uniq[:max_lines_per_page]})
    return out

# ============================================================
# Normalisation "OCR bruité" pour evidence_text (espaces intra-mots)
# ============================================================

def normalize_ocrish_text(s: str) -> str:
    """
    Normalize OCR-like strings where letters are separated by spaces.
    Conservative: removes spaces between alphanumerics, keeps normal word spacing.
    """
    if s is None:
        return s
    s = s.strip()
    s = re.sub(r"(?<=\w)\s+(?=\w)", "", s)  # remove intra-word spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ============================================================
# Parsing ft-in (incl. fractions like 1/2) + formatting
# ============================================================

ALT_RE = re.compile(
    r"""^\s*(?P<ft>-?\d+)\s*'\s*-\s*(?P<in>\d+)(?:\s+(?P<frac>\d+)\s*/\s*(?P<fracden>\d+))?\s*"\s*$""",
    re.VERBOSE
)

def parse_ft_in_to_inches(s: str) -> Optional[float]:
    """
    Parse strings like:
    - 7'-0"
    - 1'-4"
    - 3'-0 1/2"
    - 4'8" (rare) -> normalized to 4'-8"
    Returns inches as float, or None if cannot parse.
    """
    if s is None:
        return None
    s0 = s.strip()
    s0 = s0.replace("’", "'").replace("″", '"').replace("”", '"').replace("“", '"')
    s0 = re.sub(r"\s+", " ", s0)

    # normalize "4'8\"" -> "4'-8\""
    s0 = re.sub(r"(\d+)'\s*(\d+)\s*\"", r"\1'-\2\"", s0)

    m = ALT_RE.match(s0)
    if not m:
        # relaxed: allow missing quote
        s1 = s0.replace('"', '')
        m2 = re.match(r"^\s*(?P<ft>-?\d+)\s*'\s*-\s*(?P<in>\d+)(?:\s+(?P<frac>\d+)\s*/\s*(?P<fracden>\d+))?\s*$", s1)
        if not m2:
            return None
        ft = int(m2.group("ft"))
        inch = int(m2.group("in"))
        frac = m2.group("frac")
        fracden = m2.group("fracden")
        frac_val = (int(frac) / int(fracden)) if (frac and fracden) else 0.0
        return ft * 12.0 + inch + frac_val

    ft = int(m.group("ft"))
    inch = int(m.group("in"))
    frac = m.group("frac")
    fracden = m.group("fracden")
    frac_val = (int(frac) / int(fracden)) if (frac and fracden) else 0.0
    return ft * 12.0 + inch + frac_val

def inches_to_ft_in_str(inches: float) -> str:
    """
    Convert inches (float) to ft-in with 1/2" resolution.
    Example: 100 -> 8'-4"
             36.5 -> 3'-0 1/2"
    """
    if inches is None:
        return "NAN"

    inches_rounded = round(inches * 2.0) / 2.0  # nearest 1/2"
    sign = "-" if inches_rounded < 0 else ""
    inches_rounded = abs(inches_rounded)

    ft = int(inches_rounded // 12)
    rem = inches_rounded - 12 * ft
    inch_int = int(rem // 1)
    frac = rem - inch_int

    if abs(frac) < 1e-9:
        return f"{sign}{ft}'-{inch_int}\""
    if abs(frac - 0.5) < 1e-9:
        return f"{sign}{ft}'-{inch_int} 1/2\""
    return f"{sign}{ft}'-{rem:.2f}\""

def compute_unit_length_inches(item: Dict[str, Any]) -> Optional[float]:
    """
    Compute unit length in inches from structured fields:
    - shape == "L"       -> a+b using legs_ft_in (2 legs)
    - shape == "stirrup" -> 2*(a+b) using stirrup_dims_ft_in (2 dims)
    - shape == "straight"-> explicit_length_ft_in if present
    """
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
            v = parse_ft_in_to_inches(L)
            if v is not None:
                return v

    return None


# ============================================================
# Schéma JSON (modèle -> structuration; code -> calc final)
# ============================================================

DocStatus = Literal["ok", "doc_not_plan", "plan_no_rebar"]
ShapeType = Literal["L", "stirrup", "straight", "other"]

class RebarItemModel(BaseModel):
    item_id: int
    diameter: Literal["10M", "15M", "20M", "25M", "other"]
    shape: ShapeType

    count: Union[int, Literal["NAN"]]

    # Dimensions extraites (strings ft-in). Le modèle ne fait PAS l’arithmétique finale.
    legs_ft_in: Optional[List[str]] = None           # pour L : ["7'-0\"", "1'-4\""]
    stirrup_dims_ft_in: Optional[List[str]] = None   # pour étrier : ["3'-0 1/2\"", "4'-8\""]
    explicit_length_ft_in: Optional[str] = None      # pour barre droite si longueur explicitement écrite

    formula_hint: Optional[str] = None               # "a+b" ou "2*(a+b)" ou explication
    page: Union[int, Literal["NAN"]]

    raw_label: Optional[str] = None                  # label du CIBLE
    dimensions_raw: Optional[str] = None             # UNIQUEMENT ce qui est écrit; sinon null
    spacing_raw: Optional[str] = None
    estimation_method: Optional[str] = None

    # Preuves: extraits EXACTS vus dans le PDF CIBLE (même si bruités)
    evidence_text_raw: List[str] = Field(default_factory=list)
    source_view_hint: Optional[List[Union[int, str]]] = None

    notes: str

class RebarReportModel(BaseModel):
    doc_status: DocStatus
    document_name: str
    items: List[RebarItemModel] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# ============================================================
# Post-traitement: normaliser evidence + calculer longueurs finales (source-of-truth)
# ============================================================

def postprocess_report(report: RebarReportModel) -> Dict[str, Any]:
    out: Dict[str, Any] = report.model_dump()

    if out["doc_status"] != "ok":
        return out

    new_items = []
    for it in out["items"]:
        evidence_raw = it.get("evidence_text_raw") or []
        evidence_norm = [normalize_ocrish_text(x) for x in evidence_raw]

        # Calcul final depuis dims
        unit_in = compute_unit_length_inches(it)
        unit_ft_in_final = inches_to_ft_in_str(unit_in) if unit_in is not None else "NAN"

        it["evidence_text_norm"] = evidence_norm
        it["unit_length_in_final"] = unit_in if unit_in is not None else "NAN"
        it["unit_length_ft_in_final"] = unit_ft_in_final

        # (Option) sanity check: si dims manquantes, avertir
        if unit_ft_in_final == "NAN" and out.get("doc_status") == "ok":
            out["warnings"].append(
                f"Item {it.get('item_id')}: impossible de calculer unit_length_ft_in_final (dimensions manquantes ou non parsables)."
            )

        new_items.append(it)

    out["items"] = new_items
    return out



import os

def get_openai_key() -> str:
    # 1) env var
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # 2) streamlit secrets (si streamlit est dispo)
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    raise RuntimeError("OPENAI_API_KEY introuvable (env var ou st.secrets).")


def pdf_to_report_AI(target_pdf: str, user_prompt: str=None) -> Dict[str, Any]:

    # for tests/debug: load example report as candidates (to avoid calling AI)
    #return json.loads(open(f"{pdf_dir}/Plan_Structure_Base_St-Roch.json", "r", encoding="utf-8").read())
    

    # Préparer le client OpenAI + extraire candidats textuels
    client = OpenAI(api_key=get_openai_key())
    target_candidates = extract_rebar_text_candidates(target_pdf)

    # Upload files
    example_file = client.files.create(file=open(EXAMPLE_PDF, "rb"), purpose="user_data")
    target_file  = client.files.create(file=open(target_pdf,  "rb"), purpose="user_data")

    # Call the model
    response = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": example_file.id},
                    {"type": "input_file", "file_id": target_file.id},
                    {"type": "input_text", "text": USER_PROMPT(target_candidates) if user_prompt is None else user_prompt},
                ],
            },
        ],
        text_format=RebarReportModel,
    )

    report_model: RebarReportModel = response.output_parsed

    final_report = postprocess_report(report_model)

    #print(json.dumps(final_report, indent=2, ensure_ascii=False))
    # file_name = os.path.splitext(os.path.basename(target_pdf))[0]
    # with open(f'{pdf_dir}/{file_name}.json', 'w') as f:
    #     json.dump(final_report, f, indent=2, ensure_ascii=False)

    return final_report


import time
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError

def pdf_to_report_AI(target_pdf: str, user_prompt: str=None, n_trials: int=3) -> dict:
    client = OpenAI(
        api_key=get_openai_key(),   # ta fonction env/st.secrets
        timeout=120.0,              # important (PDF)
        max_retries=0,              # on gère nous-mêmes
    )

    target_candidates = extract_rebar_text_candidates(target_pdf)


    # Upload files avec retry
    def _upload(path):
        return client.files.create(file=open(path, "rb"), purpose="user_data")

    for attempt in range(n_trials):
        try:
            example_file = _upload(EXAMPLE_PDF)
            target_file  = _upload(target_pdf)
            break
        except APIConnectionError as e:
            if attempt == n_trials - 1:
                raise
            time.sleep(2.0 * (attempt + 1))

    prompt_text = USER_PROMPT(target_candidates) if user_prompt is None else user_prompt

    # Call model avec retry
    for attempt in range(n_trials):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "input_file", "file_id": example_file.id},
                        {"type": "input_file", "file_id": target_file.id},
                        {"type": "input_text", "text": prompt_text},
                    ]},
                ],
                text_format=RebarReportModel,
            )
            #report_model = response.output_parsed
            report_model: RebarReportModel = response.output_parsed
            return postprocess_report(report_model)

        except RateLimitError as e:
            if attempt == 2:
                raise
            time.sleep(3.0 * (attempt + 1))

        except APIConnectionError as e:
            if attempt == n_trials - 1:
                raise
            time.sleep(2.0 * (attempt + 1))

        except APIStatusError as e:
            # ex: 502/503 → retry
            if e.status_code in (500, 502, 503, 504) and attempt < n_trials - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise


if __name__ == "__main__":
    file_name = "55792_Plan_Structure_AOP"
    file_name = "Plan_Structure_Base_St-Roch"
    
    TARGET_PDF  = f"{pdf_dir}/{file_name}.pdf"
    report = pdf_to_report_AI(TARGET_PDF)
    print(json.dumps(report, indent=2, ensure_ascii=False))