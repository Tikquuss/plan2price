import re
import json
from typing import Any, Dict, Optional, Tuple

# -----------------------------
# Conventions (à adapter si besoin)
# -----------------------------
KG_PER_M = {
    "10M": 0.785,
    "15M": 1.57,
    "20M": 2.355,   # optionnel
    "25M": 3.925,   # optionnel
    "other": None
}

INCH_TO_M = 0.0254

# -----------------------------
# Parsing ft-in -> inches (supporte 1/2)
# -----------------------------
def parse_ft_in_to_inches(s: str) -> Optional[float]:
    """
    Parse strings like:
      7'-0"
      1'-4"
      3'-0 1/2"
      4'8"  -> normalized to 4'-8"
    Returns inches (float) or None.
    """
    if s is None:
        return None
    s0 = s.strip()
    s0 = s0.replace("’", "'").replace("″", '"').replace("”", '"').replace("“", '"')
    s0 = re.sub(r"\s+", " ", s0)

    # normalize "4'8\"" -> "4'-8\""
    s0 = re.sub(r"(\d+)'\s*(\d+)\s*\"", r"\1'-\2\"", s0)

    # match ft'-in [frac] "
    m = re.match(r"^\s*(\d+)\s*'\s*-\s*(\d+)(?:\s+(\d+)\s*/\s*(\d+))?\s*\"\s*$", s0)
    if not m:
        # relaxed: allow missing quote
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
    """
    Format inches -> ft'-in" with 1/2 resolution.
    """
    if inches is None:
        return "NAN"
    inches_rounded = round(inches * 2.0) / 2.0  # nearest 1/2"
    ft = int(inches_rounded // 12)
    rem = inches_rounded - 12 * ft
    inch_int = int(rem // 1)
    frac = rem - inch_int
    if abs(frac) < 1e-9:
        return f"{ft}'-{inch_int}\""
    if abs(frac - 0.5) < 1e-9:
        return f"{ft}'-{inch_int} 1/2\""
    return f"{ft}'-{rem:.2f}\""

def inches_to_m(inches: float) -> float:
    return inches * INCH_TO_M

# -----------------------------
# Fallback: extraire une formule depuis "notes"
#   supporte:
#     "4'-8\" + 2×1'-4\""
#     "11'-4\" + 2×1'-4\""
#   (on gère ×, x, *, "2×", "2 x", "2*")
# -----------------------------
def compute_length_from_notes(notes: str) -> Optional[float]:
    if not notes:
        return None

    # Remplace symboles × par *
    t = notes.replace("×", "*").replace("x", "*")
    # On ne cherche que la partie où il y a une somme explicite avec des ft-in
    # On extrait toutes les occurrences de ft-in du type 4'-8" ou 1'-4"
    terms = re.findall(r"\d+\s*'\s*-\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\"", t)
    if len(terms) == 0:
        return None

    # Cas attendu : "A + 2*B" ou "A + 2 * B"
    # On tente d'identifier A et B.
    # Exemple: ... "4'-8\" + 2*1'-4\""
    # On va chercher un pattern explicite A + 2*B
    pat = re.compile(
        r"(?P<A>\d+\s*'\s*-\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\")\s*\+\s*2\s*\*\s*(?P<B>\d+\s*'\s*-\s*\d+(?:\s+\d+\s*/\s*\d+)?\s*\")"
    )
    m = pat.search(t)
    if m:
        A = parse_ft_in_to_inches(m.group("A"))
        B = parse_ft_in_to_inches(m.group("B"))
        if A is not None and B is not None:
            return A + 2.0 * B

    # fallback plus simple: si exactement 2 termes trouvés et présence de "2*"
    if len(terms) >= 2 and ("2*" in t or "2 *" in t):
        A = parse_ft_in_to_inches(terms[0])
        B = parse_ft_in_to_inches(terms[1])
        if A is not None and B is not None:
            return A + 2.0 * B

    # sinon: si juste "A + B" (rare)
    if len(terms) >= 2 and "+" in t:
        A = parse_ft_in_to_inches(terms[0])
        B = parse_ft_in_to_inches(terms[1])
        if A is not None and B is not None:
            return A + B

    return None

# -----------------------------
# Prendre la longueur unitaire (inches) d'un item
#   priorité:
#     1) unit_length_in_final si déjà numérique
#     2) calculer depuis legs/stirrup/explicit_length
#     3) fallback depuis notes (items 6/7 typiquement)
# -----------------------------
def get_unit_length_inches(item: Dict[str, Any]) -> Optional[float]:
    v = item.get("unit_length_in_final", None)
    if isinstance(v, (int, float)):
        return float(v)

    shape = item.get("shape")

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

        # fallback: parse from notes (ex: 4'-8" + 2×1'-4")
        return compute_length_from_notes(item.get("notes", ""))

    # fallback generic (notes)
    return compute_length_from_notes(item.get("notes", ""))

# -----------------------------
# Etape O2: calcul quantités + poids + coût
# -----------------------------
def compute_o2(report: Dict[str, Any], price_per_kg: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    price_per_kg: dict optionnel, ex:
        {"10M": 2.10, "15M": 2.30}  # en $/kg (ou unité monétaire de ton choix)
    Si None, on calcule seulement longueurs/poids.
    """
    out = json.loads(json.dumps(report))  # deep copy

    if out.get("doc_status") != "ok":
        out["o2_summary"] = {"status": "skipped", "reason": out.get("doc_status")}
        return out

    summary_items = []
    totals_by_diameter = {}

    for item in out.get("items", []):
        diameter = item.get("diameter", "other")
        count = item.get("count")

        # count peut être "NAN"
        if not isinstance(count, int):
            count_int = None
        else:
            count_int = count

        unit_in = get_unit_length_inches(item)
        unit_ft_in = inches_to_ft_in_str(unit_in) if unit_in is not None else "NAN"

        # Calculs métriques
        unit_m = inches_to_m(unit_in) if unit_in is not None else None

        if count_int is not None and unit_m is not None:
            total_m = count_int * unit_m
        else:
            total_m = None

        kg_per_m = KG_PER_M.get(diameter)
        if kg_per_m is not None and total_m is not None:
            weight_kg = total_m * kg_per_m
        else:
            weight_kg = None

        # Coût
        cost = None
        unit_price = None
        if price_per_kg and weight_kg is not None:
            unit_price = price_per_kg.get(diameter)
            if unit_price is not None:
                cost = weight_kg * unit_price

        # injecter résultats "O2" dans l'item (sans casser le reste)
        item["unit_length_in_o2"] = unit_in if unit_in is not None else "NAN"
        item["unit_length_ft_in_o2"] = unit_ft_in
        item["unit_length_m_o2"] = unit_m if unit_m is not None else "NAN"
        item["total_length_m_o2"] = total_m if total_m is not None else "NAN"
        item["kg_per_m_o2"] = kg_per_m if kg_per_m is not None else "NAN"
        item["weight_kg_o2"] = weight_kg if weight_kg is not None else "NAN"
        item["unit_price_per_kg_o2"] = unit_price if unit_price is not None else "NAN"
        item["cost_o2"] = cost if cost is not None else "NAN"

        # accumulate totals by diameter
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
            "count": count,
            "unit_length_ft_in_o2": item["unit_length_ft_in_o2"],
            "total_length_m_o2": item["total_length_m_o2"],
            "weight_kg_o2": item["weight_kg_o2"],
            "cost_o2": item["cost_o2"],
        })

    # Totaux globaux
    grand_total_m = 0.0
    grand_weight_kg = 0.0
    grand_cost = 0.0
    has_any_cost = False

    for d, t in totals_by_diameter.items():
        grand_total_m += t["total_m"]
        grand_weight_kg += t["weight_kg"]
        if t["has_cost"]:
            grand_cost += t["cost"]
            has_any_cost = True

    out["o2_table"] = summary_items
    out["o2_totals_by_diameter"] = totals_by_diameter
    out["o2_grand_totals"] = {
        "total_m": grand_total_m,
        "weight_kg": grand_weight_kg,
        "cost": grand_cost if has_any_cost else "NAN",
    }

    # warnings si longueurs manquantes
    for item in out.get("items", []):
        if item.get("unit_length_ft_in_o2") == "NAN":
            out.setdefault("warnings", []).append(
                f"Item {item.get('item_id')}: longueur unitaire introuvable (même après fallback notes) => poids/coût non calculables."
            )
        if item.get("count") == "NAN":
            out.setdefault("warnings", []).append(
                f"Item {item.get('item_id')}: quantité = NAN => longueurs totales/poids/coût non calculables."
            )

    return out

def round_for_display(result: dict, nd_m: int = 2, nd_kg: int = 2, nd_cost: int = 2) -> dict:
    """
    Arrondit les champs d'affichage pour éviter les floats moches.
    Ne change pas les strings "NAN".
    """
    def r(x, nd):
        if isinstance(x, (int, float)):
            return round(float(x), nd)
        return x

    # table
    for row in result.get("o2_table", []):
        row["total_length_m_o2"] = r(row.get("total_length_m_o2"), nd_m)
        row["weight_kg_o2"] = r(row.get("weight_kg_o2"), nd_kg)
        row["cost_o2"] = r(row.get("cost_o2"), nd_cost)

    # totals by diameter
    for d, t in result.get("o2_totals_by_diameter", {}).items():
        t["total_m"] = r(t.get("total_m"), nd_m)
        t["weight_kg"] = r(t.get("weight_kg"), nd_kg)
        t["cost"] = r(t.get("cost"), nd_cost)

    # grand totals
    gt = result.get("o2_grand_totals", {})
    gt["total_m"] = r(gt.get("total_m"), nd_m)
    gt["weight_kg"] = r(gt.get("weight_kg"), nd_kg)
    gt["cost"] = r(gt.get("cost"), nd_cost)

    return result