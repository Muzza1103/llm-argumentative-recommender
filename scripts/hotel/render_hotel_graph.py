from __future__ import annotations

import argparse
import html
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _escape(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return html.escape(str(value), quote=True)


def _number(value: object) -> str:
    if isinstance(value, bool):
        return _escape(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return _escape(value)


def _json(value: object) -> str:
    return html.escape(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        quote=True,
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, list) else []


def _table(
    headers: Sequence[tuple[str, str]],
    rows: Iterable[Mapping[str, Any]],
    *,
    empty: str = "Aucune donnée.",
) -> str:
    rendered_rows = list(rows)
    if not rendered_rows:
        return f'<p class="empty">{_escape(empty)}</p>'
    head = "".join(f"<th>{_escape(label)}</th>" for _, label in headers)
    body = []
    for row in rendered_rows:
        cells = "".join(
            f"<td>{row.get(key, '—')}</td>" for key, _ in headers
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _status_badge(status: object) -> str:
    normalized = str(status or "unknown").casefold()
    css = {
        "satisfied": "ok",
        "eligible": "ok",
        "included_in_dfquad": "ok",
        "violated": "bad",
        "ineligible": "bad",
        "rejected": "bad",
        "unknown": "unknown",
    }.get(normalized, "neutral")
    return f'<span class="badge {css}">{_escape(status)}</span>'


def _source_summary(sources: object) -> str:
    rendered = []
    for source in _sequence(sources):
        item = _mapping(source)
        rendered.append(
            "<div class='source-line'><code>"
            + _escape(item.get("source_ref"))
            + "</code> — "
            + _escape(item.get("value"))
            + "</div>"
        )
    return "".join(rendered) or "—"


def _constraint_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for outcome_value in _sequence(payload.get("constraint_outcomes")):
        outcome = _mapping(outcome_value)
        constraint = _mapping(outcome.get("constraint"))
        comparison = _mapping(outcome.get("comparison"))
        rows.append(
            {
                "constraint_id": _escape(constraint.get("constraint_id")),
                "mode": _escape(constraint.get("mode")),
                "target": _escape(
                    constraint.get("target", constraint.get("field"))
                ),
                "qualifiers": _json(constraint.get("qualifiers", {})),
                "source_text": _escape(
                    constraint.get("source_text", constraint.get("text"))
                ),
                "importance_raw": _number(constraint.get("importance_raw")),
                "normalized_weight": _number(
                    constraint.get("normalized_weight")
                ),
                "weighting_method": _escape(
                    constraint.get("weighting_method")
                ),
                "status": _status_badge(outcome.get("status")),
                "reason": _escape(outcome.get("reason")),
                "requested": _escape(comparison.get("requested_value")),
                "metadata": _escape(comparison.get("metadata_value")),
                "requested_canonical": _escape(
                    comparison.get("requested_canonical")
                ),
                "metadata_canonical": _escape(
                    comparison.get("metadata_canonical")
                ),
                "proof": _source_summary(outcome.get("fact_sources")),
            }
        )
    return rows


def _aspect_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    preferences = _mapping(payload.get("session_preferences"))
    aspects = _mapping(preferences.get("aspect_preferences"))
    observed = set(_sequence(payload.get("observed_preference_aspects")))
    rows = []
    for aspect, details_value in aspects.items():
        details = _mapping(details_value)
        rows.append(
            {
                "aspect": _escape(aspect),
                "source_text": _escape(details.get("source_text")),
                "importance_raw": _number(details.get("importance_raw")),
                "normalized_weight": _number(
                    details.get("normalized_weight")
                ),
                "weighting_method": _escape(details.get("weighting_method")),
                "hotel_evidence": _status_badge(
                    "observed" if aspect in observed else "missing"
                ),
            }
        )
    return rows


def _argument_card(argument_value: object, arg_type: str) -> str:
    argument = _mapping(argument_value)
    metadata = _mapping(argument.get("metadata"))
    evidence = _sequence(argument.get("evidence"))
    evidence_html = "".join(
        f"<li>{_escape(item)}</li>" for item in evidence
    ) or "<li>—</li>"
    refs = argument.get("preference_refs", [])
    sources = argument.get("source_refs", [])
    return f"""
    <article class="argument-card {arg_type}">
      <div class="argument-title"><code>{_escape(argument.get('id'))}</code></div>
      <div class="argument-force">importance = {_number(argument.get('importance_raw'))} ; coefficient /5 = {_number(argument.get('normalized_weight'))} ; Wilson/confiance = {_number(argument.get('evidence_score'))} ; force = {_number(argument.get('intrinsic_strength'))}</div>
      <div><strong>Méthode :</strong> {_escape(metadata.get('weighting_method'))}</div>
      <p>{_escape(argument.get('text'))}</p>
      <div><strong>Préférences :</strong> {_json(refs)}</div>
      <div><strong>Sources :</strong> {_json(sources)}</div>
      <details><summary>Preuves exactes</summary><ul>{evidence_html}</ul></details>
    </article>
    """


def _graph_html(payload: Mapping[str, Any]) -> str:
    arguments = [
        _mapping(argument) for argument in _sequence(payload.get("arguments"))
    ]
    supports = [
        argument for argument in arguments
        if argument.get("arg_type") == "support"
    ]
    attacks = [
        argument for argument in arguments
        if argument.get("arg_type") == "attack"
    ]
    dfquad = _mapping(payload.get("dfquad"))
    root = f"""
      <article class="root-card">
        <strong>ROOT</strong>
        <span>Recommander cet hôtel</span>
        <span>initial = {_number(dfquad.get('root_base_score'))}</span>
        <span>final = {_number(payload.get('dfquad_score'))}</span>
      </article>
    """
    support_cards = "".join(
        _argument_card(argument, "support") for argument in supports
    ) or '<p class="empty">Aucun support inclus.</p>'
    attack_cards = "".join(
        _argument_card(argument, "attack") for argument in attacks
    ) or '<p class="empty">Aucune attaque incluse.</p>'
    return f"""
    <div class="graph-grid">
      <div class="graph-column">
        <h3>Supports</h3>{support_cards}
      </div>
      <div class="root-column"><div class="arrow support-arrow">→</div>{root}<div class="arrow attack-arrow">←</div></div>
      <div class="graph-column">
        <h3>Attaques</h3>{attack_cards}
      </div>
    </div>
    """


def _accepted_rows(validation: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for value in _sequence(validation.get("accepted_arguments")):
        argument = _mapping(value)
        rows.append(
            {
                "id": _escape(argument.get("id")),
                "kind": _escape(argument.get("kind")),
                "type": _escape(argument.get("type")),
                "status": _status_badge(argument.get("scoring_status")),
                "unit": _escape(argument.get("scoring_unit_id")),
                "preferences": _json(argument.get("preference_refs", [])),
                "sources": _json(argument.get("source_refs", [])),
                "text": _escape(argument.get("text")),
            }
        )
    return rows


def _rejected_rows(validation: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for value in _sequence(validation.get("rejected_arguments")):
        rejected = _mapping(value)
        proposal = _mapping(rejected.get("proposal"))
        rows.append(
            {
                "id": _escape(proposal.get("id")),
                "kind": _escape(proposal.get("kind")),
                "type": _escape(proposal.get("type")),
                "reasons": _json(rejected.get("reasons", [])),
                "proposal": _json(rejected.get("proposal")),
            }
        )
    return rows


def _relation_rows(validation: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for value in _sequence(validation.get("relations")):
        relation = _mapping(value)
        rows.append(
            {
                "id": _escape(relation.get("id")),
                "source": _escape(relation.get("source_argument_id")),
                "target": _escape(relation.get("target_argument_id")),
                "type": _escape(relation.get("relation_type")),
                "accepted": _status_badge(
                    "accepted" if relation.get("accepted") else "rejected"
                ),
                "scoring": _escape("explanatory_only"),
                "reasons": _json(relation.get("reasons", [])),
            }
        )
    return rows


def _evidence_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    hybrid = _mapping(payload.get("hybrid"))
    prepared = _mapping(hybrid.get("prepared_context"))
    sources = {
        str(_mapping(value).get("source_id")): _mapping(value)
        for value in _sequence(prepared.get("authorized_sources"))
    }
    rows = []
    for argument_value in _sequence(payload.get("arguments")):
        argument = _mapping(argument_value)
        source_refs = list(_sequence(argument.get("source_refs")))
        if not source_refs:
            source_refs = list(
                _sequence(_mapping(argument.get("metadata")).get("source_refs"))
            )
        for source_ref in source_refs:
            source = sources.get(str(source_ref), {})
            rows.append(
                {
                    "argument_id": _escape(argument.get("id")),
                    "source_ref": _escape(source_ref),
                    "kind": _escape(source.get("kind")),
                    "evidence": _escape(source.get("evidence_text")),
                    "payload": _json(source.get("payload", {})),
                }
            )
    return rows


def _unit_rows(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for value in _sequence(payload.get("scoring_units")):
        unit = _mapping(value)
        rows.append(
            {
                "id": _escape(unit.get("scoring_unit_id")),
                "kind": _escape(unit.get("kind")),
                "type": _escape(unit.get("type")),
                "intent": _escape(unit.get("intent_ref")),
                "importance": _number(unit.get("importance_raw")),
                "weight": _number(unit.get("normalized_weight")),
                "method": _escape(unit.get("weighting_method")),
                "confidence": _number(unit.get("confidence_factor")),
                "formula": _escape(unit.get("force_formula")),
                "force": _number(unit.get("final_force")),
                "counted": _status_badge(
                    "included_in_dfquad"
                    if unit.get("included_in_dfquad")
                    else "not_counted"
                ),
                "reason": _escape(
                    unit.get("dfquad_reason", unit.get("availability_reason"))
                ),
            }
        )
    return rows


def build_hotel_html(result: Mapping[str, Any] | object) -> str:
    """Build one dependency-free, escaped HTML report from a public result."""
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, Mapping):
        raise TypeError("hotel result must be a mapping or expose to_dict()")
    payload = result
    preferences = _mapping(payload.get("session_preferences"))
    constraint_rows = _constraint_rows(payload)
    hard_rows = [row for row in constraint_rows if row["mode"] == "hard"]
    soft_rows = [row for row in constraint_rows if row["mode"] == "soft"]
    hybrid = _mapping(payload.get("hybrid"))
    validation = _mapping(hybrid.get("validation"))
    excluded_rows = [
        {
            "id": _escape(_mapping(value).get("argument_id")),
            "reason": _escape(_mapping(value).get("reason")),
            "unit": _escape(_mapping(value).get("scoring_unit_id")),
            "counted": _escape(_mapping(value).get("counted_argument_id")),
        }
        for value in _sequence(validation.get("excluded_arguments"))
    ]
    dfquad = _mapping(payload.get("dfquad"))
    scoring_units = list(_sequence(payload.get("scoring_units")))
    registered_units = len(scoring_units)
    counted_units = sum(
        bool(_mapping(unit).get("included_in_dfquad"))
        for unit in scoring_units
    )
    score_rows = [
        {
            "method": _escape(payload.get("weighting_method")),
            "status": _status_badge(payload.get("scoring_status")),
            "personalized": _escape(payload.get("is_personalized")),
            "registered": _number(registered_units),
            "counted": _number(counted_units),
            "initial": _number(dfquad.get("root_base_score")),
            "support": _number(dfquad.get("aggregated_support")),
            "attack": _number(dfquad.get("aggregated_attack")),
            "dfquad": _number(payload.get("dfquad_score")),
            "linear": _number(payload.get("linear_empirical_score")),
        }
    ]
    hard_headers = (
        ("constraint_id", "ID"),
        ("target", "Cible"),
        ("source_text", "Demande originale"),
        ("status", "Statut"),
        ("requested", "Demandée"),
        ("metadata", "Métadonnée"),
        ("requested_canonical", "Canonique demandée"),
        ("metadata_canonical", "Canonique métadonnée"),
        ("reason", "Raison"),
        ("proof", "Preuve"),
    )
    soft_headers = (
        ("constraint_id", "ID"),
        ("target", "Cible canonique"),
        ("qualifiers", "Qualifiers"),
        ("source_text", "Demande originale"),
        ("importance_raw", "Importance"),
        ("normalized_weight", "Coefficient importance / 5"),
        ("weighting_method", "Méthode"),
        ("status", "Statut"),
        ("reason", "Raison"),
        ("proof", "Facility / métadonnée de preuve"),
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Évaluation argumentative — {_escape(payload.get('hotel_name'))}</title>
<style>
:root {{ --blue:#173a63; --green:#217a3c; --green-bg:#eaf7ee; --red:#b42318; --red-bg:#fff0ee; --ink:#17212b; --muted:#617080; --line:#dce3ea; --panel:#fff; --bg:#f4f7fa; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 Inter,Segoe UI,Arial,sans-serif; }}
.page {{ max-width:1600px; margin:auto; padding:24px; }} header,section {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 3px 14px rgba(23,58,99,.06); margin-bottom:18px; padding:20px; }}
h1 {{ margin:0 0 8px; color:var(--blue); }} h2 {{ color:var(--blue); margin:0 0 14px; font-size:20px; }} h3 {{ margin:0 0 10px; }} .subtitle {{ color:var(--muted); }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-top:16px; }} .metric {{ background:#f7f9fb; border:1px solid var(--line); border-radius:10px; padding:10px; }} .metric strong {{ display:block; color:var(--muted); font-size:12px; }}
.request {{ white-space:pre-wrap; background:#f7f9fb; border-left:4px solid var(--blue); padding:12px; border-radius:8px; }}
.table-scroll {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:760px; }} th,td {{ border-bottom:1px solid var(--line); text-align:left; vertical-align:top; padding:9px; }} th {{ background:#f7f9fb; color:#364554; position:sticky; top:0; }} code {{ white-space:normal; overflow-wrap:anywhere; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-weight:700; font-size:12px; }} .badge.ok {{ color:var(--green); background:var(--green-bg); }} .badge.bad {{ color:var(--red); background:var(--red-bg); }} .badge.unknown {{ color:#8a5b00; background:#fff6d8; }} .badge.neutral {{ color:#445; background:#edf1f5; }}
.graph-grid {{ display:grid; grid-template-columns:minmax(260px,1fr) 220px minmax(260px,1fr); gap:20px; align-items:center; }} .graph-column {{ display:grid; gap:12px; align-content:start; }} .argument-card,.root-card {{ border-radius:12px; padding:13px; }} .argument-card.support {{ background:var(--green-bg); border:2px solid var(--green); }} .argument-card.attack {{ background:var(--red-bg); border:2px solid var(--red); }} .argument-title {{ font-weight:800; }} .argument-force {{ font-weight:700; margin:4px 0; }} .root-column {{ display:grid; place-items:center; gap:10px; }} .root-card {{ background:var(--blue); color:#fff; text-align:center; display:grid; gap:5px; min-width:190px; }} .arrow {{ font-size:34px; font-weight:900; }} .support-arrow {{ color:var(--green); }} .attack-arrow {{ color:var(--red); }}
.empty {{ color:var(--muted); font-style:italic; }} .source-line {{ margin-bottom:4px; }} details {{ margin-top:8px; }}
@media (max-width:900px) {{ .graph-grid {{ grid-template-columns:1fr; }} .root-column {{ order:-1; }} .arrow {{ display:none; }} .page {{ padding:10px; }} }}
</style>
</head>
<body><main class="page">
<header>
  <h1>{_escape(payload.get('hotel_name'))}</h1>
  <div class="subtitle">hotel_id : <code>{_escape(payload.get('hotel_id'))}</code></div>
  <div class="summary">
    <div class="metric"><strong>Éligibilité</strong>{_status_badge(_mapping(payload.get('eligibility')).get('status'))}</div>
    <div class="metric"><strong>Mode</strong>{_escape(payload.get('argument_mode'))}</div>
    <div class="metric"><strong>Personnalisation</strong>{_escape(payload.get('is_personalized'))}</div>
    <div class="metric"><strong>Statut du scoring</strong>{_status_badge(payload.get('scoring_status'))}</div>
    <div class="metric"><strong>Pondération</strong>{_escape(payload.get('weighting_method'))}</div>
    <div class="metric"><strong>Unités enregistrées / comptées</strong>{_number(registered_units)} / {_number(counted_units)}</div>
    <div class="metric"><strong>Score DF-QuAD</strong>{_number(payload.get('dfquad_score'))}</div>
    <div class="metric"><strong>Baseline linéaire</strong>{_number(payload.get('linear_empirical_score'))}</div>
  </div>
</header>
<section><h2>Demande utilisateur</h2><div class="request">{_escape(preferences.get('original_text'))}</div></section>
<section><h2>1. Contraintes dures</h2>{_table(hard_headers, hard_rows)}</section>
<section><h2>2. Préférences qualitatives par aspect</h2>{_table((("aspect","Aspect"),("source_text","Demande originale"),("importance_raw","Importance"),("normalized_weight","Coefficient importance / 5"),("weighting_method","Méthode"),("hotel_evidence","Preuves hôtel")), _aspect_rows(payload))}</section>
<section><h2>3. Contraintes factuelles souples</h2>{_table(soft_headers, soft_rows)}</section>
<section><h2>4. Graphe argumentatif compté dans DF-QuAD</h2>{_graph_html(payload)}</section>
<section><h2>5. Arguments acceptés</h2>{_table((("id","ID"),("kind","Nature"),("type","Polarité"),("status","Statut score"),("unit","Unité"),("preferences","Préférences"),("sources","Sources"),("text","Texte")), _accepted_rows(validation), empty="Aucun argument hybride accepté.")}</section>
<section><h2>6. Arguments acceptés mais exclus du score</h2>{_table((("id","ID"),("reason","Raison"),("unit","Unité"),("counted","Argument compté")), excluded_rows, empty="Aucun argument accepté n'a été exclu.")}</section>
<section><h2>7. Propositions rejetées</h2>{_table((("id","ID"),("kind","Nature"),("type","Polarité"),("reasons","Raisons"),("proposal","Proposition brute")), _rejected_rows(validation), empty="Aucune proposition rejetée.")}</section>
<section><h2>8. Relations validées, uniquement explicatives</h2>{_table((("id","ID"),("source","Source"),("target","Cible"),("type","Relation"),("accepted","Validation"),("scoring","Effet score"),("reasons","Raisons")), _relation_rows(validation), empty="Aucune relation proposée.")}</section>
<section><h2>9. Preuves exactes et source_ref</h2>{_table((("argument_id","Argument"),("source_ref","source_ref"),("kind","Type de source"),("evidence","Preuve exacte"),("payload","Payload autorisé")), _evidence_rows(payload), empty="Aucune source référencée.")}</section>
<section><h2>10. Registre auditable des unités de score</h2>{_table((("id","Unité"),("kind","Nature"),("type","Polarité"),("intent","Intention"),("importance","Importance"),("weight","Coefficient / 5"),("method","Méthode"),("confidence","Wilson / confiance"),("formula","Formule"),("force","Force finale"),("counted","Comptée"),("reason","Raison")), _unit_rows(payload))}</section>
<section><h2>11. Calcul final</h2>{_table((("method","Méthode"),("status","Statut"),("personalized","Personnalisé"),("registered","Unités enregistrées"),("counted","Unités comptées DF-QuAD"),("initial","Score initial"),("support","Support agrégé"),("attack","Attaque agrégée"),("dfquad","Score DF-QuAD"),("linear","Baseline linéaire")), score_rows)}</section>
</main></body></html>"""


def render_hotel_graph(
    result: Mapping[str, Any] | object,
    output_path: str | Path | None = None,
) -> str:
    rendered = build_hotel_html(result)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one public hotel evaluation as autonomous HTML."
    )
    parser.add_argument("--input", required=True, help="Evaluation JSON path")
    parser.add_argument("--output", required=True, help="Output HTML path")
    arguments = parser.parse_args()
    payload = json.loads(Path(arguments.input).read_text(encoding="utf-8"))
    render_hotel_graph(payload, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
