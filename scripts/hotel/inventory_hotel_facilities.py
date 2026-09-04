from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.hotel import FacilityOntology, normalize_facility_text


def build_facility_inventory(
    raw_dataset: object,
    ontology: FacilityOntology,
) -> dict[str, Any]:
    if not isinstance(raw_dataset, dict):
        raise ValueError("hotel profiles must be a JSON object")
    hotels = raw_dataset.get("hotels")
    if not isinstance(hotels, list):
        raise ValueError("hotel profiles must contain a hotels list")

    hotels_by_pair: dict[tuple[int, str], set[str]] = defaultdict(set)
    names_by_id: dict[int, set[str]] = defaultdict(set)
    ids_by_name: dict[str, set[int]] = defaultdict(set)
    display_name_by_normalized: dict[str, str] = {}
    for hotel_index, hotel in enumerate(hotels):
        if not isinstance(hotel, dict):
            continue
        hotel_id = str(hotel.get("hotel_id", f"hotel_index_{hotel_index}"))
        metadata = hotel.get("hotel_metadata", {})
        facilities = metadata.get("facilities", []) if isinstance(
            metadata, dict
        ) else []
        if not isinstance(facilities, list):
            continue
        for facility in facilities:
            if not isinstance(facility, dict):
                continue
            facility_id = facility.get("facility_id", facility.get("id"))
            name = facility.get("name", facility.get("facility_name"))
            if isinstance(facility_id, bool) or not isinstance(
                facility_id, int
            ):
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            clean_name = name.strip()
            normalized_name = normalize_facility_text(clean_name)
            hotels_by_pair[(facility_id, clean_name)].add(hotel_id)
            names_by_id[facility_id].add(clean_name)
            ids_by_name[normalized_name].add(facility_id)
            display_name_by_normalized.setdefault(
                normalized_name,
                clean_name,
            )

    observed_pairs = []
    recognized = []
    unmapped = []
    for (facility_id, name), hotel_ids in sorted(
        hotels_by_pair.items(),
        key=lambda item: (item[0][1].casefold(), item[0][0]),
    ):
        mapping = ontology.get_mapping(facility_id)
        name_matches = bool(
            mapping
            and normalize_facility_text(name)
            in {
                normalize_facility_text(expected)
                for expected in mapping.expected_names
            }
        )
        row = {
            "facility_id": facility_id,
            "facility_name": name,
            "hotel_frequency": len(hotel_ids),
            "recognized": name_matches,
        }
        observed_pairs.append(row)
        if name_matches and mapping is not None:
            recognized.append(
                {
                    **row,
                    "capability": mapping.capability,
                    "qualifiers": dict(mapping.qualifiers),
                }
            )
        else:
            reason = (
                "facility_id_name_mismatch"
                if mapping is not None
                else "unmapped_facility_id"
            )
            unmapped.append({**row, "reason": reason})

    id_name_conflicts = [
        {"facility_id": facility_id, "facility_names": sorted(names)}
        for facility_id, names in sorted(names_by_id.items())
        if len(names) > 1
    ]
    name_id_conflicts = [
        {
            "facility_name": display_name_by_normalized[normalized_name],
            "facility_ids": sorted(ids),
        }
        for normalized_name, ids in sorted(ids_by_name.items())
        if len(ids) > 1
    ]
    return {
        "schema_version": "1.0",
        "scope": {
            "dataset_name": raw_dataset.get("dataset_name"),
            "n_hotels_declared": raw_dataset.get("n_hotels"),
            "n_hotels_inspected": len(hotels),
            "note": (
                "Observed current-dataset facilities only; not an exhaustive "
                "provider catalogue."
            ),
        },
        "summary": {
            "unique_id_name_pairs": len(observed_pairs),
            "recognized_pairs": len(recognized),
            "unmapped_pairs": len(unmapped),
            "ids_with_multiple_names": len(id_name_conflicts),
            "names_with_multiple_ids": len(name_id_conflicts),
        },
        "observed_facilities": observed_pairs,
        "ids_with_multiple_names": id_name_conflicts,
        "names_with_multiple_ids": name_id_conflicts,
        "recognized_facilities": recognized,
        "unmapped_facilities": unmapped,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory observed facility id/name pairs and compare them with "
            "the deterministic canonical ontology."
        )
    )
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profiles_path = Path(args.profiles)
    raw_dataset = json.loads(profiles_path.read_text(encoding="utf-8"))
    ontology = FacilityOntology.load(args.ontology)
    inventory = build_facility_inventory(raw_dataset, ontology)
    rendered = json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(json.dumps(inventory["summary"], indent=2))


if __name__ == "__main__":
    main()
