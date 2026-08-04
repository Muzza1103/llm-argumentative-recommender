from __future__ import annotations

from .errors import HotelDataValidationError


HOTEL_ASPECTS: tuple[str, ...] = (
    "localisation_transport",
    "personnel_accueil_service",
    "proprete_hygiene",
    "chambre_taille_confort",
    "salle_de_bain",
    "petit_dejeuner_restauration",
    "bruit_calme",
    "prix_valeur",
    "climatisation_chauffage_temperature",
    "wifi_internet",
    "parking_voiture",
    "equipements_chambre",
    "piscine_spa_bien_etre",
    "accessibilite_batiment",
    "vue_environnement",
)

HOTEL_ASPECT_SET = frozenset(HOTEL_ASPECTS)


def validate_hotel_aspect(value: object, *, path: str = "aspect") -> str:
    """Return an official aspect name or fail without silently normalizing it."""
    if not isinstance(value, str) or not value:
        raise HotelDataValidationError(
            "expected a non-empty aspect name",
            path=path,
        )

    if value not in HOTEL_ASPECT_SET:
        raise HotelDataValidationError(
            f"unknown hotel aspect {value!r}",
            path=path,
        )

    return value
