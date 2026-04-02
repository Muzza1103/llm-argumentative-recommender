ATTRIBUTE_LABELS = {
    "RestaurantsPriceRange2": "price_range",
    "RestaurantsTakeOut": "takeout",
    "RestaurantsDelivery": "delivery",
    "OutdoorSeating": "outdoor_seating",
    "RestaurantsAttire": "attire",
    "Alcohol": "alcohol",
    "NoiseLevel": "noise",
    "RestaurantsGoodForGroups": "good_for_groups",
    "GoodForKids": "good_for_kids",
    "RestaurantsReservations": "reservations",
}

IMPORTANT_ATTRIBUTES = list(ATTRIBUTE_LABELS.keys())


def format_history(history: list[dict]) -> str:
    lines = []

    for i, item in enumerate(history, start=1):
        name = item.get("name", "Unknown")
        categories = ", ".join(item.get("categories", []))
        rating = item.get("user_stars", "N/A")
        attributes = format_attributes(item.get("attributes", {}))

        line = (
            f"{i}. {name} | "
            f"categories: {categories} | "
            f"rating: {rating}"
        )

        if attributes != "none":
            line += f" | attributes: {attributes}"

        lines.append(line)

    return "\n".join(lines)


def clean_attribute_value(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        value = value.replace("u'", "'")
        value = value.strip("'")

        if value.lower() == "none":
            return None

    return value


def format_attributes(attributes: dict) -> str:
    if not attributes:
        return "none"

    parts = []

    for key in IMPORTANT_ATTRIBUTES:
        if key not in attributes:
            continue

        value = clean_attribute_value(attributes[key])
        if value is None:
            continue

        label = ATTRIBUTE_LABELS[key]
        parts.append(f"{label}={value}")

    if not parts:
        return "none"

    return "; ".join(parts)


def format_target_item(target_item: dict) -> str:
    name = target_item.get("name", "Unknown")
    categories = ", ".join(target_item.get("categories", []))
    attributes = format_attributes(target_item.get("attributes", {}))
    global_stars = target_item.get("global_stars", "N/A")

    return (
        f"{name} | "
        f"categories: {categories} | "
        f"attributes: {attributes} | "
        f"global_rating: {global_stars}"
    )