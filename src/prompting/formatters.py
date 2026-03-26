def format_history(history: list[dict]) -> str:
    lines = []

    for i, item in enumerate(history, start=1):
        name = item.get("name", "Unknown")
        categories = ", ".join(item.get("categories", []))
        rating = item.get("user_stars", "N/A")

        line = (
            f"{i}. {name} | "
            f"categories: {categories} | "
            f"rating: {rating}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_attributes(attributes: dict) -> str:
    if not attributes:
        return "none"

    parts = []
    for key, value in attributes.items():
        parts.append(f"{key}: {value}")

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