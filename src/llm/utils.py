import json


def extract_first_json_object(text: str):
    """
    - extracts everything between the first '{' and the last '}'
    - tries to parse it as JSON
    - returns None if parsing fails
    """
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        return None

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None