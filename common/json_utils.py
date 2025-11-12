import json
import re

def fix_json_response(response: str) -> dict:
    """
    Makes a best-effort to coerce LLM output into valid JSON.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    response = re.sub(r'```json\n|```|json', '', response)
    response = response.replace('“', '"').replace('”', '"')
    response = re.sub(r'(\d+)/(\d+)', lambda m: str(float(m.group(1)) / float(m.group(2))), response)
    response = response.strip()

    match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', response)
    cleaned = match.group(0) if match else response

    open_curly = cleaned.count('{'); close_curly = cleaned.count('}')
    open_square = cleaned.count('['); close_square = cleaned.count(']')

    if open_curly == 1 and close_curly == 0:
        cleaned += '}'
    elif close_curly == 1 and open_curly == 0:
        cleaned = '{' + cleaned
    elif open_square == 1 and close_square == 0:
        cleaned += ']'
    elif close_square == 1 and open_square == 0:
        cleaned = '[' + cleaned

    if open_curly == 0 and close_curly == 0 and open_square == 0 and close_square == 0:
        cleaned = '{' + cleaned + '}'

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = cleaned.replace("'", '"').replace("\n", " ").replace("\t", " ")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                return json.loads(f"[{cleaned}]")
            except json.JSONDecodeError:
                raise ValueError("Unable to fix JSON response")


def extract_braces_content(s: str) -> str:
    m = re.search(r'\{(.*?)\}', s, re.DOTALL)
    return m.group(0) if m else ""
