import re
from typing import List, Optional, Tuple, Any


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def compact_plate_text(text: str) -> str:
    """Clean OCR text and keep only letters and numbers."""

    if not text:
        return ""

    text = str(text).upper()

    text = text.replace("|", "I")
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("\t", "")

    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


def normalize_plate_text(text: str) -> str:
    """Normalize OCR plate text."""
    compacted = compact_plate_text(text)
    if not compacted or validate_nigerian_plate(compacted):
        return compacted

    for match in re.finditer(r"[A-Z]{3}[0-9]{3}[A-Z]{2}", compacted):
        candidate = match.group(0)
        if validate_nigerian_plate(candidate):
            return candidate

    return compacted


# ============================================================
# NIGERIAN PLATE VALIDATION
# ============================================================

def validate_nigerian_plate(text: str) -> bool:
    """
    Expected Nigerian plate structure:

        LLLNNNLL

    Example:

        KJA123AA
        MUS993HJ
        SMK919HM
        LSD100GX
    """

    if not text:
        return False

    text = compact_plate_text(text)

    pattern = r"^[A-Z]{3}[0-9]{3}[A-Z]{2}$"

    return bool(re.match(pattern, text))


def is_plausible_nigerian_plate(text: str) -> bool:
    """Check whether OCR text follows the expected plate structure."""

    if not text:
        return False

    text = compact_plate_text(text)

    if len(text) != 8:
        return False

    if not text[:3].isalpha():
        return False

    if not text[3:6].isdigit():
        return False

    if not text[6:8].isalpha():
        return False

    return True


def is_ocr_quality_plate(text: str) -> bool:
    """Final OCR plate quality check."""

    if not text:
        return False

    text = compact_plate_text(text)

    return validate_nigerian_plate(text)


# ============================================================
# OCR CONFUSION TABLES
# ============================================================

OCR_DIGIT_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}


OCR_CHARACTER_CONFUSIONS = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L"],
    "2": ["Z"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],

    "O": ["0", "D", "Q"],
    "D": ["0", "O"],
    "Q": ["0", "O"],
    "I": ["1", "L"],
    "L": ["1", "I"],
    "Z": ["2"],
    "S": ["5"],
    "G": ["6"],
    "T": ["7"],
    "B": ["8"],

    # OCR can confuse M and H.
    "M": ["H"],
    "H": ["M"],
}


# ============================================================
# LETTER POSITION CORRECTION
# ============================================================

def _correct_letter_position(char: str) -> str:
    """Correct common OCR mistakes in a letter position."""

    corrections = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }

    return corrections.get(char, char)


# ============================================================
# DIGIT POSITION CORRECTION
# ============================================================

def _correct_digit_position(char: str) -> str:
    """Correct common OCR mistakes in a digit position."""

    corrections = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "T": "7",
        "B": "8",
    }

    return corrections.get(char, char)


# ============================================================
# APPLY PLATE POSITIONS
# ============================================================

def _apply_plate_positions(text: str) -> str:
    """
    Nigerian plate structure:

        0-2 = letters
        3-5 = numbers
        6-7 = letters
    """

    if len(text) != 8:
        return text

    chars = list(text)

    # First three characters
    for i in range(3):
        chars[i] = _correct_letter_position(chars[i])

    # Three numbers
    for i in range(3, 6):
        chars[i] = _correct_digit_position(chars[i])

    # Last two characters
    for i in range(6, 8):
        chars[i] = _correct_letter_position(chars[i])

    return "".join(chars)


# ============================================================
# HANDLE 9 CHARACTER OCR RESULT
# ============================================================

def _reduce_nine_character_plate(text: str) -> str:
    """
    Handle OCR output such as:

        SHKF919HM

    Expected:

        LLLNNNLL

    OCR result:

        LLLLNNNLL

    Remove the fourth leading character:

        SHKF919HM
           ↓
        SHK919HM
    """

    if len(text) != 9:
        return text

    pattern = r"^[A-Z]{4}[0-9]{3}[A-Z]{2}$"

    if re.match(pattern, text):

        return text[:3] + text[4:]

    return text


# ============================================================
# MAIN POSITIONAL CORRECTION
# ============================================================

def correct_positional_plate(
    text: str
) -> Tuple[str, bool]:
    """
    Correct OCR output using the Nigerian plate structure.

    Returns:

        corrected_text
        correction_applied
    """

    if not text:
        return "", False

    original = compact_plate_text(text)

    if not original:
        return "", False

    # Already correct
    if validate_nigerian_plate(original):
        return original, False

    corrected = original
    correction_applied = False

    # --------------------------------------------------------
    # 9 character OCR result
    # --------------------------------------------------------

    if len(corrected) == 9:

        reduced = _reduce_nine_character_plate(corrected)

        if reduced != corrected:
            corrected = reduced
            correction_applied = True

    # --------------------------------------------------------
    # 8 character OCR result
    # --------------------------------------------------------

    if len(corrected) == 8:

        positioned = _apply_plate_positions(corrected)

        if positioned != corrected:
            corrected = positioned
            correction_applied = True

    return corrected, correction_applied


# ============================================================
# GENERATE ALTERNATIVES
# ============================================================

def generate_plate_alternatives(text: str) -> List[str]:
    """
    Generate possible OCR interpretations.
    """

    if not text:
        return []

    text = compact_plate_text(text)

    if not text:
        return []

    alternatives = []

    # Original
    alternatives.append(text)

    # Corrected
    corrected, _ = correct_positional_plate(text)

    if corrected and corrected not in alternatives:
        alternatives.append(corrected)

    # Only generate alternatives for valid length
    if len(corrected) != 8:
        return alternatives

    for i in range(len(corrected)):

        char = corrected[i]

        replacements = OCR_CHARACTER_CONFUSIONS.get(
            char,
            []
        )

        for replacement in replacements:

            candidate = (
                corrected[:i]
                + replacement
                + corrected[i + 1:]
            )

            if candidate not in alternatives:
                alternatives.append(candidate)

    return alternatives


# ============================================================
# EDIT DISTANCE
# ============================================================

def _plate_edit_distance(a: str, b: str) -> int:
    """Calculate Levenshtein edit distance."""

    a = compact_plate_text(a)
    b = compact_plate_text(b)

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):

        current = [i]

        for j, char_b in enumerate(b, start=1):

            insertion = current[j - 1] + 1
            deletion = previous[j] + 1

            if char_a == char_b:
                substitution = previous[j - 1]
            else:
                substitution = previous[j - 1] + 1

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# DATABASE VERIFICATION
# ============================================================

def verify_plate_with_database(
    ocr_text: str,
    database_lookup: Any = None,
    max_distance: int = 1
) -> Tuple[str, bool, str]:
    """
    Verify OCR plate against the project's database.

    IMPORTANT:

    The existing pipeline.py expects THREE return values:

        corrected_text,
        correction_applied,
        correction_reason

    The second argument is the database lookup METHOD,
    not a list.
    """

    if not ocr_text:

        return "", False, ""

    original = compact_plate_text(ocr_text)

    if not original:

        return "", False, ""

    # --------------------------------------------------------
    # First perform normal OCR correction.
    # --------------------------------------------------------

    corrected, correction_applied = (
        correct_positional_plate(original)
    )

    correction_reason = ""

    if correction_applied:

        correction_reason = (
            "Corrected OCR text using Nigerian plate "
            "position rules."
        )

    # --------------------------------------------------------
    # If there is no database lookup method, return OCR result.
    # --------------------------------------------------------

    if database_lookup is None:

        return (
            corrected,
            correction_applied,
            correction_reason
        )

    # --------------------------------------------------------
    # Database lookup
    # --------------------------------------------------------

    try:

        # The project's plate_registry.lookup is a METHOD.
        #
        # Call it instead of trying to iterate over it.

        database_result = database_lookup(
            corrected
        )

    except TypeError:

        # Some lookup methods may expect the original OCR text.
        try:

            database_result = database_lookup(
                original
            )

        except Exception:

            database_result = None

    except Exception:

        database_result = None

    # --------------------------------------------------------
    # No database match
    # --------------------------------------------------------

    if database_result is None:

        return (
            corrected,
            correction_applied,
            correction_reason
        )

    # --------------------------------------------------------
    # Handle different possible database return formats.
    # --------------------------------------------------------

    matched_plate = None

    # String result
    if isinstance(database_result, str):

        matched_plate = compact_plate_text(
            database_result
        )

    # Dictionary result
    elif isinstance(database_result, dict):

        possible_keys = [
            "plate_number",
            "plate",
            "plate_number_text",
            "registration_number",
            "text",
        ]

        for key in possible_keys:

            value = database_result.get(key)

            if value:

                matched_plate = compact_plate_text(
                    str(value)
                )

                break

    # Object result
    else:

        for attribute in [
            "plate_number",
            "plate",
            "registration_number",
            "text",
        ]:

            try:

                value = getattr(
                    database_result,
                    attribute,
                    None
                )

                if value:

                    matched_plate = compact_plate_text(
                        str(value)
                    )

                    break

            except Exception:
                pass

    # --------------------------------------------------------
    # If database returned a plate, use it.
    # --------------------------------------------------------

    if matched_plate:

        if matched_plate != corrected:

            return (
                matched_plate,
                True,
                "OCR result corrected using database match."
            )

        return (
            matched_plate,
            correction_applied,
            correction_reason
        )

    # --------------------------------------------------------
    # No usable match
    # --------------------------------------------------------

    return (
        corrected,
        correction_applied,
        correction_reason
    )


# ============================================================
# FORMAT PLATE
# ============================================================

def format_plate_text(text: str) -> str:
    """Format plate number for display."""

    if not text:
        return ""

    return compact_plate_text(text)


# ============================================================
# FINAL PLATE PROCESSOR
# ============================================================

def process_plate_text(text: str) -> str:
    """
    Complete OCR plate processing.
    """

    if not text:
        return ""

    cleaned = compact_plate_text(text)

    if not cleaned:
        return ""

    corrected, _ = correct_positional_plate(
        cleaned
    )

    if validate_nigerian_plate(corrected):
        return corrected

    return ""


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    tests = [
        "MUS993HJ",
        "MU5993HJ",
        "SMK919HM",
        "SHKF919HM",
        "LSD100GX",
        "KJA123AA",
        "ABC123XY",
        "LAGOS",
    ]

    print("=" * 60)
    print("NIGERIAN PLATE NORMALIZATION TEST")
    print("=" * 60)

    for test in tests:

        cleaned = compact_plate_text(test)

        corrected, correction_applied = (
            correct_positional_plate(test)
        )

        valid = validate_nigerian_plate(
            corrected
        )

        print()
        print("Input:              ", test)
        print("Cleaned:            ", cleaned)
        print("Corrected:          ", corrected)
        print("Correction applied: ", correction_applied)
        print("Valid:              ", valid)

    print()
    print("=" * 60)