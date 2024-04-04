
import unicodedata
def remove_quotes(text):
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    else:
        return text
def normalize_characters(text):
    # Normalize Greek characters
    greek_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω']
    for char in greek_chars:
        text = text.replace(char, unicodedata.normalize('NFC', char))

    # Normalize space characters
    space_chars = ['\xa0', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u202f', '\u205f', '\u3000']
    for space in space_chars:
        text = text.replace(space, ' ')

    # Normalize single quotes
    single_quotes = ['‘', '’', '‛', '′', '‹', '›', '‚', '‟']
    for quote in single_quotes:
        text = text.replace(quote, "'")

    # Normalize double quotes
    double_quotes = ['“', '”', '„', '‟', '«', '»', '〝', '〞', '〟', '＂']
    for quote in double_quotes:
        text = text.replace(quote, '"')

    # Normalize brackets
    brackets = {
        '【': '[', '】': ']',
        '（': '(', '）': ')',
        '｛': '{', '｝': '}',
        '〚': '[', '〛': ']',
        '〈': '<', '〉': '>',
        '《': '<', '》': '>',
        '「': '[', '」': ']',
        '『': '[', '『': ']',
        '〔': '[', '〕': ']',
        '〖': '[', '〗': ']'
    }
    for old, new in brackets.items():
        text = text.replace(old, new)

    # Normalize hyphens and dashes
    hyphens_and_dashes = ['‐', '‑', '‒', '–', '—', '―']
    for dash in hyphens_and_dashes:
        text = text.replace(dash, '-')

    # Normalize line breaks
    line_breaks = ['\r\n', '\r']
    for line_break in line_breaks:
        text = text.replace(line_break, '\n')

    # Normalize superscripts and subscripts to normal numbers
    superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    subscripts = '₀₁₂₃₄₅₆₇₈₉'
    normal_numbers = '0123456789'

    for super_, sub_, normal in zip(superscripts, subscripts, normal_numbers):
        text = text.replace(super_, normal).replace(sub_, normal)

    # Remove or normalize any remaining special characters using the 'NFKD' method
    text = unicodedata.normalize('NFKD', text)

    return remove_quotes(text)

def save_predictions(predictions, output_file):
    """
    Save model predictions to a file.

    :param predictions: A list of prediction labels.
    :param output_file: Path to the output file where predictions will be saved.
    """
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def load_predictions(input_file):
    """
    Load model predictions from a file.

    :param input_file: Path to the file containing predictions.
    :return: A list of prediction labels.
    """
    predictions = []
    with open(input_file, 'r') as f:
        for line in f:
            predictions.append(line.strip())
    return predictions
