from nltk import word_tokenize

map_chars = {
    '\u2009': " ",
    '\u2264': "leq",
    '\u2014': '-',
    '\u2013': '-',
    '\u2260': "neq",
    '\u2265': "geq",
    '\u00b7': " * ",
    '\u00d7': " * ",
    '\u00a0': " ",
    "\u0421": "C",
    "\u2019": '\'',
    "\u0441": 'c',
    "\u230a": "lceil",
    "\u2308": "lceil",
    "\u230b": 'rceil',
    "\u2309": "rceil",
    "\u0445": "x",
    "\u0438": ";",
    "\u00ab": "\"",
    "\u00bb": "\"",
    "\u200b": "",
    "\u0410": "A",
    "\u0456": "i",
    "\u2116": "No",
    "\u010d": "c",
    "\u0425": "X",
    "\u03b7": "eta",
    "\u2192": "->",
    "\u03a8": "psi",
    "\u03c8": "psi",
    "\u03c0": "pi",
    "\u03c6": "phi",
    "\u2026": "...",
    "\u041c": "M",
    "\u00b0": " degree ",
    "\u221e": " inf ",
    "\u0412": "B",
    "\u03c1": "rho",
    "\u03b2": "beta",
    "\u03b1": "alpha",
    "\u2248": "asymp",
    "\u0430": "a",
    "\u03b5": "eps",
    "\u202f": " ",
    "\u2122": " ",
    "\u00e0": "a",
    "\u2208": "in",
    "\u03c9": "omega",
    "\u2002": " ",
    "\u044e": " ",
    "\u0422": "T",
    "\u00e1": "a",
    "\u0161": "s",
    "\uff1a": ":",
    "\u00e6": "",
    "\u2212": "-",
    "\u0435": "e",
    "\u2022": " ",
    "\u007f": " ",
    "\u201c": "\"",
    "\u201d": "\"",
    "\u044d": "e",
    "\u0442": "t",
    "\u0131": "i",
    "\u043e": "o",
    "\u2261": "=",
    "\u041e": "O",
    "\u00f7": "/",
    "\u2018": "'",
    "\u2061": "",
    "\u03a3": "sum",
    "\u0417": "",
    "\u732b": "",
    "\u5c4b": "",
    "\u0151": "o",
    "\u03bb": "lambda",
    "\u00b9": "1",
    "\\,": "",
    "\\\\cdot": "*",
    "\\\\times": "*",
    "\\le": "<",
}

stopwords = {"the", "of", "to", "and", "is", "that", "it", "with"}


def replace_chars(text, char_map) -> str:
    """Normalizes the surface form of special tokens, particularly those appearing in formulas"""
    for ch in char_map:
        text = text.replace(ch, char_map[ch])
    return text


def clean_text(text) -> str:
    return replace_chars(text, map_chars)


def clean_string(content) -> str:
    content = " ".join(clean_text(content).split())
    return content


def filter_sentence(sentence):
    """Splits sentence into tokens and removes stopwords"""
    return [x.lower() for x in word_tokenize(sentence)
            if len(x) >= 1 and x.lower() not in stopwords]
