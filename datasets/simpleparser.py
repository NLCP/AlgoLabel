from preprocessing.text.util import replace_chars
from nltk import sent_tokenize
from parse import parse
import unicodedata
import re


def parse_number(token: str):
    """Check if string token denotes a number"""

    try:
        number = int(token)
    except:
        try:
            if len(token) > 1:
                number = "".join([str(unicodedata.digit(ch)) for ch in token])
            else:
                number = unicodedata.digit(token)
        except:
            return None

    return int(number)


def wrap_power(token: str):
    """Normalize surface form for pow expressions"""
    temp = replace_chars(token, {"{": "@",
                                 "}": "@",
                                 " ": ""})
    res = parse("{}^@{}@", temp)
    if not res:
        return token

    base, exp = res

    if not base.lstrip('-').isdigit():
        return token

    if not base.lstrip('-').isdigit():
        return token

    exp = parse_number(exp)
    return exp


def wrap_digit(token):
    """Normalize surface form for numeric expressions"""
    if "^" in token and "{" in token and "}" in token:
        exp = wrap_power(token)
        if not exp:
            return token

        return "10^{}".format(exp)

    if not token.isdigit():
        return token

    number = parse_number(token)

    if not number:
        return token

    if 1 <= number <= 10:
        return token

    if 10 < number <= 50:
        return '50'

    if 50 < number <= 100:
        return '100'

    if 100 < number <= 500:
        return '500'

    num_digits = len(token) - 1
    return "10^{}".format(num_digits)


def digit_parser(args, sample, content):
    operators = ["+", "-", "/", "*", ",", "(", ")"]
    op_map = {op: " {} ".format(op) for op in operators}

    content = replace_chars(content, op_map)
    tokens = content.split()

    new_tokens = []
    ignore_next_token = False

    for idx, token in enumerate(tokens):

        if "^{" in token and "}" in token:
            new_tokens[-1] += token
            continue

        # ^{ - 6}
        if "^{" in token and idx + 2 < len(tokens) and tokens[idx + 1] == "-":
            tokens[idx + 2] = "^{-" + tokens[idx + 2]
            ignore_next_token = True
            continue

        if not ignore_next_token:
            new_tokens.append(token)
        else:
            ignore_next_token = False

        if idx + 1 < len(tokens) and token.isdigit() and tokens[idx + 1].isdigit():
            new_tokens[-1] += tokens[idx + 1]
            ignore_next_token = True

    # return content
    return " ".join(map(wrap_digit, new_tokens))


def is_index(token):
    """Verify if token denotes an index expression"""

    res = parse("{}_{}", token)
    if not res:
        if len(token) == 2:
            res = token[:-1], token[-1]
        elif len(token) == 4:
            if token[2] in {"-", "+"}:
                res = token[0], "{} {} {}".format(token[1], token[2], token[3])

    if not res:
        return False

    var, index = res
    if "{" in index and "}" in index:
        # print(token)
        # print(var, index)
        # exit(0)
        index = index[1:-1]

    for token in index:
        if not token.isalpha() and \
                not token.isdigit() and \
                not token in {"^", "+", "-", "*", "/", "_", " "}:
            return False

    return var, index


def parse_pair(token):
    res = parse("( {} , {} )", token)

    if not res:
        return False

    for ch in res[0] + res[1]:
        if is_operator(ch):
            return False

    if has_range_delim(res[0]) or has_range_delim(res[1]):
        return False

    return "pair ( {} , {} )".format(res[0], res[1])


def is_operator(token):
    for ops in [",", "\ldots", "\dots", "\cdot", "..."]:
        if token in ops:
            return True
    return False


def has_range_delim(token):
    for delim in {">", "geq", "leq", "leqslant", "<", "<q", "\\le", "\\leq", "\\leqslant"}:
        if delim in token:
            return True
    return False


def parse_range(stream):
    result = {
        "start": None,
        "vars": [],
        "end": None
    }

    delim = {">", "geq", "leq", "leqslant", "<", "<q", "\\le", "\\leq", "\\leqslant"}

    # print("[parse_range] 1", stream)

    def clean_end(end):

        toks = re.split(r"\\\\cdot|\\cdot|\*", end)
        if len(toks) == 2:
            return toks[-1]

        return end

    def clean_start(start):

        if len(start) > 0 and start[0] == "-" and len(start.split()) == 2:
            return "".join(start.split())
        return start

    if "," in stream:
        stream = stream.split(",")

        # print("[parse_range] 2", stream)

        if len(stream) == 2:
            lhs, rhs = stream

            lhs_tks = lhs.split()
            rhs_tks = rhs.split()

            if len(lhs_tks) == 3 and len(rhs_tks) == 3 and lhs_tks[1] in delim and rhs_tks[1] in delim:
                end = clean_end(rhs_tks[2]).strip()
                return "range ( {} , {} ) , range ( {} , {} )".format(lhs_tks[2], end, rhs_tks[0], end)

        result = map(parse_range, stream)
        for item in result:
            if item is None:
                return None

        # print("[parse_range] 3", stream)

        return " , ".join(result)

    tokens = re.split(r"\\leqslant|\\leq|\\geq|\\le|\\ge|leq|geq|<q|<|>", stream)

    if len(tokens) <= 2:

        if len(tokens) == 2:

            start = tokens[0]
            end = tokens[1]

            if (end.isdigit() or "^" in end) and not start.isdigit():
                return "range ( {} , {} )".format(start, end)

        return None

    result["start"] = tokens[0]
    result["end"] = tokens[-1]
    result["vars"] = tokens[1:-1]

    if len(result["vars"]) == 0:
        return None

    result["start"] = clean_start(result["start"])

    if len(result["start"].split()) > 1:
        return None

    result["end"] = clean_end(result["end"])
    vars = ",".join(result["vars"])

    seq = parse_sequence(vars)
    if seq:
        vars = seq

    return "range ( {} , {} )".format(vars.strip(), result["end"])


def parse_sequence(stream):
    result = {
        "seq": None,
        "seq_size": None
    }

    count_index = 0
    op_count = 0
    stream = stream.replace(" - ", "-").split()

    # print("\n")
    # print(stream)

    for token in stream:

        token = token.replace(",", "")

        # print(token)

        if is_operator(token):
            op_count += 1
            continue

        # print("check index")
        res = is_index(token)
        if not res:
            # print("check pair", token)
            res = parse_pair(token)

        # print("result: ", res)

        if res:
            count_index += 1

            if result["seq"] and result["seq"] != res[0]:
                return None

            result["seq"] = res[0]
            result["seq_size"] = res[1]
        else:
            break

    # print(count_index, op_count, result)
    if count_index > 1 and op_count >= 1 and result["seq"] and result["seq_size"]:
        # print(result)
        result = "sequence ( {} , {} )".format(result["seq"], result["seq_size"])
        # print(result)
        # print("!!!")
        # exit(0)
        return result
    else:
        return None


def parse_simple_operation(formula):
    tokens = formula.split()

    if len(tokens) == 3 and tokens[1] in {"-", "+", "*", "="}:

        if "^" in tokens[0] and tokens[2].isdigit():
            return tokens[0]

        if tokens[1] == "=":
            return "set ( {} )".format(tokens[0])

        # n - 1  || n + 1

        if not tokens[1].isdigit() and tokens[2].isdigit():
            return tokens[0]

        if tokens[0].isdigit() and not tokens[2].isdigit():
            return tokens[2]

    return None


def simplify_formula(formula):
    formula = formula.strip()

    if len(formula) == 0:
        return formula

    res = parse_pair(formula)
    if res:
        return res.strip()

    if formula[0] == "(" and formula[-1] == ")":
        formula = formula[1:-1]

    res = parse_range(formula)
    if not res:
        res = parse_sequence(formula)
    # if not res:
    #     res = parse_simple_operation(formula)

    if res:
        return res.strip()

    return formula.strip()


def formula_parser(args, sample, content):
    if "$$$" not in content:
        return content

    if "formulas" not in sample:
        sample["formulas"] = {}
        sample["formulas_idx"] = {}

    content = "$$$ {} $$$".format(content.lower())

    formula_pattern = "(?<=\$\$\$)(.*?)(?=\$\$\$)"
    formulas = re.findall(formula_pattern, content)

    for idx, chunk in enumerate(formulas):
        if idx % 2 == 1:
            formulas[idx] = simplify_formula(chunk)

    formulas = list(map(simplify_formula, formulas))

    for idx, formula in enumerate(formulas[1::2]):
        actual_index = 1 + idx * 2

        f = formulas[actual_index].strip()
        if len(f) == 0:
            continue

        fs = f.split("@@@")

        placeholders = ""

        for f in fs:

            f = " ".join(f.split())

            if f not in args["formulas"]:

                placeholder = " |formula{}| ".format(len(args["formulas"]))
                args["formulas"][f] = placeholder

                sample["formulas"][f] = placeholder
                sample["formulas_idx"][placeholder] = f

                placeholders += placeholder + " "

            else:
                placeholder = args["formulas"][f]

                sample["formulas"][f] = placeholder
                sample["formulas_idx"][placeholder] = f

                placeholders += placeholder + " "

            args["formulas_count"][f] += 1

        formulas[actual_index] = placeholders

    formulas = map(lambda x: x.strip(), formulas)
    content = " ".join(formulas).strip()
    return content


def split_content(args, content):
    delims = [".", "?", "!", ",", ";"]

    for delim in delims:
        content = content.replace(delim, " {} ".format(delim))
    return sent_tokenize(content)
