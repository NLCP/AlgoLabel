
import logging
from datasets.dataset import Dataset
from parse import parse

import re
from tqdm import tqdm
import unicodedata
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize


class TextDataset(Dataset):

    def __init__(self, args, data):
        super().__init__(args, data)
        self.params = self.args["prepare"]["text"]

        relevant_fields = self.params["fields"]
        self.fields     = [field for field in relevant_fields
                           if relevant_fields[field]]

        self.formulas       = {}
        self.formulas_count = defaultdict(int)

    def filter_irrelevant_tasks(self):

        result = []
        for sample in self.data:
            if "contest_name" in sample and "April Fools" in sample["contest_name"]:
                continue

            for field in self.fields:
                if field not in sample:
                    break
            else:
                result.append(sample)

        return result

    def prepare(self):

        # Remove problems with no statements
        self.data = self.filter_irrelevant_tasks()

        logging.debug("Preparing statements for {} problems..".format(len(self.data)))

        count_solutions = 0
        skipped         = 0
        result          = []

        for sample in tqdm(self.data):

            sample = self.prepare_problem(sample)
            if not sample:
                skipped += 1
            else:
                result.append(sample)

        self.data = result
        logging.info("{} problems with invalid statements skipped..".format(skipped))
        logging.info("Prepared {} problems with {} statements".format(len(self.data), count_solutions))
        return result

    @staticmethod
    def _parse_number(token):

        try:
            number = int(token)
        except ValueError:
            try:
                if len(token) > 1:
                    number = "".join([str(unicodedata.digit(ch)) for ch in token])
                else:
                    number = unicodedata.digit(token)
            except ValueError:
                return None

        return int(number)

    @staticmethod
    def _wrap_power(token):

        ch_map = {
            "{": "@",
            "}": "@",
            " ": ""
        }

        temp = TextDataset._replace_chars(token, ch_map)
        res = parse("{}^@{}@", temp)
        if not res:
            return token

        base, exp = res

        if not base.lstrip('-').isdigit():
            return token

        if not base.lstrip('-').isdigit():
            return token

        exp = TextDataset._parse_number(exp)
        return exp

    @staticmethod
    def _wrap_digit(token):
        if "^" in token and "{" in token and "}" in token:
            exp = TextDataset._wrap_power(token)
            if not exp:
                return token

            return "10^{}".format(exp)

        if not token.isdigit():
            return token

        number = TextDataset._parse_number(token)

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

    @staticmethod
    def _replace_chars(text, char_map):
        for ch in char_map:
            text = text.replace(ch, char_map[ch])
        return text

    def _parse_digits(self, content, sample):
        """
            Normalize surface form of numbers appearing in text.
        """
        operators = ["+", "-", "/", "*", ",", "(", ")"]
        op_map = {op: " {} ".format(op) for op in operators}

        content = self._replace_chars(content, op_map)
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

        return " ".join(map(TextDataset._wrap_digit, new_tokens))

    @staticmethod
    def _is_index(token):

        res = parse("{}_{}", token)
        if not res:
            if len(token) == 2:
                res = token[:-1], token[-1]

        if not res:
            return False

        var, index = res
        if "{" in index and "}" in index:
            index = index[1:-1]

        for token in index:
            if token.isalpha() or token.isdigit() or token in {"^", "+", "-", "*", "/", "_"}:
                continue
            return False

        return var, index

    @staticmethod
    def _parse_pair(token):
        res = parse("( {} , {} )", token)
        if not res:
            return False
        return "pair ( {} , {} )".format(res[0], res[1])

    @staticmethod
    def _is_operator(token):
        if token in {",", "\ldots", "\dots", "\cdot", "..."}:
            return True

    @staticmethod
    def _parse_range(stream):

        result = {
            "start": None,
            "vars": [],
            "end": None
        }

        delim = {">", "geq", "leq", "leqslant", "<", "\\le", "\\leq", "\\leqslant"}

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

            if len(stream) == 2:
                lhs, rhs = stream

                lhs_tks = lhs.split()
                rhs_tks = rhs.split()

                if len(lhs_tks) == 3 and len(rhs_tks) == 3 and lhs_tks[1] in delim and rhs_tks[1] in delim:
                    end = clean_end(rhs_tks[2]).strip()
                    return "range ( {} , {} ) @@@range ( {} , {} )".format(lhs_tks[2], end, rhs_tks[0], end)

            result = map(TextDataset._parse_range, stream)
            for item in result:
                if item is None:
                    return None
            return " , ".join(result)

        tokens = re.split(r"\\leqslant|\\leq|\\geq|\\le|\\ge|leq|geq|<|>", stream)

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
        vars          = ",".join(result["vars"])
        seq           = TextDataset._parse_sequence(vars)
        if seq:
            vars = seq

        return "range ( {} , {} )".format(vars.strip(),
                                          result["end"])

    @staticmethod
    def _parse_sequence(stream):

        result = {
            "seq": None,
            "seq_size": None
        }

        count_index = 0
        op_count = 0
        stream = stream.replace(" - ", "-").split()

        for token in stream:

            if TextDataset._is_operator(token):
                op_count += 1
                continue

            res = TextDataset._is_index(token)
            if not res:
                res = TextDataset._parse_pair(token)

            if res:
                count_index += 1

                if result["seq"] and result["seq"] != res[0]:
                    return None

                result["seq"] = res[0]
                result["seq_size"] = res[1]
            else:
                break

        if count_index > 1 and op_count > 1 and result["seq"] and result["seq_size"]:
            result = "sequence ( {} , {} )".format(result["seq"], result["seq_size"])

            return result
        else:
            return None

    @staticmethod
    def _parse_simple_operation(formula):

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

    @staticmethod
    def _simplify_formula(formula):

        formula = formula.strip()

        if len(formula) == 0:
            return formula

        res = TextDataset._parse_pair(formula)
        if res:
            return res.strip()

        if formula[0] == "(" and formula[-1] == ")":
            formula = formula[1:-1]

        res = TextDataset._parse_range(formula)
        if not res:
            res = TextDataset._parse_sequence(formula)
        if not res:
            res = TextDataset._parse_simple_operation(formula)

        if res:
            return res.strip()

        return formula.strip()

    def _parse_formulas(self, content, sample):
        if "$$$" not in content:
            return content

        if "formulas" not in sample:
            sample["formulas"] = {}
            sample["formulas_idx"] = {}

        content = "$$$ {} $$$".format(content)

        formula_pattern = "(?<=\$\$\$)(.*?)(?=\$\$\$)"
        formulas = re.findall(formula_pattern, content)
        formulas = list(map(TextDataset._simplify_formula, formulas))

        for idx, formula in enumerate(formulas[1::2]):
            actual_index = 1 + idx * 2

            f = formulas[actual_index].strip()
            if len(f) == 0:
                continue

            fs = f.split("@@@")

            placeholders = ""

            for f in fs:

                f = " ".join(f.split())

                if f not in self.formulas:
                    placeholder                         = "formula{}".format(len(self.formulas))
                    self.formulas[f]                    = placeholder
                    sample["formulas"][f]               = placeholder
                    sample["formulas_idx"][placeholder] = f
                    placeholders += placeholder + " "
                else:
                    placeholder                         = self.formulas[f]
                    sample["formulas"][f]               = placeholder
                    sample["formulas_idx"][placeholder] = f

                    placeholders += placeholder + " "

                self.formulas_count[f] += 1

            formulas[actual_index] = placeholders

        formulas = map(lambda x: x.strip(), formulas)
        content = " ".join(formulas).strip()
        return content

    @staticmethod
    def _split_content(content):
        delims = [".", "?", "!", ",", ";"]
        for delim in delims:
            content = content.replace(delim, " {} ".format(delim))
        return content

    stopwords = {"the", "of", "to", "and", "is", "that", "it", "with"}

    @staticmethod
    def filter_sentence(sentence):
        return [x.lower() for x in word_tokenize(sentence)
                if len(x) >= 1 and x.lower() not in TextDataset.stopwords]

    def prepare_problem(self, sample):

        pipeline = [
            self._parse_digits,
            self._parse_formulas
        ]

        for field in self.fields:
            for fun in pipeline:
                sample[field] = fun(content=sample[field], sample=sample)

            content = sample[field]

            if "formulas_idx" in sample:
                for formula_idx in sample["formulas_idx"]:
                    pattern = " {} ".format(sample["formulas_idx"][formula_idx])
                    content = content.replace(formula_idx, pattern)

            content      = TextDataset._split_content(content)
            preprocessed = TextDataset.filter_sentence(content)
            sample["{}_pre".format(field)] = preprocessed

        return sample

    def flatten_samples(self, dataset):

        for sample in dataset:
            if "solutions" in sample:
                del sample["solutions"]
        return dataset
