from datasets.simpleparser import digit_parser, formula_parser, split_content
from pprint import pprint as pp


def debug():
    from collections import defaultdict

    sample = {
        "url": "https://open.kattis.com/problems/flowfinder",
        "index": "Flow Finder",
        "source": "Kattis",
        "title": "Kattis - Flow Finder",
        "tags": [],
        "stats": {
            "total_submissions": "924",
            "total_submissions_ac": "132",
            "total_submissions_ratio": "14%",
            "total_users": "168",
            "total_users_ac": "106",
            "total_users_ratio": "63%"
        },
        "statement": "Last summer, Carla the Cartographer went on an expedition on    behalf of the National Center for Positioning and Charting (the    NCPC). The goal was to measure water flows of a river system    far away in the north. However, the area is quite remote and    Carla is not the adventurous type, so she was only able to    measure the water flows in some of the locations. Carla is now    worried that the NCPC will send her back to this wilderness    next summer, so she consulted some algorithm experts (you) to    see if it is possible to reconstruct the missing data.    The river system is represented by a rooted tree with    $$$ n $$$ vertices numbered from    $$$ 1 $$$ to $$$ n $$$. The leaves of this tree are the    sources, and the other vertices correspond to confluences    (places where multiple rivers join together). Water flows from    higher-numbered vertices to lower-numbered vertices. Vertex    $$$ 1 $$$, the root of the tree,    is the mouth of the river system, where it flows into the    ocean. The water flow of a source can be any positive integer,    while the water flow of a confluence is the sum of the water    flows of its children. You will be given this tree along with    the water flows at some of its vertices, and your task is to    find the water flows at all the vertices or determine that this    is impossible.",
        "input": "The first line of input contains an integer $$$ n $$$ ($$$ 2    \\leq n \\leq 3 \\cdot 10^5 $$$), the number of vertices in    the tree. Then follows a line containing $$$ n-1 $$$ integers $$$ p_2, \\ldots , p_{n} $$$ ($$$ 1 \\le p_ i &lt; i $$$), where    $$$ p_ i $$$ is the number of    the parent of vertex $$$ i $$$.    Finally there is a line containing $$$ n $$$ integers $$$ a_1, \\ldots , a_ n $$$ ($$$ 0 \\le a_ i \\le 10^9 $$$), where    $$$ a_ i $$$ represents the    water flow at vertex $$$ i $$$.    If $$$ a_ i $$$ is equal to    $$$ 0 $$$, then the water flow    at that vertex is unknown. Otherwise, $$$ a_ i $$$ is equal to the water flow at    vertex $$$ i $$$. Note that the    upper bound $$$ 10^9 $$$ on    $$$ a_ i $$$ does not apply to    the unknown values, they can be any positive integer.",
        "output": "If all the $$$ n $$$ water    flows can be reconstructed uniquely, then output them in    increasing order of vertex number. Otherwise, output    \u201cimpossible\u201d. Note that it is    possible that there is no way of reconstructing the water flows    (if the data provided by Carla is inconsistent somehow). In    that case you should also output \u201cimpossible\u201d.",
        "variables": [
            "n-1",
            "n",
            "i"
        ],
        "time_limit": "4 seconds",
        "memory_limit": "1024 MB",
        "contest": "Nordic Collegiate Programming Contest (NCPC) 2019",
        "license": ""
    }

    args = {
        "formulas": {},
        "formulas_idx": {},
        "formulas_count": defaultdict(int)
    }

    content = formula_parser(args, sample=sample, content=sample["input"])

    # for formula in args["formulas"]:
    #     if formula in content:
    #         content.replace(formula, args["formulas"][formula])

    pp(args)
    pp(content)


def preprocess_text_sample(sample, args):
    fields = args["preprocess"]["text"]["fields"]
    fields = [field for field in fields if fields[field] and field in sample]

    if len(fields) == 0:
        return sample

    sample["sentences"] = {}
    for field in fields:
        content = sample[field]

        if args["preprocess"]["text"]["digit_parser"]:
            content = digit_parser(args, sample=sample, content=content)

        if args["preprocess"]["text"]["formula_parser"]:
            content = formula_parser(args, sample=sample, content=content)

        sample[field] = content

        sample["sentences"][field] = split_content(args, content)

        if "formulas" in sample:
            for idx, sentence in enumerate(sample["sentences"][field]):
                for formula in sample["formulas_idx"]:
                    key = sample["formulas_idx"][formula]
                    sample["sentences"][field][idx] = \
                        sample["sentences"][field][idx].replace(formula, " {}  ".format(key))

    return sample


if __name__ == "__main__":
    debug()
