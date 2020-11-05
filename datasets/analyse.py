from util import load_dataset, dump_dataset, fcall, print_defaultdict
from collections import defaultdict


@fcall
def compute_len_stats(args):
    model = args["model"]
    model_params = args["models"][model]
    encoders = model_params["encoders"]
    inputs = encoders["inputs"]
    input_type = encoders["input_type"]

    stats = {}
    for input in inputs:
        field = input["field"]
        stats[field + "_len"] = []

    for ds in ["train", "dev", "test"]:

        if input_type == "code":
            ds = "{}_complete".format(ds)

        dataset = load_dataset("./data/datasets/split/{}/{}.json".format(input_type,
                                                                         ds))

        for sample in dataset:
            for input in inputs:
                field = input["field"]
                stats[field + "_len"].append(str(len(sample[field])))

    for field in stats:
        with open("./logs/stats_{}.csv".format(field), "w") as f:
            f.write(field + "\n")
            for x in stats[field]:
                f.write(x + "\n")


@fcall
def compute_code_statistics(args, dataset):
    vocab = set()
    num_tokens = 0

    without_solutions = 0
    avg_num_solutions = 0

    avg_num_tokens = 0
    avg_num_ast = 0
    avg_safe = 0

    max_source = 0
    max_ast = 0
    max_safe = 0

    problems = set()

    no_tags = 0
    with_tags = 0

    for sample in dataset:

        if "solutions" in sample:
            avg_num_solutions += len(sample["solutions"])

            if "tags" not in sample or len(sample["tags"]) == 0:
                no_tags += len(sample["solutions"])
            else:
                with_tags += len(sample["solutions"])

        else:
            index = sample["index"].split("_")[1]
            problems.add(index)

        if "tokens" in sample:
            avg_num_tokens += len(sample["tokens"])
            max_source = max(max_source, len(sample["tokens"]))

        if "starts" in sample:
            avg_num_ast += len(sample["starts"])

        if "safe" in sample:
            avg_safe += len(sample["safe"])

        if "tokens" in sample and len(sample["tokens"]) == max_source:
            max_ast = len(sample["starts"])
            max_safe = len(sample["safe"])

        # for token in sample["tokens"]:
        #     vocab.add(token)
        #
        # num_tokens += len(sample["tokens"])

    # print("\n")
    # print("Code Statistics:")
    # print("Vocabulary size", len(vocab))
    # print("Avg. num. tokens", float(num_tokens) / len(dataset))
    # print("\n")

    print("Code stats!")
    if avg_num_solutions:
        print("Problems without solutions:", without_solutions)
        print("Avg. num. of solutions:", avg_num_solutions / len(dataset), "\n\n")
        print("Total num. of solutions:", avg_num_solutions)
    else:
        print("Avg tokens", avg_num_tokens / len(dataset))
        print("Avg AST", avg_num_ast / len(dataset))
        print("Avg safe", avg_safe / len(dataset))
        # print("Max source", max_source)
        # print("Avg tokens", (avg_num_tokens - max_source) / len(dataset))
        # print("Avg AST", (avg_num_ast - max_ast) / len(dataset))
        # print("Avg safe", (avg_safe- max_safe) / len(dataset))
        print("Num. problems", len(problems))

    print("With labels", with_tags)
    print("Without labels", no_tags)


@fcall
def compute_text_statistics(args, dataset):
    vocab = set()

    field_stats = {
        "statement": defaultdict(int),
        "input": defaultdict(int),
        "output": defaultdict(int)
    }

    discard_sample = 0
    for sample in dataset:

        for field in ["statement", "input", "output"]:
            if field not in sample:
                discard_sample += 1
                break
        else:
            for field in ["statement", "input", "output"]:
                tokens = sample[field].split()
                field_stats[field]["num_tokens"] += len(tokens)
                for token in tokens:
                    vocab.add(token)

    print("\n")
    print("Text Statistics:")
    print("Vocabulary size", len(vocab))
    for field in field_stats:
        print("Avg. num. tokens", field, float(field_stats[field]["num_tokens"]) / (len(dataset) - discard_sample))
    print("\n")


@fcall
def compute_tag_statistics(args, dataset):
    print("Number of samples", len(dataset))

    tag_distro = defaultdict(int)
    avg_num_tags = 0
    samples_without_tags = 0

    for sample in dataset:

        if "tags" not in sample or len(sample["tags"]) == 0:
            samples_without_tags += 1
            continue

        for tag in sample["tags"]:
            tag_distro[tag] += 1
        avg_num_tags += len(sample["tags"])

    _ = print_defaultdict(tag_distro)

    print("#Samples without tags", samples_without_tags)
    print("Average number of tags", avg_num_tags / (len(dataset) - samples_without_tags))


@fcall
def compute_counter_statistics(args):
    full_dataset = []
    for source in args["sources"]:

        dataset = load_dataset("./data/datasets/{}.json".format(source))
        full_dataset += dataset

        compute_tag_statistics(args, dataset)

        # if "code" in args["sources"][source]["type"]:
        #     compute_code_statistics(args, dataset)

        if "text" in args["sources"][source]["type"]:
            compute_text_statistics(args, dataset)

    compute_tag_statistics(args, full_dataset)


def fix_y(args):
    train_all = load_dataset("./data/datasets/split/code/train_all.json")

    for sample in train_all:

        Y = []
        for tag in args["split"]["labels"]:
            if tag in sample["tags"]:
                Y.append(1)
            else:
                Y.append(0)
        sample["Y"] = Y

    dump_dataset("./data/datasets/split/code/train_all.json", train_all)


def investigate_curiosity(args):
    test = load_dataset("./data/datasets/split/text/test.json")

    tag_distro = defaultdict(int)

    for sample in test:

        for label in args["split"]["labels"]:
            if label in sample["tags"]:
                tag_distro[label] += 1

    print_defaultdict(tag_distro)


def analyse_license_situation(args):
    bad = 0
    not_so_bad = 0

    licenses = defaultdict(int)
    for sample in load_dataset("./data/datasets/kattis.json"):
        if "license" in sample:
            licenses[" ".join(sample["license"].strip().split())] += 1
            if "Restricted" in sample["license"]:
                for tag in sample["tags"]:
                    if tag in args["split"]["labels"]:
                        bad += 1
            elif "educational" in sample["license"]:
                for tag in sample["tags"]:
                    if tag in args["split"]["labels"]:
                        not_so_bad += 1

    print("Bad", bad)
    print("Not so bad", not_so_bad)
    print_defaultdict(licenses)


@fcall
def analyse_split(args):
    for input_type in ["code"]:  # ["code", "text"]:

        for split in ["train", "dev", "test"]:
            # if input_type == "code":
            #     split = split + "_all"

            dataset = load_dataset("./data/datasets/split/{}/{}.json".format(input_type, split))
            if input_type == "code":
                compute_code_statistics(args, dataset)
            else:
                compute_text_statistics(args, dataset)

            compute_tag_statistics(args, dataset)

        break


@fcall
def diff_stats(args):
    cf = 0

    for split in ["dev", "test", "train"]:

        dataset = load_dataset("./data/datasets/split/{}/{}.json".format("text", split))

        count = defaultdict(int)
        for sample in dataset:
            count[sample["difficulty_class"]] += 1
        print_defaultdict(count)

    #     cf += len(dataset)
    #
    # for sample in load_dataset("./data/datasets/split/text/train.json"):
    #     if sample["source"] == "codeforces":
    #         cf += 1

    print("Total CF", cf)


def analyse_dataset(args):
    diff_stats(args)
    analyse_split(args)
    # investigate_curiosity(args)
    # compute_counter_statistics(args)
    # analyse_license_situation(args)
    # fix_y(args)
    # compute_len_stats(args)
