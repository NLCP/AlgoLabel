from util import run_system_command
import xml.etree.ElementTree as ET


def remove_nonascii(code):
    """Remove non-ASCII characters"""
    return ''.join([i if ord(i) < 128 else ' ' for i in code])


def preprocess_code_samples(sample, args):
    if "solutions" not in sample:
        sample["solutions"] = []
        return sample

    solutions = sample["solutions"]
    if len(solutions) == 0:
        return sample

    params = args["preprocess"]["code"]

    new_solutions = []
    for sol_idx, solution in enumerate(solutions):

        if not solution["code"]:
            continue

        solution["index"] = "{}_{}_{}".format(sample["source"],
                                              sample["index"],
                                              sol_idx)

        path = "./data/code/sources/{}.cpp".format(solution["index"])
        with open(path, "w") as f:
            ascii_code = remove_nonascii(solution["code"])
            f.write(ascii_code)
        solution["code"] = remove_unused(path, ascii_code)

        if params["clang"]:
            run_system_command("clang-format -i {}".format(path), verbose=False)
            with open(path, "r") as f:
                solution["code"] = f.read()

        new_solutions.append(solution)

    sample["solutions"] = new_solutions
    return sample


def remove_unused(path, source=None):
    file_name = path.split("/")[-1]
    log_file  = "./logs/tmp/cppcheck_{}.xml".format(file_name[:-4])
    cmd = "cppcheck --enable=all --xml -q --output-file=\"{}\" \"{}\"".format(log_file, path)

    run_system_command(cmd, verbose=False, split=False, shell=True)

    if not source:
        with open(path, "r") as f:
            source = f.read()

    try:
        lines  = source.split("\n")
        tree   = ET.parse(log_file)
        root   = tree.getroot()
        errors = root.find("errors")
    except Exception:
        return

    if not errors:
        return

    remove_lines = set()

    for error in errors.findall("error"):

        if error.get('id') == "unusedFunction":
            msg          = error.get('msg')
            fun          = msg.split("'")[1]
            location     = int(error.find('location').get('line')) - 1
            count_ph     = 0
            seen_the_end = False
            index        = location

            for line in lines[location:]:

                remove_lines.add(index)
                index += 1
                for ch in line:
                    if ch == "{":
                        count_ph += 1
                    elif ch == "}":
                        count_ph -= 1
                        seen_the_end = True

                if count_ph == 0 and seen_the_end:
                    break

    lines = [line for idx, line in enumerate(lines) if idx not in remove_lines]
    return "\n".join(lines)
