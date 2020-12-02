import errno
import json
import os
import re
from shutil import copyfile

from datasets.codeparser import remove_nonascii, remove_unused
from util import run_system_command

code_dict = {}


def save_solution_to_problem(identifier, path, counter):
    try:
        number = int(str(list(map(int, re.findall(r'\d+', identifier)))[0]))
        if number > 10000 or number < 100:
            if "codeforces" not in path.lower() and "cf" not in path.lower() and "codeforce" not in path.lower():
                print(f"Not a solution!", identifier, path)
                return
    except Exception:
        pass
    print(f"Solution found: [{counter + 1}]")
    save_solution_to_problem_exec(identifier, path)


def save_solution_to_problem_exec(identifier, path):
    identifier = identifier.upper()
    print(identifier, path)

    with open(path, 'r', errors='ignore') as f:
        data = f.read()
    # Cleanup code
    with open(path, "w") as f:
        ascii_code = remove_nonascii(data)
        f.write(ascii_code)
    data = remove_unused(path, ascii_code)
    run_system_command("clang-format -i \"./{}\"".format(path), verbose=False, shell=True, split=False)
    with open(path, "r") as f:
        data = f.read()

    if not code_dict.get(identifier, None):
        code_dict[identifier] = []
    code_dict[identifier].append({"solution": data, "owner": (path.split('/')[2]), "repository": (path.split('/')[3])})

    new_path = "results_code/" + "/".join(path.split('/')[1:-1]) + "/" + identifier

    # Check if file already exists
    file_name = identifier
    if os.path.isfile(new_path):
        expand = 1
        while True:
            expand += 1
            new_file_name = identifier.split(".CPP")[0] + "_" + str(expand) + ".CPP"
            new_path = "results_code/" + "/".join(path.split('/')[1:-1]) + "/" + new_file_name
            if os.path.isfile(new_path):
                continue
            else:
                file_name = new_file_name
                break

    new_path = "results_code/" + "/".join(path.split('/')[1:-1]) + "/" + file_name

    # Save a copy of the file
    if not os.path.exists(os.path.dirname(new_path)):
        try:
            os.makedirs(os.path.dirname(new_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    copyfile(path, new_path)


counter = 0
for root, dirs, files in os.walk("results/"):
    for name in files:
        try:
            # .../336A.cpp
            matchObj = re.match("[^0-9]*([0-9]+)[^ABCDE]*([ABCDE]).*(\.cpp)$", name, re.I | re.I | re.I)
            if matchObj:
                pretender = matchObj.group(1) + matchObj.group(2) + matchObj.group(3)
                save_solution_to_problem(pretender, os.path.join(root, name), counter)
                counter = counter + 1
                continue

            # .../336/A.cpp
            matchObj_withDir =  re.match("^([ABCDE])[^ABCDE]*(\.cpp)$", name, re.I | re.I)
            if matchObj_withDir:
                parent_dir = os.path.basename(os.path.normpath(root))
                matchObj_dir = re.match("^([0-9]+).*$", parent_dir, re.I)
                if matchObj_dir:
                    pretender = matchObj_dir.group(1) + matchObj_withDir.group(1) + matchObj_withDir.group(2)
                    save_solution_to_problem(pretender, os.path.join(root, name), counter)
                    counter = counter + 1
                    continue


            # .../A/336.cpp
            matchObj_withDir = re.match("^([0-9]+).*(\.cpp)$", parent_dir, re.I | re.I)
            if matchObj_withDir:
                parent_dir = os.path.basename(os.path.normpath(root))
                matchObj_dir = re.match("^([ABCDE])[^ABCDE]*$", name, re.I)
                if matchObj_dir:
                    pretender = matchObj_withDir.group(1) + matchObj_dir.group(1) + matchObj_dir.group(2)
                    save_solution_to_problem(pretender, os.path.join(root, name), counter)
                    counter = counter + 1
                    continue

            # .../336A/blah.cpp
            matchObj = re.match("[^0-9]*([0-9]+)[^ABCDE]*([ABCDE]).*(\.cpp)$", name, re.I | re.I | re.I)
            if matchObj:
                pretender = matchObj.group(1) + matchObj.group(2) + matchObj.group(3)
                save_solution_to_problem(pretender, os.path.join(root, name), counter)
                counter = counter + 1
                continue
        except Exception:
            continue

with open('codeforces.json', 'w') as f:
    json.dump(code_dict, f)
