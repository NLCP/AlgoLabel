
import logging
from datasets.dataset import Dataset
from util import run_system_command, fcall, ensure_path
from util import load_dataset, parse_line_csv, parse_csv
from util import remove_folder, multi_process
from pprint import pprint as pp
import xml.etree.ElementTree as ET
import os
from sys import platform
from pathlib import Path
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from sys import platform


class SourceDataset(Dataset):

    def __init__(self, args, data):
        super().__init__(args, data)
        self.params = self.args["prepare"]["source"]

        self.tmp_path = Path.cwd() / "data" / "tmp"
        ensure_path(self.tmp_path)

        self.original    = self.tmp_path / "original.cpp"
        self.source_path = self.tmp_path / "solution.cpp"
        self.log_path    = self.tmp_path / "cppcheck_log.xml"
        self.tokens_path = self.tmp_path / "solution.tokens"
        self.dest_path   = Path.cwd() / "data" / "datasets" / "source"

        self.compiler = self.params["compiler"]
        if self.compiler != "g++":
            raise NotImplementedError("Unsupported compiler {}".format(self.compiler))

        self.tokenizer = self._ensure_tokenizer_exists()
        self.parallel  = args["prepare"]["parallel"]

    def _ensure_tokenizer_exists(self):

        tokenizer_dir  = Path.cwd() / "tokenizer" / "src"

        if platform == "win32":
            tokenizer_exe = "tokenizer.exe"
        else:
            tokenizer_exe = "tokenizer"

        tokenizer_path = tokenizer_dir / tokenizer_exe

        if not os.path.exists(tokenizer_path):
            current_path = Path.cwd()
            os.chdir(tokenizer_dir)
            run_system_command("{} *.cpp *.h -o {}".format(self.compiler, tokenizer_path))
            os.chdir(current_path)

        return tokenizer_path

    @staticmethod
    def _filter_irrelevant_task(sample):
        if "solutions" not in sample:
            return False
        if len(sample["solutions"]) == 0:
            return False
        if "contest_name" in sample and "April Fools" in sample["contest_name"]:
            return False
        return True

    def filter_irrelevant_tasks(self):

        return [sample for sample in self.data
                if self._filter_irrelevant_task(sample)]

    def prepare_problem(self, sample):

        if self.parallel:
            current_process    = mp.current_process().name
            self.original_path = self.tmp_path / "original_{}.cpp".format(current_process)
            self.source_path   = self.tmp_path / "solution_{}.cpp".format(current_process)
            self.log_path      = self.tmp_path / "cppcheck_log_{}.xml".format(current_process)
            self.tokens_path   = self.tmp_path / "solution_{}.tokens".format(current_process)

        solutions = []
        for idx, solution in enumerate(sample["solutions"]):
            try:
                solution = self.prepare_solution(solution)
                if not solution:
                    continue
                solutions.append(solution)
            except Exception as e:
                logging.error("Exception '{}' occurred while parsing solution {}".format(repr(e), str(solution["index"])))
            sample["solutions"] = solutions

        if len(solutions) == 0:
            return None
        return sample

    def _prepare_parallel(self):

        self.parallel = True

        result    = multi_process(self.prepare_problem, self.data[:2],
                                  cpus=2)
        self.data = [x for x in result if x]

        self.parallel = False
        return self

    def prepare(self):

        # Remove problems with no solutions
        self.data = self.filter_irrelevant_tasks()

        logging.debug("Preparing sources for {} problems..".format(len(self.data)))

        count_solutions = 0
        skipped         = 0
        result          = []

        for sample in tqdm(self.data):

            sample = self.prepare_problem(sample)
            if not sample:
                skipped += 1
            elif len(sample["solutions"]) > 0:
                result.append(sample)
                count_solutions += len(sample["solutions"])

        self.data = result

        logging.info("{} problems with invalid solutions skipped..".format(skipped))
        logging.info("Prepared {} problems with {} solutions".format(len(self.data), count_solutions))
        return result

    def prepare_solution(self, solution):

        code = solution["code"]
        if not code or len(code.split("\n")) > self.params["min_num_lines"]:
            return None

        raw      = code
        pipeline = [
            self.remove_non_ascii,
            self.remove_apostrophe,
            self.remove_external_includes,
            self.run_preprocessor,
            self.remove_unused_fcalls
        ]

        for fun in pipeline:
            code = fun(code)
            if not code:
                return None

        tokens = self.split_source_tokens(code)

        solution["raw"]    = raw
        solution["code"]   = code
        solution["tokens"] = tokens

        return solution

    @staticmethod
    def dump_source(path: str, code: str):
        """ Write solution to file """
        with open(path, "w") as f:
            f.write(code)

    @staticmethod
    def load_source(path: str):
        """
        Parse source code from given path.
        :param path:
        :return:
        """
        with open(path, "r") as f:
            lines = f.readlines()
            code  = []
            for idx, line in enumerate(lines):
                if len(line.strip()) > 0:
                    code.append(line.rstrip())
            code = "\n".join(code)
        return code

    @staticmethod
    def remove_non_ascii(code):
        """Remove non-ASCII characters"""
        if not code:
            return
        return ''.join([i if ord(i) < 128 else ' ' for i in code])

    def remove_apostrophe(self, code):
        """Normalize surface form of floats for new C++ code"""
        res = ""
        for idx, ch in enumerate(code):
            if ch == '\'':
                if code[idx - 1].isdigit() and code[idx + 1].isdigit():
                    continue
            res += ch
        return res

    def remove_external_includes(self, code):

        """
        Remove external header includes.
        :param code:
        :return:
        """

        exclude = [
            "#include",
            "using namespace",
            "#import",
            "#pragma"
        ]

        lines = []
        for line in code.split("\n"):
            skip = False
            for restricted in exclude:
                if restricted in line:
                    skip = True
                    break
            if skip:
                continue
            lines.append(line)
        return "\n".join(lines)

    def run_preprocessor(self, code):

        original_code_path = self.tmp_path / "original.cpp"
        self.dump_source(original_code_path, code)

        cmd = "{} -E -P {} -o {} -DONLINE_JUDGE".format(
            self.compiler,
            original_code_path,
            self.source_path
        )

        rc = run_system_command(cmd, verbose=False)
        if rc:
            raise Exception("Compilation error")

        code = self.load_source(self.source_path)
        return code

    def remove_unused_fcalls(self, code):
        '''
            Detect and remove unused function calls using the cppcheck tool.
            Dependency: cppcheck
            :return: code
        '''

        if not self.params["remove_unused_fcalls"]:
            return code

        self.dump_source(self.source_path, code)

        # Run cppcheck
        cmd  = "cppcheck --enable=all --xml -q --output-file=\"{}\" {}"\
            .format(self.log_path, self.source_path)
        run_system_command(cmd, verbose=False, split=False, shell=False)

        try:
            lines        = code.split("\n")
            tree         = ET.parse(self.log_path)
            root         = tree.getroot()
            errors       = root.find("errors")
            remove_lines = set()

            if not errors:
                return code

            for error in errors.findall("error"):

                if error.get('id') == "unusedFunction":
                    msg = error.get('msg')
                    fun = msg.split("'")[1]
                    location = int(error.find('location').get('line')) - 1
                    count_ph = 0
                    seen_the_end = False
                    index = location

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

            lines = [line for idx, line in enumerate(lines)
                     if idx not in remove_lines and len(line) > 0]
            return "\n".join(lines)

        except Exception as e:
            logging.critical(e)
            return code

    def remove_unused_code(self, code):

        excluded = ["#include", "#pragma", "using namespace", "#import"]

        res = []
        in_comment_block = False

        for line in code.split("\n"):

            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("//"):
                if line.endswith("\\\\"):
                    in_comment_block = True
                continue
            elif in_comment_block and line.endswith("\\"):
                in_comment_block = True
                continue
            elif in_comment_block:
                in_comment_block = False
                continue

            if self.params["remove_imports"]:
                for s in excluded:
                    if line.startswith(s) and "*/" not in line:
                        break
                else:
                    res.append(line)
            else:
                res.append(line)

        return "\n".join(res)

    def split_source_tokens(self, code):

        """
        Obtain list of tokens from available source code.
        :param code:
        :return:
        """

        self.dump_source(self.source_path, code)

        tokenizer_cmd = "{} {}".format(self.tokenizer, self.source_path)
        with open(self.tokens_path, "w") as f:
            rc = run_system_command(tokenizer_cmd,
                                    stdout=f,
                                    stderr=f,
                                    verbose=False)
            if rc:
                raise Exception("Failure occured during tokenization!")

        with open(self.tokens_path, "r") as f:
            tokens = []
            for line in f:
                if "//" not in line and "/*" not in line:
                    tokens.append(line.strip())
                elif "EOF encountered" in line:
                    raise Exception("Failure occured during tokenizer!")

        return tokens

    def flatten_samples(self, dataset):
        """
        The original dataset is a nested list of problems, each problem having a list of solutions.
        This functions returns a simple list, containing only solutions.
        :param dataset:
        :return:
        """

        result = []
        for sample in dataset:
            if "solutions" in sample:
                for solution in sample["solutions"]:
                    if "Y" in sample:
                        solution["Y"] = sample["Y"]
                    if "tags" in sample:
                        solution["tags"] = sample["tags"]
                    result.append(solution)
        return result

    # @fcall
    # def split_data(self, verbose=True):
    #
    #     '''
    #     Split dataset in separate training/validation/test datasets.
    #     Solutions belonging to the same problem are split together.
    #     :return:
    #     '''
    #
    #     params  = self.args["split"]
    #     labels  = params["labels"]
    #
    #     np.random.shuffle(self.data)
    #
    #     labeled, unlabeled = self.separate_unlabeled_samples(labels)
    #
    #     if params["difficulty_based"]:
    #         distribution = self.split_on_difficulty(labeled)
    #         dataset = distribution["Easy"] + distribution["Medium"] + distribution["Hard"]
    #         train, dev, test = self.split_stratified(dataset)
    #         train += distribution["Various"]
    #     else:
    #         train, dev, test = self.split_stratified(labeled)
    #
    #     data_split = {
    #         "train": self.flatten_solutions(train),
    #         "dev": self.flatten_solutions(dev),
    #         "test": self.flatten_solutions(test),
    #         "unlabeled": self.flatten_solutions(unlabeled)
    #     }
    #
    #     if verbose:
    #         for split in data_split:
    #             if split != "unlabeled":
    #                 logging.info("Stats for the {} data split:".format(split))
    #                 self.compute_tag_distribution(data_split[split])
    #
    #     return data_split

    def _extract_ast(self, sources_path, result_path, force_rewrite=False):

        params        = self.args["features"]["types"]["code2vec"]
        count_samples = 0

        for sample in self.data:

            sample_path = sources_path / "{}.cpp".format(sample["index"])

            if not force_rewrite and os.path.exists(sample_path):
                continue

            with open(sample_path, "w") as f:
                f.write(sample["code"])

            count_samples += 1

        logging.info("Written {} source files to disk!".format(count_samples))

        # Invoke astminer for cpp
        astminer_cmd = "java -Xmx52G -jar {} pathContexts".format(self.args["astminer"])
        astminer_params = "--lang cpp --maxContexts {} --maxH {} --maxW {}".format(
            params["max_contexts"],
            params["max_height"],
            params["max_width"]
        )

        rc = run_system_command("{} {} --project {} --output {}".format(
            astminer_cmd,
            astminer_params,
            str(sources_path),
            str(result_path)
        ), verbose=False)

        rc = "Success" if rc == 0 else "Failure"
        logging.info("astminer is done - {}!".format(rc))

    def _preprocess_path_contexts(self, result_path):

        result = {}

        for file in ["node_types", "paths", "tokens"]:

            file_path = result_path / "{}.csv".format(file)
            if not os.path.exists(file_path):
                raise Exception("File {} was not created by astminer.".format(file_path))

            result[file] = parse_csv(str(file_path), delimiter=",")

        path_contexts = parse_line_csv(result_path / "path_contexts.csv", delimiter=" ")
        file_paths    = {}

        for index, paths in path_contexts:

            all_paths = []
            for path in paths:
                path = path.split(",")
                if len(path) != 3:
                    logging.critical("No valid paths for {} !".format(index))
                    break
                all_paths.append({
                    "start": path[0],
                    "path_index": path[1],
                    "end": path[2]
                })
            file_paths[index] = all_paths

        result["file_paths"] = file_paths

        def _create_index_map(data, field_name):

            field2index = "{}2index".format(field_name)
            index2field = "index2{}".format(field_name)

            result[field2index] = {}
            result[index2field] = {}

            for match in data:
                idx   = match["id"]
                field = match[field_name]
                result[field2index][field] = idx
                result[index2field][idx] = field

        _create_index_map(result["node_types"], "node_type")
        _create_index_map(result["tokens"], "token")
        _create_index_map(result["paths"], "path")

        return result

    def preprocess_ast(self, force_rewrite=False, path=None):

        '''
        :param force_rewrite: for each sample, if a file with
        the corresponding name already exists on disk, don't overwrite it
        :param path: target folder
        :return:
        '''

        if not path:
            path = Path.cwd() / "data" / "code" / "tmp"

        if force_rewrite:
            remove_folder(path / "sources")

        sources_path = path / "sources"
        ensure_path(sources_path)

        result_path = path / "contexts"
        ensure_path(result_path)

        # TODO: check weird race condition
        # print(os.path.exists(path / "sources"))
        # print(os.path.exists(path / "contexts"))

        self._extract_ast(sources_path=sources_path,
                          result_path=result_path,
                          force_rewrite=force_rewrite)

        result_path /= "cpp"
        return self._preprocess_path_contexts(result_path)


def prepare(sample, args):
    handler = SourceDataset(args, [])
    return handler.prepare_problem(sample)


def match_ast_data(fold_data, train_data):

    meta = {
        "fold2train_token": {},
        "fold2train_type": {},
        "fold2train_path": {}
    }

    for token in fold_data["token2index"]:
        if token in train_data["token2index"]:
            fold_index  = fold_data["token2index"][token]
            train_index = train_data["token2index"][token]
            meta["fold2train_token"][fold_index] = train_index

    for node_type in fold_data["node_type2index"]:
        if node_type in train_data["node_type2index"]:
            fold_index  = fold_data["node_type2index"][node_type]
            train_index = train_data["node_type2index"][node_type]
            meta["fold2train_type"][fold_index] = train_index

    paths_found = 0

    for path_info in fold_data["paths"]:

        index      = path_info["id"]
        fold_path  = path_info["path"]
        train_path = []

        for node_type in fold_path.split():
            if node_type not in meta["fold2train_type"]:
                logging.debug("NodeType {} not found!".format(node_type))
                break
            train_path.append(meta["fold2train_type"][node_type])
        else:
            train_path = " ".join(train_path)
            if train_path in train_data["path2index"]:
                train_path_index = train_data["path2index"][train_path]
                meta["fold2train_path"][index] = train_path_index
                paths_found += 1
                continue

    logging.info("Total paths found {} / {}".format(paths_found, len(fold_data["paths"])))

    result = {}

    for file_index in fold_data["file_paths"]:

        all_paths    = fold_data["file_paths"][file_index]
        sample_index = file_index.split("\\")[-1][:-4]
        result[sample_index] = []

        logging.debug("Investigating {}".format(sample_index))

        paths_found = 0
        token_found = 0

        for path_data in all_paths:

            path_index = path_data["path_index"]

            if path_index in meta["fold2train_path"]:
                train_path_index = meta["fold2train_path"][path_index]
                paths_found += 1
            else:
                train_path_index = -1

            start_index = path_data["start"]
            if start_index in meta["fold2train_token"]:
                train_start_index = meta["fold2train_token"][start_index]
                token_found += 1
            else:
                train_start_index = -1

            end_index = path_data["end"]
            if end_index in meta["fold2train_token"]:
                train_end_index = meta["fold2train_token"][end_index]
                token_found += 1
            else:
                train_end_index = -1

            result[sample_index].append({
                "start": int(train_start_index),
                "path_index": int(train_path_index),
                "end": int(train_end_index),
            })

        logging.info("Paths found {} / {}".format(paths_found, len(all_paths)))
        logging.info("Tokens found {} / {}".format(token_found, 2 * len(all_paths)))

    return result


def save_code2vec_index(dataset, path_index, fold_label):

    result       = []
    num_failures = 0

    if fold_label == "train":
        meta = {}
        for file_index in path_index["file_paths"]:
            sample_index       = file_index.split("\\")[-1][:-4]
            meta[sample_index] = path_index["file_paths"][file_index]
        path_index = meta

    for sample in dataset:

        if sample["index"] not in path_index or len(path_index[sample["index"]]) == 0:
            logging.debug("Failure to compute code2vec paths for {}".format(sample["index"]))
            num_failures += 1
            continue

        sample["code2vec"] = path_index[sample["index"]]
        result.append(sample)

    logging.info("{} failures for {} fold!".format(num_failures, fold_label))
    return result
