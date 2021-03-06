# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/

from safe.asm_embedding.FunctionAnalyzerRadare import RadareFunctionAnalyzer
from argparse import ArgumentParser
from safe.asm_embedding.FunctionNormalizer import FunctionNormalizer
from safe.asm_embedding.InstructionsConverter import InstructionsConverter
from safe.neural_network.SAFEEmbedder import SAFEEmbedder
from safe.utils import utils
from pathlib import Path

from util import run_system_command, load_dataset, fcall
from pprint import pprint as pp
import logging
from os import path
from tqdm import tqdm


class SAFE:

    def __init__(self,
                 model_path="./data/safe_trained_X86.pb",
                 instr_conv="./data/i2v/word2id.json",
                 max_instr=150):

        self.converter = InstructionsConverter(instr_conv)
        self.normalizer = FunctionNormalizer(max_instruction=max_instr)
        self.embedder = SAFEEmbedder(model_path)
        self.embedder.loadmodel()
        self.embedder.get_tensor()

    def embedd_function(self, filename, address):
        analyzer = RadareFunctionAnalyzer(filename, use_symbol=False, depth=0)
        functions = analyzer.analyze()
        instructions_list = None
        for function in functions:
            if functions[function]['address'] == address:
                instructions_list = functions[function]['filtered_instructions']
                break
        if instructions_list is None:
            print("Function not found")
            return None
        converted_instructions = self.converter.convert_to_ids(instructions_list)
        instructions, length = self.normalizer.normalize_functions([converted_instructions])
        embedding = self.embedder.embedd(instructions, length)
        return embedding

    def embedd_functions(self, filename):
        analyzer = RadareFunctionAnalyzer(filename, use_symbol=False, depth=0)
        functions = analyzer.analyze()
        embeddings = []
        for function in functions:
            instructions_list = functions[function]['filtered_instructions']
            converted_instructions = self.converter.convert_to_ids(instructions_list)
            instructions, length = self.normalizer.normalize_functions([converted_instructions])
            embedding = self.embedder.embedd(instructions, length)
            embeddings.append(embedding)
        return embeddings


def scrub(code):
    code = code.replace("#include<conio.h>", "\n")
    code = code.replace("void main(", "int main(")
    return code


def extract_safe_embeddings(safe, source):

    source_path = Path.cwd() / "safe" / "tmp" / "source.cpp"
    binary_path = Path.cwd() / "safe" / "tmp" / "source.o"

    with open(source_path, "w") as f:
        code = scrub(source)
        f.write(code)

    result = run_system_command("x86_64-w64-mingw32-g++ {} -c -o {} -std=c++17 -O2"
                                .format(source_path, binary_path))

    if not path.exists(binary_path) or result:
        return None

    embeddings = [x for x in safe.embedd_functions(binary_path) if x]
    return embeddings


@fcall
def prepare_dataset(args, safe, dataset):

    not_safe = 0

    for sample in tqdm(dataset):
        emb = extract_safe_embeddings(safe, sample["raw"])
        if emb:
            sample["safe"] = emb
        else:
            not_safe += 1

    logging.info("Failed SAFE compilations {}".format(not_safe))

    return dataset


if __name__ == '__main__':

    utils.print_safe()

    parser = ArgumentParser(description="Safe Embedder")

    parser.add_argument("-m", "--model",   help="Safe trained model to generate function embeddings")
    parser.add_argument("-i", "--input",   help="Input executable that contains the function to embedd")
    parser.add_argument("-a", "--address", help="Hexadecimal address of the function to embedd")

    args = parser.parse_args()
    address = int(args.address, 16)
    safe = SAFE(args.model)

    embedding = safe.embedd_function(args.input, address)
    print(embedding[0])




