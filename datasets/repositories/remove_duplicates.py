from datasets.repositories.search_repositories import REPOSITORIES_OUTPUT_PATH
from util import dump_dataset, load_dataset


def main():
    # Read input from to REPOSITORIES_OUTPUT_PATH
    data = load_dataset(REPOSITORIES_OUTPUT_PATH)

    # Remove duplicates from data["repo_full_name"]
    # This preserves list order.
    seen = set()
    result = []
    for item in data["repo_full_name"]:
        if item not in seen:
            seen.add(item)
            result.append(item)

    # Write output to REPOSITORIES_OUTPUT_PATH
    dump_dataset(REPOSITORIES_OUTPUT_PATH, data)


if __name__ == "__main__":
    main()
