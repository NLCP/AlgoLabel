import os
from time import sleep

from datasets.repositories.get_repositories_content import execute_command
from datasets.repositories.search_repositories import SLEEP_TIME_BETWEEN_API_CALLS_SEC
from util import load_dataset


def main():
    data = load_dataset("repositories.json")
    counter = 0
    corrupted = 0
    total_data_len = len(data["repo_full_name"])
    for repo in data["repo_full_name"]:
        counter = counter + 1
        print(f"[ITERATION {counter}/{total_data_len}]")
        print(f"[CORRUPTED {corrupted}/{total_data_len}]")

        user_name = repo.split('/')[0]
        repo_name = repo.split('/')[1]

        if os.path.isdir(f'./results/{user_name}/{repo_name}'):
            continue
        corrupted = corrupted + 1
        execute_command(user_name, repo_name)
        sleep(2 * SLEEP_TIME_BETWEEN_API_CALLS_SEC)


if __name__ == "__main__":
    main()
