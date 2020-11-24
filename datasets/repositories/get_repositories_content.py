import subprocess
from time import sleep

from datasets.repositories.search_repositories import SLEEP_TIME_BETWEEN_API_CALLS_SEC
from util import load_dataset

command_template = "mkdir -p ./results/%s && " \
                   "cd ./results/%s && " \
                   "git clone https://github.com/%s/%s && " \
                   "cd - "

def execute_command(user_name, repo_name):
    subprocess.call(command_template % (user_name, user_name, user_name, repo_name), shell=True)


def main():
    # For further development only
    data = load_dataset("repositories.json")
    counter = 0
    total_data_len = len(data["repo_full_name"])
    for repo in data["repo_full_name"]:
        counter = counter + 1
        print(f"[ITERATION {counter}/{total_data_len}]")

        user_name = repo.split('/')[0]
        repo_name = repo.split('/')[1]
        execute_command(user_name, repo_name)
        sleep(SLEEP_TIME_BETWEEN_API_CALLS_SEC)


if __name__ == "__main__":
    main()
