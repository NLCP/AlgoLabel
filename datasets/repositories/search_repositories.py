from time import sleep

import requests

# Only the first 1000 search results are available, https://docs.github.com/v3/search/
from util import dump_dataset

GITHUB_MAX_RESULTS = 1000

# Default no. of results per page
PER_PAGE = 30

# Keyword to search for
KEYWORDS = "codeforces"

# In case of failure, retry MAX_RETRY times
MAX_RETRY = 3

# API rate limiter, https://developer.github.com/v3/#rate-limiting
# One API call each SLEEP_TIME_BETWEEN_API_CALLS_SEC seconds
SLEEP_TIME_BETWEEN_API_CALLS_SEC = 20

DATASET_PATH = 'dataset.json'
REPOSITORIES_OUTPUT_PATH = 'repositories.json'

QUERY_URL_TEMPLATES = [
    "https://api.github.com/search/repositories?q=codeforces+created:<2017-01-01+language:cpp&per_page=30",
    "https://api.github.com/search/repositories?q=codeforces+created:2017-01-01..2018-09-01+language:cpp&per_page=30",
    "https://api.github.com/search/repositories?q=codeforces+created:2018-09-01..2020-01-01+language:cpp&per_page=30",
    "https://api.github.com/search/repositories?q=codeforces+created:2020-01-01..2020-06-01+language:cpp&per_page=30",
    "https://api.github.com/search/repositories?q=codeforces+created:2020-06-01..2020-10-01+language:cpp&per_page=30",
    "https://api.github.com/search/repositories?q=codeforces+created:>2020-10-01+language:cpp&per_page=30"
    ]

# For running this script, you should generate you own token
# https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token

# e.g. used token in testing:
# TOKEN = 'a9bb8139a8f766e3b66e45586fffc933d4b36c3f'
# Note: this is revoked
TOKEN = None


def main():
    git_repos = {"repo_full_name": []}

    for query_url in QUERY_URL_TEMPLATES:
        # Query templates
        query_url_page = query_url + "&page="
        resp = requests.get(query_url, auth=('raresraf', TOKEN)).json()

        NO_REQUESTS = min(int(resp["total_count"] / PER_PAGE), int(GITHUB_MAX_RESULTS / PER_PAGE))
        for i in range(1, NO_REQUESTS + 1 + 1):
            # Iterate NO_REQUESTS time
            print("[ITERATION] ", i, "/", NO_REQUESTS + 1)
            query_url_page_formatted = query_url_page + str(i)
            resp = requests.get(query_url_page_formatted,
                                auth=('raresraf', TOKEN)).json()
            retry = MAX_RETRY
            while retry > 0 and not resp.get("items", None):
                # Retry in case of failure, MAX_RETRY times
                print("... [retry]", MAX_RETRY - retry + 1, "/", MAX_RETRY)
                resp = requests.get(query_url_page + str(i),
                                    auth=('raresraf', TOKEN)).json()
                sleep(10 * SLEEP_TIME_BETWEEN_API_CALLS_SEC)
                retry = retry - 1
            if retry == 0:
                continue
            for r in resp["items"]:
                git_repos["repo_full_name"].append(r["full_name"])

            sleep(SLEEP_TIME_BETWEEN_API_CALLS_SEC)

    # Write output to REPOSITORIES_OUTPUT_PATH
    dump_dataset(REPOSITORIES_OUTPUT_PATH, git_repos)


if __name__ == "__main__":
    main()
