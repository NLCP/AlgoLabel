import json
from time import sleep

import requests

# Only the first 1000 search results are available, https://docs.github.com/v3/search/
from util import load_dataset, dump_dataset

GITHUB_MAX_RESULTS = 1000

# Default no. of results per page
PER_PAGE = 30

# Keyword to search for
KEYWORDS = "codeforces"

# In case of failure, retry MAX_RETRY times
MAX_RETRY = 10

# API rate limiter, https://developer.github.com/v3/#rate-limiting
# One API call each SLEEP_TIME_BETWEEN_API_CALLS_SEC seconds
SLEEP_TIME_BETWEEN_API_CALLS_SEC = 60

DATASET_PATH = 'dataset.json'
REPOSITORIES_OUTPUT_PATH = 'repositories.json'


def main():

    # For further development only
    data = load_dataset(DATASET_PATH)

    # Query templates
    query_url = "https://api.github.com/search/repositories?q=%s+language:cpp&per_page=30" % (KEYWORDS)
    query_url_page = "https://api.github.com/search/repositories?q=%s+language:cpp&per_page=30&page=" % (KEYWORDS)
    resp = requests.get(query_url).json()

    git_repos = {"repo_full_name": []}
    NO_REQUESTS = min(int(resp["total_count"] / PER_PAGE), int(GITHUB_MAX_RESULTS / PER_PAGE))
    for i in range(1, NO_REQUESTS + 1):
        # Iterate NO_REQUESTS time
        print("[ITERATION] ", i, "/", NO_REQUESTS)
        query_url_page_formatted = query_url_page + str(i)
        resp = requests.get(query_url_page_formatted).json()
        retry = MAX_RETRY
        while retry > 0 and not resp.get("items", None):
            # Retry in case of failure, MAX_RETRY times
            print("... [retry]", MAX_RETRY - retry + 1, "/", MAX_RETRY)
            resp = requests.get(query_url_page + str(i)).json()
            sleep(SLEEP_TIME_BETWEEN_API_CALLS_SEC)
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
