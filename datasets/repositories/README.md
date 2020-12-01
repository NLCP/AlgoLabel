# Generic crawler for solutions code on GitHub

## Usage

- All commands are presented relative to the root path of the repository.

### Prerequisited
1. Edit `QUERY_URL_TEMPLATES` in `search_repositories.py`
2. Add [TOKEN](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/creating-a-personal-access-token) in `search_repositories.py`
3. Add to python path root dir: `export PYTHONPATH="${PYTHONPATH}:."`

### Run script for searching repositories:
This will generate `repositories.json` file containing 
1. `python3 datasets/repositories/search_repositories.py`
2. (optional) Add a manual entry to the generated json file: `"date" : "DD-MMM-YYYY"`

### Remove possible duplicates
1. `python3 datasets/repositories/remove_duplicates.py`

### Get repositories content
1. `python3 datasets/repositories/get_repositories_content.py`

### Run sanity checks and get repositories that failed in the previous step
1. `python3 datasets/repositories/sanity_check_repositories_content.py`
2. You can re-run command from 1. as many times as you want. Note that some repositories from that list may have been deleted.


### TODO:  python3 datasets/repositories/repos_to_sources.py