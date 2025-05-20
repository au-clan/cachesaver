# Cachesaver

Cachesaver is a high-efficiency caching library for experiments with large language models (LLMs), designed to minimize costs, improve reproducibility, and streamline debugging. The library enables caching of multiple responses per query, tracks usage for unique sampling across runs, and ensures statistical integrity. Built on Python’s asyncio, Cachesaver supports asynchronous execution, request batching, prompt deduplication, and race-condition prevention.

## Outer links

## Contributing
Multiple people are working on this project so we establish some rules.
- If you're working on something or you're planning to work on something, please indicate it in the [implementation schedule so that there's no two people working on the same thing.
- Everyone works in their own branch.
- When something is ready to be included in the main branch open a pull request.
- Pull requests will be reviewed Friday and Tuesday afternoon so please open your PRs before 2pm CET.
- Before opening a pull request make sure that your branch has been updated with the latest version of the main branch. You can do that using git merge or git rebase.

## Installation
**Disclaimer:** This project is currently in development and so some of these instructions might be outdated. If you face an installation problem please create a GitHub Issue even if you manage to find the solution. If the solution is found please include that as well :blush:

```bash
conda create -n cachesaver python=3.10
pip install -r requirements.txt
```

You will also need to install `cachesaver-core`. We plan to share cachesaver-core in pypi but for the moment you need to get it from the source. The following can be done at any directory (it doesn't/shouldn't have to be the current one).

```bash
cd cachesaver-core
conda activate cachesaver
pip install -e ".[test]"
pytest test/ -v # Run tests to verify everything works
```

You'll also need to firstly install any LLM APIs that you'd like to use (eg Groq, OpenAI, Together, etc.), and secondly, save their corresponding keys.

## Misc
### Local folder
You can create and name a folder named `./local`. Anything you save there will not be pushed. If possible, please use this instead of adding specific files to `.gitignore`.
