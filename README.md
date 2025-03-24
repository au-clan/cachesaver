# Cachesaver

Cachesaver is a high-efficiency caching library for experiments with large language models (LLMs), designed to minimize costs, improve reproducibility, and streamline debugging. The library enables caching of multiple responses per query, tracks usage for unique sampling across runs, and ensures statistical integrity. Built on Python’s asyncio, Cachesaver supports asynchronous execution, request batching, prompt deduplication, and race-condition prevention.

This is a repository utilizing Cachesaver to implement multiple frameworks and tasks. For the source code of the core Cachesaver package you can look into [cachesaver-core](https://github.com/au-clan/cachesaver-core/tree/nearchos). The core repository is currently private. If you need it access to either cachesaver-core or anything else, please feel free to [email us](mailto:nearchos.potamitis@cs.au.dk) and ask for it.

## Outer links
- [Cachesaver-Core github repository](https://github.com/au-clan/cachesaver)
- [Implementation schedule](https://docs.google.com/spreadsheets/d/197HLUTozH9usUUboGlSNpkFJqjBezceA2snmEN5btbQ/edit?usp=sharing)

## Contributing
Multiple people are working on this project so we establish some rules.
- If you're working on something or you're planning to work on something, please indicate it in the [implementation schedule](https://docs.google.com/spreadsheets/d/197HLUTozH9usUUboGlSNpkFJqjBezceA2snmEN5btbQ/edit?usp=sharing) so that there's no two people working on the same thing.
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

**Note:** You might notice in the instructions below that you are not cloning the main branch. This will be resolved in the next week.

```bash
git clone --branch nearchos --single-branch https://github.com/au-clan/cachesaver-core.git
cd cachesaver-core
conda activate cachesaver
pip install -e ".[test]"
pytest test/ -v # Run tests to verify everything works
```

You'll also need to firstly install any LLM APIs that you'd like to use (eg Groq, OpenAI, Together, etc.), and secondly, save their corresponding keys.

## Modules
The project is divided into four main modules: Agents, Frameworks, Tasks and Models.

### Models
The models are essentialy the last layer of the Cachesaver module. This means that it's an object that receives a `cachesaver.Request` and returns a `cachesaver.Response` (you can look for more info of the cachesaver-core in its [readme](https://github.com/au-clan/cachesaver-core/blob/nearchos/README.md)). Essentialy there's three steps in this:
- Parse the prompt and the decoding parameters from the `cachesaver.Request` object.
- Generate something using an LLM.
- Wrap the response in the `cachesaver.Response` object (cachesaver.Response.data=<your_response>).

You can find an example of an online LLM [here](https://github.com/au-clan/cachesaver/blob/main/src/models/model_online.py). This model can accomodate clients such as `AsyncOpenAI` and `AsyncTogether`. It can also work with `AsyncGroq` but generally Groq limits the generation samples to `n=1` which is not handled within the example.

### Tasks
The task module is the module responsible for multiple methods related to a task.
It is divided in the following:

- `DataBasic`: A class responsible for loading the dataset.
- `StateBasic`: A class having a concrete representation of a state of the task.
- `Prompts`: Prompts for specific methods, unique to the task.
- `EnvironmentBasic`: A class for the environment of the task. Currently there's two distinctions within this class. Firstly, there's a prompter and a parser, directly responsible for specific actions related to the task and the LLM. Secondly, there are `StateBasic` handling methods. For example, given a state and an action the next state is returned.

### Agents
The agent is directly interacting with an environment in a task-agnostic way. You can look at `AgentLLM` for an example. This agent is designed to use an LLM to perform operations used in different reasoning frameworks.

### Frameworks
The task-agnostic implementations for different reasoning framework.

## Expanding a module
In the following, we describe things that need to be checked when someone is exapnding a module. If at any point of your work you need to install new libraries, please add a comment in `./requirements.txt` with your task's name and the corresponding libraries (and versions).

### Models
The models are interacting solely with the agents and they are a layer to the Cachesaver API. As long as they inherit `src.models.model_basic.ModelBasic` they're fine.

### Tasks
To add a task you'll need to add all of its submodules: `DataBasic`, `StateBasic`, `EnvironmentBasic` and `Prompts`. The best way to navigate this is to look at the basic classes in `src.tasks.basic`, their abstract methods and their types. Then you can check the examples in `./src.tasks.game24` or `./src.tasks.hotpotqa` to see how they're used.

### Frameworks
There's a few scenarios in cases where a new Framework needs to be added:
1. All the operations required by the reasoning framework are included in `src.agents.agent_llm.AgentLLM`: In this case you can simply add your framework in `./src.frameworks`.
2. An agent operation needs to be added and to add that operation you can use the task environments as they are: In this case, additionally to adding your framework you'll need to add the operation to `src.agents.agent_llm.AgentLLM`.
3. An agent operation needs to be added but to add this operation the task environments need to be updated: In this case, additionally to adding your framework and the agent operation, you'll need to update all task environments to accomodate the needs of the new agent operation.

Finally, the actual data from the task (eg. the puzzles) need to be stored in `./datasets`. If possible, please maintain the current format of `*.csv.gz`. If the data is too big please re-define the `src.tasks.basic.data_basic.DataBasic.download_data()` method of your task.

## Misc
### Local folder
You can create and name a folder named `./local`. Anything you save there will not be pushed. If possible, please use this instead of adding specific files to `.gitignore`.