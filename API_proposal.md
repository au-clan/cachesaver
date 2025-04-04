Code is divided in 4 main sections

1. Framework
2. Agent
3. Task
    1. Prompts
    2. Data
    3. State
    4. Environment
4. Model

- In agreement with RL: Framework is both Agent and Task agnositc.
- In disagreement with RL: Agent is Task agnostic

## 1. Framework
```python
def Framework:
    def initialize(Agent, Environment)
    def run(puzzle_idx)
```

The `Framework` is based on operations defined in the `Agent` and heuristics such as state resetting and verification of the `Environment`. A `Dummy` framework that performs a single step and an evaluation would look like this:

```python
def Dummy(Framework):
    def initialize(agent: Agent, environment: Environmet):
        self.agent = agent
        self.environment = environment

    def run(puzzle_idx: int):
        state = self.environment.reset(puzzle_idx)
        next_state = self.agent.act(state, environment)
        value = self.agent.evaluate(state)
        verification = self.environment.verify(state)
        return next_state, value, verification
```

**A note on `Verification`:** Essentially captures whether the state is terminal and correct, along with a potential message which can serve either for logging (error analysis) or reflexion.

```python
class Verification(NamedTuple):
    finished: bool
    correct: bool
    message: str
```

## 2. Agent
The LLM Agent uses the environment and the `Model` to define standard operations found frequently in reasoning frameworks. 

```python
class Agent:
    def initialize(api)
    def request(prompt)
    def act(state, environment)
    def react(state, environment)
    def reflect(state, environment)
    def evaluate(state, environment)
```

The operations of the LLM Agent can be broken down in 3 steps:
1. The `Environment` provides a `prompt` based on the current `state`.
2. The `Agent` uses the `prompt` and the Model to generate a `Response``
3. The `Environment`parses the response into an `Inference`.
4. The `Agent` uses the `Inference` to complete the operation.

Some examples are:

```python
class LLMAgent(Agent):
    def act(state, environment):
        prompt = environment.prompter.act(state)
        response = self.request(prompt)
        inference = environment.parser.act(state)
        next_state = environment.get_next_state(inference)
        return next_state
    
    def act(state, environment):
        prompt = environment.prompter.act(state)
        response = self.request(prompt)
        inference = environment.parser.act(state)
        value = inference.value
        return value
```

The `Inference` usually thinking tokens (? Not sure how to name this) that you find in reasoning models. Particularly:

```python
class Inference(NamedTuple):
    action: Optional[str] = None
    thought: Optional[str] = None
    reflexion : Optional[str] = None
    value: Optional[int] = None
```

## 3. Task

### Prompts
A python file with prompts, nothing much here.

### Data
Serves two purposes:
1. A dataloader, reading the data from a csv.
2. Defines the indices for each set.

```python
class Data:
    def initialize(csv_path)
    def get_set_idxs(set_name: str) -> List[int]
    def read(idx)
```

### State
A representation of the current state and its past actions. For different tasks the variables of the state can change. For example, HotpotQA needs a vectorspace to be included in the `State`.

```python
class State:
    puzzle: str
    current_state: str
    steps : List[str]
    randomness : int

    def __hash__()
    def log() # I overdid it with logging: i'll completely remake
    def duplicate(randomness) -> State
```

### Environment
Big point of contention here are the `Prompter` and the `Parser`. It's completely true that they never direclty interact with the environment. It just felt natural to me that each `Environment` has its own prompter/parser. Thinking more about this, I agree with in the sense that the prompter/parser are not of the `Environment` but of the task. However, I also see that they simplify the code.

```python
class Environment:
    class Prompter
    class Parser
    def get_next_state(Inference, State) -> State # Should rename to step
    def get_value(State) -> Optional[int] # Heuristic evaluation if it exists
    def verify(State) -> Verification
```