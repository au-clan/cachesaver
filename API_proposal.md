The following three classes follow the standard RL setup, with a couple key changes that will hopefully make them more reusable.
The `Agent` doesn't have just an `act` method, but a `discover_actions`. This makes it possible to return a list of actions. Very important for ToT.
The `Agent` also has methods to serialize / clone. Important for FoA.
In practice, the `Agent` will probably be a thin wrapper around the prompting mechanism for discovering new thoughts/actions.
This also means that we'll have individual agents for each task.

The `Environment` is pretty standard. The `step` function returns a new state and a reward (probably almost always 0, since we don't get intermediate reward signal).
Then there's some helper functions, they partially overlap: `is_valid` to prune out invalid actions,  `is_final` to check if a state is terminal, `evaluate` to look at the ground truth whether a state is a successful solution.
To be determined which of these helpers we keep.

I've given both the `Agent` and the `Environment` a set of `seed`, `serialize` and `clone` functions.
This assumes that they may be non-deterministic and that they hold inner state that we may want to clone.
Alternatively we could define helper classes such as `AgentState` and `EnvironmentState`, define `clone` etc on those `XXXState` classes and make all the methods on `Agent`, `Environment`, etc `@abstractmethods`.
Up for discussion. 

Finally, there's one interesting piece here: a `Heuristic` class. While the agent will probably encode the prompt for discovering actions, the heuristic encodes prompts for estimating the value.
To keep with the RL terminology I've given it two methods `value` and `advantage`. One is for estimating the value of a state. One is for estimating the advantage of a `state, action` combo.
In practice, for the algorithms we have so far, we'll use `value`.
The `Heuristic` can have a state!

This code just documents the API
```python
class Agent:
    def discover_actions(self, state) -> list of actions
    def seed(self, randomness)
    def serialize(self)
    def clone(self, randomness) # combination of clone and seed

class Environment:
    def initialize(initial_state)
    def step(self, action) -> state, reward

    def is_valid(self, state, action) -> bool
    def is_final(self, state) -> bool
    def evaluate(self, state) -> is_final, value

    def seed(self, randomness)
    def serialize(self)
    def clone(self, randomness)

class Heuristic:
    def value(self, state) -> float
    def advantage(self, state, action) -> float
    def serialize(self)
    def seed(self, randomness)
    def clone(self, randomness) # combination of clone and seed
```

Now, I think these classes are nicely reusable. For ToT, we can use the `Agent` to discover a set of new thoughts, then we can use the `Heuristic` to give them a value and select the best ones. The tree search algorithm bfs or dfs are implemented outside of `Agent`
For FoA, we give some inner state to the `Agent` and rely heavily on the `clone` mechanism.

The Agent, Environment, Heuristic are all task specific.
We create an `Algorithm` class that can use a specific combo to try and work on a specific instance of a task.
We collect instances of a task into a benchmark. For the benchmark we follow the pytorch dataset API.
Something like this:
```python

class Algorithm:
    def initialize(Agent, Environment, Heuristic)
    def solve(self, initial_state):
    def benchmark(self, Benchmark)

class FoA(Algorithm):
    concrete implementation of solve

class Benchmark:
    def len() -> int
    def get(idx) -> initial_state
```

So `FoA` is a specific algorithm, and putting it all together could look like this:

```python
from gameof24 import GameOf24Agent, GameOf24Heuristic, GameOf24Environment, GameOf24Benchmark
from algorithms import FoA

go24foa = FoA(GameOf24Agent, GameOf24Heuristic, GameOf24Environment)

# run it
go24foa.benchmark(Benchmark)
```
