###############
# The Overall Idea
# The only things we essentially care about are:
#
# 1. That the framework accepts a specific input structure containing our puzzles
#
# 2. That we can verify the solution outputted by the framework in a consistent and standardized way across all frameworks.
#
# We care quite a lot about having a centralized verifier that's used across all frameworks. This ensures that frameworks cannot implement their own, potentially looser verifiers that might accept incorrect or overly permissive outputs.
#
# In that sense, we can treat the framework itself as a bit of a black box. If a framework has its own internal logic or structure, that’s totally fine, as long as it adheres to the standardized output format, uses the official API/BatchRequests, and possibly the interfaces defined in the type.def file.
#
# Each framework will then produce an output in a shared, consistent format. This makes comparison across frameworks straightforward, all we need to do is look at the standardized output structure.

##########
# Auditing & Logging
# Beyond correctness, we also want to audit what the framework did internally, the steps it took, actions it performed, decisions it made. For that, we define a base log structure that, at a minimum, includes the puzzle and the resulting output.
#
# For more advanced frameworks (e.g. those with reasoning chains, feedback loops, or CoT-style steps), they can extend this logging to include internal steps or custom actions. This gives us observability into the black box without constraining its internal design.

######
# Final Thought
# I’m still trying to fully understand the reasoning behind enforcing structure on frameworks. I can see that having a shared structure likely helps with maintainability and comparison in the long run, especially as more frameworks are added, otherwise things could get pretty messy. At the same time, I wonder if the initial complexity and overhead might make it harder for someone new to get started. The structure adds clarity and consistency, but it also means a new contributor has to take the time to understand and adapt to it before they can really contribute. I’m guessing this tradeoff pays off at scale, but I’m still wrapping my head around where that balance lies.
###############


# input -> black box -> output: solution to puzzle, log of framework specific steps
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass, dataclass
from dataclasses import field
from typing import Any, Optional, Type, TypeVar, List, Dict, runtime_checkable, Protocol, NamedTuple

#------------------------------------------------------------State----------------------------------------------------
T = TypeVar('T', bound='BaseState')  # todo same file as BaseState


@dataclass(frozen=True)
class BaseState(ABC):
    puzzle: Any

    # ... additional fields that should be shared across

    # this is used in the logging
    def to_json(self) -> str:
        return json.dumps(self._serialize())

    # feature/debug if we want to read up a log file´and reproduce etc
    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        data = json.loads(json_str)
        return cls._deserialize(data)

    def _serialize(self) -> dict:
        def convert(value):
            if is_dataclass(value):
                return asdict(value)
            return value

        return {k: convert(v) for k, v in asdict(self).items()}

    @classmethod
    def _deserialize(cls: Type[T], data: dict) -> T:
        return cls(**data)


# Comment for extending the baseState
# 1)
# The state is different for different puzzle and frameworks, so here i either suggest we extend the baseclass with
# a framework and task specific state this could be something like(framework:rafa, task:Game24):
@dataclass(frozen=True)
class RAFAGame24State(BaseState):
    history: list[str]
    feedbacks: list[str]
    reflects: list[str]
    value_reflects: list[str]


# 2)
# alternatively to this task and framework specific approach one could make a framework specific
# state that should work across different task such as:
@dataclass(frozen=True)
class RAFAState(BaseState):
    history: Optional[list[str]] = field(default_factory=list)
    feedbacks: Optional[list[str]] = field(default_factory=list)
    reflects: Optional[list[str]] = field(default_factory=list)
    value_reflects: Optional[list[str]] = field(default_factory=list)
    some_list_used_for_a_different_task: Optional[list[str]] = field(default_factory=list)

#------------------------------------------------------------Environment----------------------------------------------------

# base environment that all Tasks will have
class BaseEnvironment(ABC):
    class Prompter:
        # here should the default prompt techniques go, this could be cot,
        @staticmethod
        def input(string_to_prompt: str) -> str:  # this is a simple prompt
            pass

    @staticmethod
    @abstractmethod
    def verify_puzzle_output(puzzle: Any, framework_solution, correct_solution):
        pass


class GameOf24Environment(BaseEnvironment):
    @staticmethod
    def verify_puzzle_output(puzzle: Any, framework_solution, correct_solution):
        if framework_solution == correct_solution:
            return True  # maybe not a single truth value but some datastructures
            # that captures both the puzzle and the framework solution


class HotChocolateyEnvironment(BaseEnvironment):
    @staticmethod
    def verify_puzzle_output(puzzle: Any, framework_solution, correct_solution):
        if framework_solution == correct_solution:
            return True  # maybe not a single truth value but some datastructures
            # that captures both the puzzle and the framework solution


# framework environment, here we have the prompts and parsers for all the puzzles
# required for the framework, this could be split up further at some point
class RAFAEnvironment:
    def __init__(self, puzzle_env: BaseEnvironment):
        self.puzzle_env = puzzle_env  # this enables be to be puzzle agnostic

    class prompter:
        @staticmethod
        def format_prompt_rafa_style1(some_input: Any):
            return f"Style1 prompt: {some_input}"

        @staticmethod
        def format_prompt_rafa_style2(some_input: Any):
            return f"Style2 prompt: {some_input}"

    class parser:
        @staticmethod
        def parse_rafa_style1(some_input: Any):
            return some_input.strip()

        @staticmethod
        def parse_rafa_style2(some_input: Any):
            return some_input.strip()

    def verify(self, puzzle, framework_solution, correct_solution):
        return self.puzzle_env.verify_puzzle_output(puzzle, framework_solution, correct_solution)


#------------------------------------------------------------AGent/API----------------------------------------------------
@dataclass(frozen=True)
class BaseRequest:
    prompt: str
    n: int
    request_id: str
    namespace: str
    messages:Optional[List[Any]]=field(default_factory=list)

@dataclass(frozen=True)
class Request(BaseRequest):
    max_completion_tokens: int = None
    temperature: float = None
    top_p: float = None
    stop: str = None

class Batch(NamedTuple):
    requests: List[BaseRequest]


class Response(NamedTuple):
    data: List[Any]
    cached: List[bool] = None
    duplicated: List[bool] = None

@runtime_checkable
class BatchRequestModel(Protocol):
    """Model that processes multiple requests at once, returns multiple results per request."""

    async def batch_request(self, batch: Batch) -> List[Response]: ...

class AgentLLM():
    def __init__(self, api: BatchRequestModel):
        super().__init__()
        self.name = "LLM Agent"
        self.api = api
        self.calls = {"total": 0, "cached": 0, "duplicated": 0}
        self.tokens = {
            "total": {"in": 0, "out": 0},
            "cached": {"in": 0, "out": 0},
            "generated": {"in": 0, "out": 0}
        }

    async def request(self, prompt: str, n: int, request_id: str, namespace: str, config: DictConfig,
                      messages: List[Any] = None) -> List[Any]:
        """
        Makes a request to the api and tracks the number of calls.
        """
        request = Request(
            prompt=prompt,  # todo maybe just use this for storing in cache? idk
            messages=messages,
            n=n,
            request_id=request_id,
            namespace=namespace,
            max_completion_tokens=config.max_completion_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop
        )

        response = await self.api.batch_request(Batch(requests=[request]))
        self.calls = {
            "total": self.calls["total"] + sum(len(r.data) for r in response),
            "cached": self.calls["cached"] + sum(sum(r.cached) for r in response if r.cached),
            "duplicated": self.calls["duplicated"] + sum(sum(r.duplicated) for r in response if r.duplicated)
        }

        all_data = [item for r in response for item in r.data]
        all_cached = [flag for r in response for flag in (r.cached or [])]

        messages, tokin, tokout = zip(*all_data)

        # Step 3: Compute metrics
        cached_tokin = [int(tokens * cached) for tokens, cached in zip(tokin, all_cached)]
        cached_tokout = [int(tokens * cached) for tokens, cached in zip(tokout, all_cached)]
        generated_tokin = [int(tokens * (not cached)) for tokens, cached in zip(tokin, all_cached)]
        generated_tokout = [int(tokens * (not cached)) for tokens, cached in zip(tokout, all_cached)]

        self.tokens["total"]["in"] += sum(tokin)
        self.tokens["total"]["out"] += sum(tokout)
        self.tokens["cached"]["in"] += sum(cached_tokin)
        self.tokens["cached"]["out"] += sum(cached_tokout)
        self.tokens["generated"]["in"] += sum(generated_tokin)
        self.tokens["generated"]["out"] += sum(generated_tokout)
        return messages

#------------------------------------------------------------Framework----------------------------------------------------
class BaseFramework(ABC):
    @abstractmethod
    def run(self, states: List[BaseState]) -> List[Dict[str, Any]]:
        """
        Accepts a list of puzzle states and returns the framework's solutions
        plus evaluation from the puzzle environments.
        """
        pass

class RAFAFramework(BaseFramework):
    def __init__(self, environment: RAFAEnvironment, config: Any):
        self.environment = environment  # framework-specific prompts & parsers
        #the agent is a bit different as all the logic will be moved to the framework and we will instead of
        # using the agent we will use the api directly, this way we can capture tokens etc wit hthe current setup

        self.config=config #same as current setup with the different config values


    def run(self, states: List[BaseState]) -> List[Dict[str, Any]]:
        results = []
        for state in states:
            puzzle = state.puzzle

            # BLACK BOX: framework-specific logic goes here
            framework_solution = self.solve(state)

            # SECURE EVAL: only the puzzle_env knows how to verify
            verification = self.environment.puzzle_env.verify_puzzle_output(
                puzzle=puzzle,
                framework_solution=framework_solution,
                correct_solution=None  # optional, depends on your setup
            )

            results.append({
                "puzzle": puzzle,
                "solution": framework_solution,
                "verification": verification,
                "state": state
            })
        return results

    def solve(self, state: BaseState) -> Any:
        """
        This is where framework-specific logic goes, e.g., prompting a model.
        For now, just returns dummy string.
        """
        #############
        ### ALL   ###
        ### LOGIC ###
        ### FOR   ###
        ### framework   ###
        #############
        #obs in this class there could potentially be helper functions
        prompt = self.environment.prompter.format_prompt_rafa_style1(state.puzzle)
        response = self.fake_model_response(prompt)
        return self.environment.parser.parse_rafa_style1(response)

