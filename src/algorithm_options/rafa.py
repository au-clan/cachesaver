from dataclasses import dataclass, field
from typing import TypedDict, Literal, List, Optional


@dataclass(frozen=True)
class RAFAOptions:
    max_step: int
    n_generate_sample: int
    n_evaluate_sample: int
    n_select_sample: int
    n_propose_sample: int


##For the requests
class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class RequestOptions:
    ## These two are for the request / namespace option in cacheSaver
    namespace: str
    request_id: str
    ##Standard settings below
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    seed: Optional[int] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[float] = None


@dataclass
class RafaRequest(RequestOptions):
    messages: List[Message] = field(default_factory=list)
    n: int = 1
    stop_token: List[str] = field(default_factory=list)

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def last_message_content(self) -> str | None:
        if self.messages:
            return self.messages[-1]["content"]
        return None

    @classmethod
    def from_request_options(cls, request_options: RequestOptions, **kwargs) -> "RafaRequest":
        return cls(**vars(request_options), **kwargs)


@dataclass(frozen=True)
class GameState_rafa:
    cur_step: Optional[int] = 0
    index: int = 0

    puzzle: Optional[str] = None
    ##used attributes
    obs_feedback: Optional[str] = ""
    obs_answer: Optional[str] = ""

    reflects: Optional[list[str]] = field(default_factory=list)
    value_reflects: Optional[list[str]] = field(default_factory=list)

    obs_history: Optional[list[dict[str, str]]] = field(default_factory=list)
    env_history: Optional[list[str]] = field(default_factory=list)

    history: Optional[list[str]] = field(default_factory=list)
