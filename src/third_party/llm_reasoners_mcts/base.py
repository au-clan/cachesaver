from typing import Generic, TypeVar, Union, Protocol, Optional, runtime_checkable, Tuple
from abc import ABC, abstractmethod

from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]


class Dynamics(ABC, Generic[State, Action]):

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

class WorldModel(Dynamics, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    def update_example(self, example: Example, prompt = None) -> None:        
        if prompt is not None:
            self.prompt = prompt
        self.example = example

    async def is_solved(self, state: State) -> bool:
        raise NotImplementedError

class Environment(Dynamics, Generic[State, Action]):
    def __init__(self) -> None:
        self.env = None

class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example, prompt = None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> AlgorithmOutput: ...


class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: Dynamics[State, Action],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.dynamics = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Optional[Example] = None, prompt = None, **kwargs) -> AlgorithmOutput[State]:
        if isinstance(self.dynamics, WorldModel):
            if example is None:
                raise ValueError("An example must be provided when using WorldModel")
            self.dynamics.update_example(example, prompt=prompt)
            self.search_config.update_example(example, prompt=prompt)
        return self.search_algo(self.dynamics, self.search_config, **kwargs)

class Evaluator():
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample_prompt(self,
                      shuffle_prompt,
                      num_shot,
                      sample_prompt_type):
        pass
    
    def evaluate(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 log_dir=None):

        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0

        disable_tqdm = self.disable_tqdm or \
            (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        for i, example in enumerate(tqdm(self.dataset,
                                            total=resume + len(self.dataset),
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            
            algo_output = reasoner(self.input_processor(example),
                                    prompt=self.sample_prompt(
                                        shuffle_prompt=shuffle_prompt,
                                        num_shot=num_shot))
            
            output = self.output_extractor(algo_output)
            answer = self.answer_extractor(example)
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
            
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
                    pickle.dump(algo_output, f)
        
        return accuracy

    def evaluate_sc(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 n_sc = 10,
                 log_dir=None):

        self.dataset = list(self.full_dataset)[resume:]
        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            os.makedirs(log_dir, exist_ok=resume > 0)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0

        disable_tqdm = self.disable_tqdm or \
            (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
        for i, example in enumerate(tqdm(self.dataset,
                                            total=resume + len(self.dataset),
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            
            prompt = self.sample_prompt(
                            shuffle_prompt=shuffle_prompt,
                            num_shot=num_shot)
            output_list = []
            save_list = []
            for j in range(n_sc):
                algo_output = reasoner(self.input_processor(example),
                                    prompt=prompt)
                terminal_state = algo_output.terminal_state
                path = ""
                for k in range(len(terminal_state)):
                    path += terminal_state[k].sub_question + " " + terminal_state[k].sub_answer + " "
                save_list.append(path)
                output = self.output_extractor(algo_output)
                output_list.append(output)
                answer = self.answer_extractor(example)
            from collections import Counter
            output = Counter(output_list).most_common(1)[0][0]
            correct = self.eval_output(answer, output)
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
                with open(os.path.join(log_dir, 'algo_output.txt'),'a') as f1:
                    print(save_list, file=f1)
        
        return accuracy
    @abstractmethod
    def eval_output(self, answer, output):
        pass
