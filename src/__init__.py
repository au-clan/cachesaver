from typing import TypedDict
from .typedefs import DecodingParameters

class BenchmarkFactory:
    registry = {}

    @classmethod
    def register(cls, benchmark_cls):
        cls.registry[benchmark_cls.__name__.lower()] = benchmark_cls
        return benchmark_cls
    
    @classmethod
    def get(cls, task: str, *args, **kwargs):
        key = f"benchmark{task}".lower()
        try:
            return cls.registry[key](path=f"datasets/dataset_{task}.csv.gz",*args, **kwargs)
        except KeyError:
            raise ValueError(f"No benchmark found for task={task}")
        
class EnvironmentFactory:
    registry = {}

    @classmethod
    def register(cls, env_cls):
        cls.registry[env_cls.__name__.lower()] = env_cls
        return env_cls
    
    @classmethod
    def get(cls, task: str, *args, **kwargs):
        key = f"environment{task}".lower()
        try:
            return cls.registry[key](*args, **kwargs)
        except KeyError:
            raise ValueError(f"No environment found for task={task}")
    
class AgentFactory:
    registry = {}

    @classmethod
    def register(cls, agent_cls):
        cls.registry[agent_cls.__name__.lower()] = agent_cls
        return agent_cls

    @classmethod
    def get(cls, agent_type: str, benchmark: str, *args, **kwargs):
        key = f"agent{agent_type}{benchmark}".lower()
        try:
            return cls.registry[key]#(*args, **kwargs) : Not initialized
        except KeyError:
            raise ValueError(f"No agent found for type={agent_type}, benchmark={benchmark}")

class AgentDictFactory:
    registry = {}

    @classmethod
    def register(cls, agent_dict_cls):
        cls.registry[agent_dict_cls.__name__.lower()] = agent_dict_cls
        return agent_dict_cls

    @classmethod
    def get(cls, method: str, *args, **kwargs):
        key = f"agentdict{method}".lower()
        try:
            return cls.registry[key](*args, **kwargs)
        except KeyError:
            raise ValueError(f"No agent dict found for method={method}")


class MethodFactory:
    registry = {}

    @classmethod
    def register(cls, method_cls):
        cls.registry[method_cls.__name__.lower()] = method_cls
        return method_cls

    @classmethod
    def get(cls, method: str, benchmark: str, params: DecodingParameters, *args, **kwargs):
        key = f"method{method}".lower()

        
        if method == "io":
            agents = {
                "step": AgentFactory.get("io", benchmark),
            }
        elif method == "cot":
            agents = {
                "step": AgentFactory.get("cot", benchmark),
            }
        elif method == "foa":
            agents = {
                "step": AgentFactory.get("act", benchmark),
                "evaluate": AgentFactory.get("evaluate", benchmark),
            }
        elif method in ["tot_bfs", "tot_dfs"]:
            agents = {
                "step": AgentFactory.get("bfs", benchmark),
                "evaluate": AgentFactory.get("evaluate", benchmark),
            }
        elif method == "got":
            agents = {
                "step": AgentFactory.get("act", benchmark),
                "aggregate": AgentFactory.get("aggregate", benchmark),
                "evaluate": AgentFactory.get("evaluate", benchmark),
            }
        elif method == "rap":
            agents = {
                "step": AgentFactory.get("react", benchmark),
                "evaluate": AgentFactory.get("selfevaluate", benchmark),
            }
        elif method == "react":
            agents = {
                "step": AgentFactory.get("react", benchmark),
            }
        else:
            raise NotImplementedError(f"Method {method} is not implemented yet.")
        
        # For the moment, only supporting same params for all agents
        agents.update({k+"_params": params for k in agents.keys()})

        try:
            return cls.registry[key](agents=agents, *args, **kwargs)
        except KeyError:
            raise ValueError(f"No method found for name={method}")