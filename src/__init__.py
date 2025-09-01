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
            return cls.registry[key](*args, **kwargs)
        except KeyError:
            raise ValueError(f"No agent found for type={agent_type}, benchmark={benchmark}")
        
class MethodFactory:
    registry = {}

    @classmethod
    def register(cls, algo_cls):
        cls.registry[algo_cls.__name__.lower()] = algo_cls
        return algo_cls

    @classmethod
    def get(cls, method: str, *args, **kwargs):
        key = f"method{method}".lower()
        try:
            return cls.registry[key](*args, **kwargs)
        except KeyError:
            raise ValueError(f"No method found for name={method}")