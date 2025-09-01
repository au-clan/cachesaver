import random
import logging
import asyncio
from typing import TypedDict, List, Tuple, Optional
from omegaconf import OmegaConf
from ..typedefs import Method, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from .. import MethodFactory, AgentDictFactory
import numpy as np

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state: State, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0
        self.actions = []
        self.is_terminal = False
        self.reward = 0.0
        logger.debug(f"Created new Node with state: {state.current_state}")

    def ucb(self, exploration_constant: float) -> float:
        if self.visits == 0:
            logger.debug(f"Node with 0 visits, returning inf for UCB")
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_constant * np.sqrt(np.log(self.parent.visits) / self.visits)
        ucb_value = exploitation + exploration
        logger.debug(f"Calculated UCB value: {ucb_value} (exploitation: {exploitation}, exploration: {exploration})")
        return ucb_value

    def add_child(self, state: State, action: str) -> 'Node':
        child = Node(state, parent=self)
        child.actions = self.actions + [action]
        self.children.append(child)
        logger.debug(f"Added child node with action: {action}, new state: {state.current_state}")
        return child

    def update(self, value: float):
        self.visits += 1
        self.value += value
        logger.debug(f"Updated node: visits={self.visits}, value={self.value}, avg_value={self.value/self.visits}")

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.update(value)
            logger.debug(f"Backpropagating value {value} to node with state: {node.state.current_state}")
            node = node.parent

@AgentDictFactory.register
class AgentDictRAP(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    evaluate_params: DecodingParameters

@MethodFactory.register
class MethodRAP(Method):
    def __init__(self,
                model: Model,
                agents: AgentDictRAP,
                env: Environment,
                config: OmegaConf
                ):
        super().__init__(model, agents, env, config)
        logger.debug("Initializing RAP Method with parameters:")
        logger.debug(f"num_iterations: {config.num_iterations}")
        logger.debug(f"num_samples: {config.num_samples}")
        logger.debug(f"num_evaluations: {config.num_evaluations}")
        logger.debug(f"exploration_constant: {config.exploration_constant}")
        logger.debug(f"max_depth: {config.max_depth}")

        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]

        self.step_params = agents["step_params"]
        self.evaluate_params = agents["evaluate_params"]

        self.num_iterations = config.num_iterations
        self.num_samples = config.num_samples
        self.num_evaluations = config.num_evaluations
        self.exploration_constant = config.exploration_constant
        self.max_depth = config.max_depth

    async def select(self, node: Node) -> Node:
        logger.debug(f"Starting selection from node with state: {node.state.current_state}")
        while node.children and not node.is_terminal:
            if len(node.children) < self.num_samples:
                logger.debug(f"Found expandable node with {len(node.children)} children")
                return node 
            
            ucb_values = [child.ucb(self.exploration_constant) for child in node.children]
            best_child_idx = np.argmax(ucb_values)
            node = node.children[best_child_idx]
            logger.debug(f"Selected child with UCB value: {ucb_values[best_child_idx]}, state: {node.state.current_state}")
        
        logger.debug(f"Selection complete, returning node with state: {node.state.current_state}")
        return node

    async def expand(self, node: Node, namespace: str, request_id: str) -> Node:
        logger.debug(f"Expanding node with state: {node.state.current_state}")
        
        action_coroutines = [
            self.step_agent.act(
                model=self.model,
                state=node.state,
                n=1,
                namespace=namespace,
                request_id=f"{request_id}-expand{i}",
                params=self.step_params
            )
            for i in range(self.num_samples)
        ]
        actions = await asyncio.gather(*action_coroutines)
        logger.debug(f"Generated {len(actions)} action sets")

        for action_list in actions:
            for action in action_list:
                logger.debug(f"Processing action: {action}")
                new_state = self.env.step(node.state.clone(randomness=random.randint(0, MAX_SEED)), action)
                child = node.add_child(new_state, action)
                
                is_final, reward = self.env.evaluate(new_state)
                if is_final:
                    child.is_terminal = True
                    child.reward = reward
                    logger.debug(f"Found terminal state with reward {reward}")
                    if reward == 1.0:
                        logger.debug("Found solution!")
                        return child

        selected_child = node.children[0] if node.children else node
        logger.debug(f"Expansion complete, selected child with state: {selected_child.state.current_state}")
        return selected_child

    async def simulate(self, node: Node, namespace: str, request_id: str) -> float:
        logger.debug(f"Starting simulation for node with state: {node.state.current_state}")
        
        value_coroutines = [
            self.eval_agent.act(
                model=self.model,
                state=node.state,
                n=self.num_evaluations,
                namespace=namespace,
                request_id=f"{request_id}-simulate{i}",
                params=self.evaluate_params
            )
            for i in range(self.num_evaluations)
        ]
        values = await asyncio.gather(*value_coroutines)
        avg_value = sum(values) / len(values)
        logger.debug(f"Simulation complete, average value: {avg_value}")
        return avg_value

    async def mcts_search(self, root_state: State, namespace: str, request_id: str, idx: int) -> Tuple[State, List[str]]:
        logger.debug(f"Starting MCTS search from root state: {root_state.current_state}")
        root = Node(root_state)
        best_state = root_state
        best_value = 0
        best_actions = []

        for iteration in range(self.num_iterations):
            if iteration % 5 == 0:
                print(f"Step: {iteration} ({idx})")
            logger.debug(f"\nMCTS Iteration {iteration + 1}/{self.num_iterations}")
            
            logger.debug("Starting selection phase")
            node = await self.select(root)
            
            if not node.is_terminal and len(node.children) < self.num_samples:
                logger.debug("Starting expansion phase")
                node = await self.expand(node, namespace, f"{request_id}-iter{iteration}")
                if node.is_terminal and node.reward == 1.0:
                    logger.debug("Found solution during expansion!")
                    return node.state, node.actions

            logger.debug("Starting simulation phase")
            value = await self.simulate(node, namespace, f"{request_id}-iter{iteration}")
            
            logger.debug("Starting backpropagation phase")
            node.backpropagate(value)

            if value > best_value:
                best_value = value
                best_state = node.state
                best_actions = node.actions
                logger.debug(f"Updated best state with value: {best_value}")

        logger.debug(f"MCTS search complete. Best value found: {best_value}")
        return best_state, best_actions

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        logger.debug(f"\nStarting solve for index {idx}")
        randomness = idx
        random.seed(randomness)
        
        root_state = state.clone(randomness=random.randint(0, MAX_SEED))
        logger.debug(f"Initialized root state: {root_state.current_state}")
        
        best_state, best_actions = await self.mcts_search(
            root_state, 
            namespace, 
            f"idx{idx}",
            idx=idx
        )

        is_final, reward = self.env.evaluate(best_state)
        if is_final and reward == 1.0:
            logger.debug("Found solution!")
            return [best_state]

        logger.debug("No solution found, returning best state")
        return [best_state]

    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        logger.debug(f"Starting benchmark with share_ns={share_ns}, cache={cache}")
        cache = {} if cache else None
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark_{index}",
                value_cache=cache
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        logger.debug("Benchmark complete")
        return results
