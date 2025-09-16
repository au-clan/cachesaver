###################
###---Prompts---###
###################

io = """You will be given a question. Simply provide the final answer. Do not provide any explanations or intermediate steps.  The format to respond is "Final Answer: ..."""

cot = """You will be given a question. Think step by step and provide your reasoning before giving the final answer. The format to respond is "Final Answer: ...", where "..." is the final answer."""


act = """Solve a human-labeled explanation task with sequential Action steps. Action can be three types:

(1) Analyze[topic], which analyzes the given topic in the context of the question and image.
(2) Explain[aspect], which provides explanations about specific aspects of the topic.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next action. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

react = """Solve a human-labeled explanation task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Analyze[topic], which analyzes the given topic in the context of the question and image.
(2) Explain[aspect], which provides explanations about specific aspects of the topic.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

bfs = """We're solving a human-labeled explanation task with sequential Action steps. Your task is to propose possible next actions given the current trajectory. Action can be three types:

(1) Analyze[topic], which analyzes the given topic in the context of the question and image.
(2) Explain[aspect], which provides explanations about specific aspects of the topic.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to propose immediate next actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}

Possible Actions:"""

evaluate = '''Analyze the trajectories of a solution to a human-labeled explanation task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types:
(1) Analyze[topic], which analyzes the given topic in the context of the question and image.
(2) Explain[aspect], which provides explanations about specific aspects of the topic.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.

Below some examples are given.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the latest available thought, action, and observation based on your reasoning analysis. Answer in the format given by the examples and mention nothing more. Make sure to indicate the correctness score at the end of your answer in the following format: "Correctness score : <score>".

Question: {question}
{current_state}

Evaluation:
'''

### Judge prompt to evaluate the final answer correctness based on the ground truth answer.
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""



################################
###---Examples for fewshot---###
################################
examples_bfs = [
"""
What is the purpose of the illustrated UI component?

Possible Actions:
Analyze[UI component layout]
Analyze[visual elements]
Analyze[interactive features]
Analyze[user flow]
Explain[component functionality]
Explain[design patterns]
""",

"""Question: How does this visualization represent the data hierarchy?
Action 1: Analyze[visualization structure]
Observation 1: The visualization uses a tree-like structure with connected nodes and branches to show relationships between different data elements.

Possible Actions:
Explain[node relationships]
Explain[visual hierarchy]
Explain[data flow]
Analyze[node types]
Analyze[connection patterns]
Finish[hierarchical tree structure]
"""]

examples_act = [
"""Question: What type of information architecture pattern is shown in the image?
Action 1: Analyze[layout structure]
Observation 1: The interface shows a nested hierarchy with main categories and subcategories organized in a tree-like structure.
Action 2: Explain[navigation pattern]
Observation 2: Users can navigate through different levels of content using expandable/collapsible sections.
Action 3: Finish[hierarchical navigation pattern]""",

"""Question: How does the color scheme contribute to the user experience?
Action 1: Analyze[color palette]
Observation 1: The interface uses a combination of primary colors for main actions and muted tones for secondary elements.
Action 2: Explain[color hierarchy]
Observation 2: The color scheme helps establish visual hierarchy and guides user attention to important elements.
Action 3: Finish[visual hierarchy and attention guidance]"""]

examples_react = [
"""Question: What type of information architecture pattern is shown in the image?
Thought 1: I need to analyze the overall layout and structure of the interface first.
Action 1: Analyze[layout structure]
Observation 1: The interface shows a nested hierarchy with main categories and subcategories organized in a tree-like structure.
Thought 2: I should examine how users navigate through this structure.
Action 2: Explain[navigation pattern]
Observation 2: Users can navigate through different levels of content using expandable/collapsible sections.
Thought 3: Based on the layout and navigation, this is clearly a hierarchical navigation pattern.
Action 3: Finish[hierarchical navigation pattern]""",

"""Question: How does the color scheme contribute to the user experience?
Thought 1: I should first analyze the colors used in the interface.
Action 1: Analyze[color palette]
Observation 1: The interface uses a combination of primary colors for main actions and muted tones for secondary elements.
Thought 2: Now I need to understand how these colors affect the interface organization.
Action 2: Explain[color hierarchy]
Observation 2: The color scheme helps establish visual hierarchy and guides user attention to important elements.
Thought 3: The color scheme primarily serves to create visual hierarchy and guide attention.
Action 3: Finish[visual hierarchy and attention guidance]"""]

examples_evaluate = [
"""Question: What type of information architecture pattern is shown in the image?
Thought 1: I need to analyze the overall layout and structure of the interface first.
Action 1: Analyze[layout structure]
Observation 1: The interface shows a nested hierarchy with main categories and subcategories organized in a tree-like structure.

Evaluation:
The trajectory is correct as it starts with analyzing the fundamental layout structure, which is essential for identifying the information architecture pattern. The observation accurately describes the hierarchical nature of the interface.
Thus the correctness score is 10""",

"""Question: How does the color scheme contribute to the user experience?
Thought 1: I should first analyze the colors used in the interface.
Action 1: Analyze[color palette]
Observation 1: The interface uses a combination of primary colors for main actions and muted tones for secondary elements.

Evaluation:
The trajectory is correct as it begins with a systematic analysis of the color palette, which is fundamental to understanding its impact on user experience. The observation provides specific details about the color usage and its purpose.
Thus the correctness score is 10"""]