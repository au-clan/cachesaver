###################
###---Prompts---###
###################

io = """You will be given a question. Simply provide the final answer. Do not provide any explanations or intermediate steps.  The format to respond is "Final Answer: ..."""

cot = """You will be given a question. Think step by step and provide your reasoning before giving the final answer. The format to respond is "Final Answer: ...", where "..." is the final answer."""


act = '''Given a problem, you need to answer based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps. 

If the input does not provide any existing steps, you need to give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output only the next step along the lines of the existing steps.

The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a single complete step, which may include detailed calculations, one node of reasoning (eg. a sentence), choosing answers, etc.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question. 

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

bfs = '''Given a problem, you need to answer on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps. 

If the input does not provide any existing steps, you need give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.

The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a single complete step, which may include detailed calculations, one node of reasoning (eg. a sentence), choosing answers, etc.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question. 

Please provide MULTIPLE alternative next steps. Use the following format:
"Next step: $...$
Next step: $...$
Next step: $...$".

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

aggregate = '''Given a  proplem, you need to answer based on your existing knowledge. The input may include some existing steps to solve the question and you should choose from the given steps, which best helps you get towards a solution to the question.

From the partial or fully solutions, your task is to select {k} partial or full solutions that best solves or calculates the problem. Your output must be the numbers of the selected partial or full solutions, without any explanation, reasoning, introduction, conclusion or modifucations.

Below is the input, please only output the {k} indexes of your choices.

Problem: {problem}
Solutions:
{steps}
Output:'''

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

summary = '''
Given a math problem and its corresponding solution, your task is to extract the final answer obtained in the solution.
You should summarize the answer using the format: "The final answer is $...$". Replace "..." with the answer obtained in the solution.
Problem: {problem}
Solution: {existing_steps}
Extracted answer:'''

evaluate = '''Your task is to assess whether the provided solution steps can successfully solve the given science/mathematics problem and output a score.
The score should be a decimal between 0 and 1. If all the provided steps are incorrect (every step is wrong), the score should be 0. If all steps are correct and the final answer is successfully calculated, the score should be 1. The more errors there are in the steps, the closer the score should be to 0. The closer the steps are to the final correct answer, the closer the score should be to 1.
A score equal to or greater than 0.9 can only be given if the answer has already been calculated to a specific value. If the thought process is complete but the answer is not computed, or only the mathematical expression is written without solving it, the score must be below 0.9.

First provide an analysis, then the score. Your analysis and scoring should be entirely based on the given steps. Do not continue solving the problem.

Below is a problem and the existing steps, with analysis and scoring. Be careful not to output the next steps in the analysis, and the scoring should be based entirely on the steps given in the input.
The output format is limited to: "Analysis:...\nScore:...", where ... indicates omitted output content, which is the part you need to fill in.

Input:
Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

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