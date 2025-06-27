# Prompts used for our experiments

**Tasks**
- [Game of 24](#game-of-24)
- [HotpotQA](#hotpotqa)
- [HumanEval](#humaneval)
- [SciBench](#scibench)
- [Sonnet Writing](#sonnet-writing)

## Game of 24
```python
# Updated
act = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next step:
14 + 2 = 16 (left: 8 8 16)

Example: 1 4 6
Possible next step:
1 * 4 = 4 (left: 4 6)

Example: 1 3
Possible next step:
1 * 3 = 3 (left: 3)

Input: {input}
Possible next step:
'''

bfs = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.  Follow the format of the following examples. Do not explain simply list possible next steps as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)

Input: {input}
Possible next steps:
'''

aggregate = '''
Please select {n_select_sample} step from the proposed step list, which you believe can reach 24. Each of the proposed steps, uses two of the input numbers to obtain a new number. Do not explain, just list the selected steps as well as all the remaining numbers and nothing else. See the examples for how you are expected to respond.

Things you should consider:
- Do not change at all a step, simply select it.
- You can only select steps from proposed next steps and you cannot change or propose new steps.
- The selected steps should be able to reach 24.
- The selected step must be a valid step, following the format of the possible next steps listed in the examples.

Example: 4 8 8
Number of steps to select: 3
Proposed next steps:
(1) 4 * 8 = 32 (left: 8 32)
(2) 8 * 8 = 64 (left: 4 64)
(3) 4 - 8 = -4 (left: -4 8)
(4) 8 - 8 = 0 (left: 0 4)
(5) 8 / 4 = 2 (left: 2 8)
Selected Next Step Set:
1, 2, 5

Remember, your task is to select {n_select_sample} steps from the proposed next steps. Do not change the steps, just select them. Return only the indexes of the selected steps. Do not include any other information, explanations, comments or conclusions.

Input: {state}
Number of steps to select: {n_select_sample}
Proposed next steps:
{proposal}
Selected Next Step Set:
'''

cot = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Return only the complete answer. If the steps are already given, just return the final expression following the given steps. Do not make any simplifications.

Example: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Example: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Example: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Example: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Example: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {input}
'''

# Updated
evaluate = '''Evaluate if given numbers can reach 24 by responding with the following: "sure", "likely" or "impossible". Follow the format of the following examples. Try to be brief.

Example: 10 14
10 + 14 = 24
sure

Example: 11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible

Example: 4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

Example: 4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

Example: 5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Example: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Example: 10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible

Example: 1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

Input: {input}
'''

# Taken from Tree of Thoughts paper
evaluate_answer = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24. Do not explain simply list the judgement.
Example: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
Example: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
Example: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
Example: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Example: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Example: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {input}
Answer: {answer}
Judge:'''

react = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else. Before providing the next step, provide a short thought on the problem and the existing steps. 

Use the following format:
"Thought: $...$"
Possible next step:
$...$".

Example: 2 8 8 14
Thought: 14 is big, maybe I can subtract something from it to get a number I can multiply up to 24.
Possible next step:
14 - 8 = 6 (left: 2 8 6)

Input: {input}
"""

# RAP prompt for Game24
rap = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Think step by step and plan your moves carefully.

Example: 2 8 8 14
Thought: I need to get to 24. Let me analyze the numbers and plan my moves.
Action: 14 + 2 = 16 (left: 8 8 16)
Thought: Now I have 8, 8, and 16. I can multiply 8 and 8 to get 64, but that's too big. Let me try another approach.
Action: 8 * 8 = 64 (left: 16 64)
Thought: 64 is too big. Let me try a different combination.
Action: 16 - 8 = 8 (left: 8 8)
Thought: Now I have two 8s. I can multiply them to get 64, but that's too big. Let me try another approach.
Action: 8 + 8 = 16 (left: 16)
Thought: I'm stuck. Let me try a different initial move.
Action: 8 / 2 = 4 (left: 4 8 14)
Thought: This looks promising. Now I have 4, 8, and 14.
Action: 14 - 4 = 10 (left: 8 10)
Thought: Now I have 8 and 10. I can multiply them to get 80, but that's too big. Let me try another approach.
Action: 10 - 8 = 2 (left: 2)
Thought: I'm stuck again. Let me try one more initial move.
Action: 2 * 8 = 16 (left: 8 14 16)
Thought: Now I have 8, 14, and 16. I can add 8 and 16 to get 24!
Action: 8 + 16 = 24 (left: 24)

Input: {input}
'''

# Self-evaluation prompts for Game24
self_evaluate_step = '''You are evaluating a reasoning step in the Game of 24. Given the current numbers and the proposed step, determine if this step is correct and logical. Consider:
1. Does the step use valid arithmetic operations?
2. Is the step a logical move towards reaching 24?
3. Does it follow the rules of using exactly two numbers at a time?

Previous steps:
{previous_steps}

Current numbers: {input}
Proposed step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to the Game of 24. Given the input numbers, the steps taken, and the final answer, determine if the solution is correct. Consider:
1. Does it use each input number exactly once?
2. Are all arithmetic operations valid?
3. Does it correctly reach 24?
4. Are the steps logically connected?

Input numbers: {input}

Steps taken:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''
```

## HotpotQA
```python
###################
###---Prompts---###
###################
act = """Solve a question answering task with sequential Action steps. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next action. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

react = """Solve a question answering task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

bfs = """We're solving a question answering task with sequential Action steps. Your task is to propose multiple possible next actions given the current trajectory. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task. When you provide your answer, only state the essential information, without full sentences or explanations.


You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to propose multiple immediate next actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}

Possible Actions:
"""

evaluate = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(2) Lookup[keyword]: ]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(3) Finish[answer]: In this case, your evaluation should be influenced based on whether the answer is correct or not which will be presented in the resulting observation.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.

Below some examples are give.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the latest available thought (if available), action, and observation based on your reasoning analysis. Answer in the format given by the examples and mention nothing more. Make sure to indicate the correctness score at the end of your answer in the following format: "Correctness score : <score>".

Question: {question}
{current_state}

Evaluation:
'''

aggregate = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation and actions that can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question, trajectories and possible actions, select {k} actions that you believe are the best and most relevant to the question. Focus on the latest available action and observation, where you should only select actions from the possible actions. Do not generate additional thoughts or actions. Return only the selected actions in the format given by the examples.

Below some examples are given.

{examples}

(END OF EXAMPLES)

Remember, your task is to select the {k} best actions from the possible actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}
possible actions:
{actions}

Selected actions:
'''

################################
###---Examples for fewshot---###
################################
examples_bfs = [
"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Possible Actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish rock music]
Search[Finish documentaries]
Search[Juice Leskinen]
Search[Documentary film]
""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Possible Actions:
Lookup[named after]
Lookup[Allie Goertz]
Lookup[Matt Groening]
Lookup[name]
Search[Allie Goertz]
Search[The Simpsons]
Search[Allie Goertz Simspons]
""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Possible Actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]
Lookup[Nicholas Ray]
Lookup[profession]
Lookup[producer]""",
]


examples_act = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Action 5: Finish[1,800 to 7,000 ft]""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Action 3: Finish[Richard Nixon]"""

"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Action 3: Finish[The Saimaa Gesture]""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Action 3: Finish[director, screenwriter, actor]""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Action 3: Finish[Arthur's Magazine]""",

"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Action 3: Finish[yes]"""]

examples_react = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]"""

"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]""",

"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]"""]

examples_evaluate = [
"""Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]

Evaluation:
This trajectory is incorrect as my search should be related to Mexican Formula One race car drivers, not winners of the Mexican Grand Prix, a seperate event. A better search would have been for the List of Formula One Grand Prix winners, as suggested.
Thus the correctness score is 3""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]

Evaluation:
This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.
Thus the correctness score is 10""",

"""Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?
Thought 1: I need to search for the football manager who recruited David Beckham, and then find out the timeframe in which he managed Manchester United.
Action 1: Search[David Beckham's football manager]
Observation 1: Could not find David Beckham's football manager. Similar: ['David Beckham', 'Inter Miami CF (redirect from Miami Beckham United)', 'Victoria Beckham', 'The Beckham Experiment', 'Miami Freedom Park'].
Thought 2: I couldn't find the information on David Beckham's football manager. I need to search Manchester United and find out the timeframe of the manager.
Action 2: Search[Manchester United]
Observation 2: Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.. Domestically, Manchester United have won a record 20 league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields.
Thought 3: The information about David Beckham's football manager is not available. I will search for Manchester United's managers to find the timeframe.
Action 3: Search[Manchester United managers]
Observation 3: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thought 4: The information about David Beckham's football manager is not available. The question asks for the timeframe when the manager managed Manchester United, so I need to find a list of Manchester United managers.
Action 4: Search[list of Manchester United managers]
Observation 4: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.

Evaluation:
This trajectory is mostly incorrect as I was not able to retrieve the relevant information to answer the question. The searches are too broad and not relevant to the specific player recruited, David Beckham. The last two actions are also too similar. It would be better to first search for David Beckham and lookup information related to his manager.
Thus the correctness score is 4""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]

Evaluation:
This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search. 
Thus the correctness score is 10"""
]

examples_aggregate = [
"""
Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Possible Actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish rock music]
Search[Finish documentaries]
Search[Juice Leskinen]
Search[Documentary film]

Selected actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish documentaries]
""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Possible Actions:
Lookup[named after]
Lookup[Allie Goertz]
Lookup[Matt Groening]
Lookup[name]
Search[Allie Goertz]
Search[The Simpsons]
Search[Allie Goertz Simspons]

Selected actions:
Lookup[named after]
Lookup[name]
Search[The Simpsons]
""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Possible Actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]
Lookup[Nicholas Ray]
Lookup[profession]
Lookup[producer]

Selected actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]""",
]

self_evaluate_step = '''You are evaluating a reasoning step in a question answering task. Given the current state and the proposed step, determine if this step is correct and logical. Consider:
1. Is the search/lookup action relevant to finding the answer?
2. Is the thought process logical and focused on the question?
3. Does it follow the rules of using Search, Lookup, and Finish actions appropriately?

Current state: {current_state}
Proposed step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a question answering task. Given the question, the steps taken, and the final answer, determine if the solution is correct. Consider:
1. Does it use appropriate search and lookup actions to find relevant information?
2. Are all actions logically connected and relevant to the question?
3. Does it correctly answer the question based on the information found?
4. Are the steps taken efficient and focused?

Question: {question}

Steps taken:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''
```

## HumanEval
```python
SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."
SIMPLE_CHAT_INSTRUCTION_V2 = """You are an AI that only responds with only {lang} code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."""

aggregate_prompt = """You are a programming assistant, who is helping user to write efficient and correct codes. You will be given multiple implementations of the same function. You should choose the {k} best implementation based on the following criterias:
1. Correctness: The implementation should return the correct output.
2. Efficiency: The implementation should be efficient in terms of time and space complexity.
3. Readability: The implementation should be readable and understandable.
4. Style: The implementation should follow the style guide of the language.
5. Testability: The implementation should be testable.

Remember your task is to choose the {k} best implementation based on the above criterias. Make sure to return only the indexes of the selected implementations, separated by commas. Do not include any other explanations, introduction, conclusions or thoughts. Just return the indexes of the selected implementations.
Function signature and docstring:
{prompt}
Implementations:
{implementations}
Chosen implementation:
"""

evaluation_prompt = """You are a programming assistant, who is helping the user to evaluate a generated code. You will be given a single implementation of a function, and you should evaluate it based on the following criteria:

1. **Correctness**: Does the implementation return the correct output for different inputs?
2. **Efficiency**: Is the implementation efficient in terms of time and space complexity?
3. **Readability**: Is the code readable and understandable? Is it easy to follow?
4. **Style**: Does the implementation follow the style guide of the language (naming conventions, indentation, etc.)?
5. **Testability**: Is the implementation testable? Can it be easily tested with unit tests?

Evaluate the code on each criterion with a score from 1 to 10 (integers only, no fractions). Then give an overall score as the sum of all scores.

Function signature and docstring:
{prompt}

Implementation:
{implementation}

Evaluation scores:
- Correctness: <score>
- Efficiency: <score>
- Readability: <score>
- Style: <score>
- Testability: <score>

Overall Score: <final score>

Do not include any further thoughts or reasoning, just the evaluation scores and the final overall score."""


SIMPLE_CHAT_INSTRUCTION_BFS = """
You are an AI that only responds with {lang} code. You will be given a function signature and its docstring by the user.
Write multiple full implementations (at least two), each restating the function signature. Use a different approach for each.
Mark the start and end of each implementation using triple backticks, like this:
\`\`\`
<Code implementation here>
\`\`\`
Each implementation should be fully contained within its own set of backticks, without any additional markers.
"""

react = """You are a programming assistant solving a coding task. Think step by step and plan your implementation carefully. You will be given a function signature and docstring, and you need to implement the function.

For each step:
1. Think about what needs to be done
2. Write code to implement that step
3. Consider edge cases and error handling

Example:
Function signature and docstring:
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''

Thought: I need to implement a simple addition function. The function takes two integers and returns their sum. I should handle basic input validation.

Action: python
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''
    # Input validation
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b


Thought: The implementation looks good. It includes:
1. Type hints for parameters and return value
2. Input validation to ensure both arguments are integers
3. Simple and efficient addition operation
4. Proper docstring preservation

Action: Finish[The implementation is complete and correct]

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Remember to think step by step and write clear, efficient code."""

self_evaluate_step = '''You are evaluating a reasoning step in a code generation task. Given the function signature, current implementation, and the proposed step, determine if this step is correct and logical. Consider:
1. Is the code syntactically correct?
2. Does it follow the function's requirements?
3. Is it a logical next step in the implementation?
4. Does it handle edge cases appropriately?

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Proposed step:
{step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a code generation task. Given the function signature, the implementation steps, and the final code, determine if the solution is correct. Consider:
1. Does the implementation match the function signature and docstring?
2. Is the code syntactically correct and follows language style guidelines?
3. Does it handle all edge cases and error conditions?
4. Is it efficient and readable?
5. Does it include appropriate tests or validation?

Function signature and docstring:
{prompt}

Implementation steps:
{steps}

Final code:
{answer}

Is this solution correct? Answer with a single word: Yes or No.
'''
```

## SciBench
```python
###################
###---Prompts---###
###################

io = '''
Given a science problem, your task is to answer the question step-by-step in a clear and specific manner.
The format of the solution is limited to: "Solution: ...\nSummary: The final answer is $...$"
Please complete the answer step-by-step, and finally outline the final answer.
Problem: {problem}
Solution:'''


react = '''Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps.

If the input does not provide any existing steps, you need to analyze the problem and then give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete reasoning step, which should include detailed calculations, reasoning, choosing answers, etc.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question. 

Before providing the next step, provide a short thought on the problem and the existing steps. Use the following format:
"Thought: $...$"
Next step: $...$".

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

# # ReST-MCTS*
act = '''Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps. 

If the input does not provide any existing steps, you need give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete step, which may include detailed calculations, reasoning, choosing answers, etc. but no reasoning.

If the existing steps are already sufficient, you can output "The final answer is: $...$" where ... indicates the final answer to the question. 

Below is the input, please follow the specified format for your output.

Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

aggregate = '''Given a science proplem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should choose from the given steps, which best helps you get towards a solution to the question.

From the partial or fully solutions, your task is to select {k} partial or full solutions that best solves or calculates the problem. Your output must be the numbers of the selected partial or full solutions, without any explanation, reasoning, introduction, conclusion or modifucations.

Below is the input, please only output the {k} indexes of your choices.

Problem: {problem}
Solutions:
{steps}
Output:'''



bfs = '''Given a science problem, you need to answer the problem based on your existing knowledge. The input may include some existing steps to solve the question and you should continue to complete the solution based on these existing steps. 

If the input does not provide any existing steps, you need give the first step in solving or calculating the problem. If partial solution steps are provided, you need to output the next step along the lines of the existing steps.
The output format is limited to: "Next step: ..." where ... indicates omitted output information, which is the next step in the answer that you should give. Your output must be a complete step, which may include detailed calculations, reasoning, choosing answers, etc. but no reasoning.

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

# Summary prompt (ReST-MCTS*)
summary = '''
Given a math problem and its corresponding solution, your task is to extract the final answer obtained in the solution.
You should summarize the answer using the format: "The final answer is $...$". Replace "..." with the answer obtained in the solution.
Problem: {problem}
Solution: {existing_steps}
Extracted answer:'''

# ReST-MCTS* (Translated using google translate)
evaluate = '''Your task is to assess whether the provided solution steps can successfully solve the given science/mathematics problem and output a score.
The score should be a decimal between 0 and 1. If all the provided steps are incorrect (every step is wrong), the score should be 0. If all steps are correct and the final answer is successfully calculated, the score should be 1. The more errors there are in the steps, the closer the score should be to 0. The closer the steps are to the final correct answer, the closer the score should be to 1.
Steps that only contain verbal descriptions without any mathematical expressions should generally receive a low score. A score equal to or greater than 0.9 can only be given if the answer has already been calculated to a specific numerical value. If the thought process is complete but the answer is not computed, or only the mathematical expression is written without solving it, the score must be below 0.9.

First provide an analysis, then the score. Your analysis and scoring should be entirely based on the given steps. Do not continue solving the problem. Please study the following examples.

{examples}

Below is a problem and the existing steps, with analysis and scoring. Be careful not to output the next steps in the analysis, and the scoring should be based entirely on the steps given in the input.
The output format is limited to: "Analysis:...\nScore:...", where ... indicates omitted output content, which is the part you need to fill in.

Input:
Problem: {problem}
Existing steps:
{existing_steps}
Output:'''

self_evaluate_step = '''You are evaluating a step in a scientific procedure. Given the task requirements and the current step, determine if this step is correct and contributes meaningfully to the solution. Consider:
1. Is the step scientifically valid and logically sound?
2. Does it follow from the previous steps?
3. Does it help progress toward solving the task?
4. Does it avoid major scientific misconceptions?

Task: {input}

Previous steps:
{previous_steps}

Current step: {step}

Is this step correct and well-reasoned? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete scientific solution. Given the task requirements and the full answer, determine if it meets all criteria. Consider:
1. Is the answer scientifically accurate?
2. Does it clearly and correctly address the main question?
3. Are all reasoning steps logically consistent and well-justified?
4. Does it use appropriate scientific terminology and avoid misconceptions?
5. Is it complete and concise?

Task: {input}

Answer:
{answer}

Is this scientific answer correct and well-reasoned? Answer with a single word: Yes or No.
'''

################################
###---Examples for fewshot---###
################################
examples_evaluate = [
"""Question: Discuss for what value of p, the generalized integral $\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$ converges.
Existing steps:
Step 1: To illustrate the convergence of the integral, consider splitting the integral into two parts: $$ \\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2} dx = \\int_0^1 \\frac{x^p \\ln x}{(1+x^2)^2} dx + \\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2} dx $$
Step 2: For the first part, $0 \\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$, so it converges if and only if $p>-2$.
Output:
Analysis: Step 1 correctly gets the idea of ​​splitting the integral, but the derivation of step 2 is wrong, and there are problems in judging the convergence. $0 \\leq \\frac{x^p \\ln x}{(1+x^2)^2} \\leq x^p$, according to \\int_0^1 x^p dx converges if and only if $p>-1$, so the original integral converges if and only if $p>-1$, not $p>-2$.
Score: 0.1""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider taking the derivative: We can regard the sequence $n^{1/n}$ as a function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By taking the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Output:
Analysis: Step 1 of the existing steps is correct. It establishes the basic idea of ​​solving the problem, that is, treating the sequence as a function and analyzing the increase and decrease of the function by taking the derivative. However, this is only part of the solution. Further steps are required to find the positive integer $n$ corresponding to the maximum value and to obtain the maximum value. Therefore, the existing steps have not yet inferred the answer.
Score: 0.2""",

"""Question: Find the average value of the function $f(x)=1+x^2$ on the interval $[-1,2]$.
Existing steps:
Step 1: Use definite integral to solve the average value: We can use definite integral to solve the average value of the function on the interval $[-1,2]$.
Step 2: First, we need to calculate the definite integral $\\int_{-1}^{2} (1+x^2) dx=6$.
Step 3: Then, we can use the properties of the definite integral to divide the result of the definite integral by the length of the interval, that is, $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}$, which should be the average value of the function on the interval.
Step 4: Calculate the above formula and get the result of $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=\\frac{6}{3}=2$, so the average value of the function is 2.
Output:
Analysis: All steps are derived correctly, and the existing steps have calculated the answer to $2$, which can get a full score of 1.
Score: 1""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider the derivative: We can regard the sequence $n^{1/n}$ as the function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By finding the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Step 2: Based on the idea of ​​the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$, and then find the derivative of g(x).
Step 3: We perform the derivation, $$\\frac{d}{dx}\\left(\\ln(f(x))\\right) = -\\frac{1}{x^2} \\ln(x) - \\frac{1}{x^2} + \\frac{1}{x^2} \\ln(x) = -\\frac{1}{x^2}$$. This derivative is always negative, indicating that $f(x)$ is decreasing within the range of positive integers. ,
Output:
Analysis: The first two steps correctly analyzed the idea of ​​​​derivation, but the specific derivation process in step 3 was wrong. The correct procedure for the derivation is: $$\\frac{d}{dx}\\left(\\ln(f(x))\\right) = -\\frac{1}{x^2} \\ln(x) + \\frac{1}{x^2}$$, not $-\\frac{1}{x^2}$.
Score: 0.2""",

"""Question: Find the average value of the function $f(x)=1+x^2$ on the interval $[-1,2]$.
Existing steps:
Step 1: Consider the value of the function at the endpoints of the interval: We can calculate the value of the function at the endpoints of the interval $x=-1$ and $x=2$, that is, $f(-1)=1+(-1)^2=2$ and $f(2)=1+2^2=5$.
Step 2: Then we can calculate the average value of the function at these two endpoints, that is, $\\frac{2+5}{2}=3.5$. This is the average value of the function on the interval $[-1,2]$.
Output:
Analysis: All derivation steps are wrong, and 0 points should be given. The average value of a function on an interval should be equal to the integral of the function on the interval divided by the length of the interval, that is, $\\frac{\\int_{-1}^{2} (1+x^2) dx}{3}=2$. It cannot be simply considered to be equal to the average value of the function value at the endpoints of the interval.
Score: 0""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider taking the derivative: We can regard the sequence $n^{1/n}$ as the function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By taking the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Step 2: Based on the idea of ​​the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get the derivative of $g(x)$ to be $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
Output:
Analysis: All the steps are derived correctly, but the value of the maximum term has not been specifically calculated, that is, the answer has not been calculated. It is also necessary to analyze the positive and negative nature of the derivative to understand the increase and decrease of $f(x)$.
Score: 0.6""",

"""Question: Discuss for what value of p, the generalized integral $\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx$ converges.
Existing steps:
Step 1: Let $J=\\int_0^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_1=\\int_0^{1} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, $J_2=\\int_1^{+\\infty} \\frac{x^p \\ln x}{(1+x^2)^2}dx $, then the generalized integral $J$ converges if and only if $J_1, J_2$ both converge.
Step 2: When $x \\rightarrow 0^+$, $\\frac{x^p \\ln x}{(1+x^2)^2} \\sim x^p \\ln x$, so $J_1$ converges if and only if $p > -1$.
Step 3: When $x \\rightarrow +\\infty$, $\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$, so $J_2$ converges if and only if $p < 4$.
Output:
Analysis: The first two steps are correct, but the deduction of step 3 is wrong. When $x \\rightarrow +\\infty$, $\\frac{x^p \\ln x}{(1+x^2)^2} \\sim \\frac{\\ln x}{x^{4-p}}$, according to \\int_0^{+\\infty} x^m dx converges if and only if $m<-1$, so the original integral converges if and only if $p-4 < -1$, that is, $p < 3$, not $p < 4$.
Score: 0.2""",

"""Question: Find the maximum value of the function $f(x)=-\\frac{1}{2}*(x^2)+2*x-1$ on R.
Existing steps:
Step 1: Find the derivative: We can find the derivative $f'(x)$ of the function $f(x)$, that is, $f'(x)=-x+2$. By taking the derivative, we can find the increase or decrease of the function, and then determine the $x$ value corresponding to the maximum value of the function on R.
Step 2: We can calculate the value of $f'(x)$ when $x=1$, that is, $f'(1)=1$. From this, we can see that at $x=1$, the function $f(x)$ reaches its maximum value, that is, its maximum value.
Output:
Analysis: The first step is correct, but the second step is wrong. Calculating the value of $f'(x)$ at $x=1$ does not tell us the increase or decrease of the function as a whole, and is meaningless. From $f'(1)=1$, it cannot be deduced that the function $f(x)$ reaches its maximum value at $x=1$, and the maximum value should satisfy the derivative of 0.
Score: 0.1""",

"""Question: Find the value of the largest term of the sequence ${n^{1/n}}$ (n=1, 2, 3... are positive integers).
Existing steps:
Step 1: Consider the derivative: We can regard the sequence $n^{1/n}$ as the function $f(x) = x^{1/x}$, and then find the derivative $f'(x)$ of the function. By finding the derivative, we can find the increase and decrease of the function, and then determine the positive integer $n$ value corresponding to the maximum value of the sequence.
Step 2: Based on the idea of ​​the previous step, for the function $f(x) = x^{1/x}$, we can take the natural logarithm to simplify the derivation process, and get $g(x)=\\ln(f(x)) = \\frac{1}{x}\\ln(x)$, and then find the derivative of g(x).
Step 3: After calculation, we get the derivative of $g(x)$ is $$-\\frac{1}{x^2}\\ln(x) + \\frac{1}{x^2}$$
Step 4: Next, we can analyze the positive and negative value of the derivative. This derivative is negative when $x > e$ and positive when $x < e$. This means that the function $f(n)$ is decreasing when $n > e$ and increasing when $n < e$.
Output:
Analysis: All the existing steps are correctly derived, and the function's increase and decrease are analyzed, but the value of the maximum term has not been specifically calculated, that is, the answer has not been calculated, so a score greater than or equal to 0.9 cannot be given. However, since the existing steps are very close to calculating the answer, the score should be close to 0.9.
Score: 0.8""",

"""Question: Find the area of ​​the figure enclosed by the function $f(x)=x+1$ and the straight lines $x=0$, $x=1$ and the x-axis.
Existing steps:
Step 1: According to the geometric meaning of definite integrals, solving the definite integral of the function is the area of ​​the required figure, and the calculation result can be directly used as the final answer.
Output:
Analysis: The analysis in step 1 is correct, but the expression is vague and it is of little help in solving the problem, and the answer is not actually calculated, so only a small score can be given. A more appropriate statement is: According to the geometric meaning of the definite integral, the area to be sought should be the definite integral of $f(x)=x+1$ on the interval $[0,1]$.
Score: 0.1"""
]
```

## Sonnet Writing
```python
act = '''You are fluent in sonnet writing and can only respond with your sonnet and nothing else.
Below, you will be given your task and keywords to include in your sonnet writing. Remember to return only your sonnet writing.

{input}'''

aggregate = '''There have been created some sonnet writings with respect to the following task:
{task}

Select the {k} best sonnet writings with respect to the task, without any modification to the sonnet writings. Do not add any explanation, comments, introduction or conclusion, your should only return the sonnets writings.
You select a sonnet writing by returning the number of that sonnet writing.

{examples}

(End of examples)

Remember, your task is to select the {k} best sonnet writings. Do not add any explanation, comments, introduction, conclusions or modifications.
You should only return the numbers of the selected sonnet writings.

{sonnets}'''

evaluate = '''You are given a sonnet writing below. Your task is to evaluete the sonnet based on the criterias:
    1. Follows the rhyme scheme: {rhyme_scheme}
    2. Contains these exact words: {words}

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the given sonnet between 0 and 10. Do not add any explanations, comments, introduction or conclusions.

{sonnet}
'''

examples_evaluate = [
    '''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

The river bends beside the morning grass (A)  
A shimmer dances on the silver tide (B)  
The hours like golden moments swiftly pass (A)  
And fortune's value flows with gentle pride (B)  

The jail of fear breaks open with the breeze (C)  
The open fields forgive the bitter rain (D)  
The grass revives beneath the waking trees (C)  
And songs of hope replace the cries of pain (D)  

Bright grasslands whisper secrets to the skies (E)  
The jail of winter thaws beneath the light (F)  
The value found in spring will never die (E)  
But bloom in endless fields beyond our sight (F)  

The grass will grow where broken dreams once fall (G)  
And value shines within the hearts of all. (G)

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

The mountain sighs beneath the winter snow
The river carves its path with solemn might
The flowers sleep beneath the frozen glow
Awaiting touch of spring’s returning light

No jail confines the wild and restless breeze
It tumbles past the valley’s sleeping door
The grass will rise when winter grants release
Yet value fades along the rocky shore

The colors blend beneath a paling sky
The stars retreat behind a drifting veil
Though grass returns, some dreams must say goodbye
And hearts once brave grow weary and grow pale

The seasons turn, and leave us wondering still
What value clings to hope, and what to will.

---END-OF-SONNET---

evaluation: 10 
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

Upon the cliffs the storm begins to roar
The candles flicker in the empty hall
The silver mist creeps underneath the door
A solemn hush descends upon the wall

No grassy fields are near this barren land
No value glimmers in the heavy rain
No jail can hold the fury in its hand
The waves collapse against the rocks in vain

The sailors cry into the endless dark
No flame to guide them through the cruel, wild gale
No harvest waits beyond the fading spark
Just shattered hopes imprisoned without bail

Their names are lost beneath the mourning wave
No grass, no value, only storms to brave.

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

The twilight burns across the shattered shore (A)  
Old lanterns flicker in the dying mist (B)  
The crows descend where empty houses mourn (C)  
A bitter memory clenched within a fist (B)  

The river moans beneath a sky of ash (D)  
Its course forgotten by the sleeping stone (E)  
The mountains tremble under thunder’s crash (D)  
The plains are silent, aching and alone (F)  

No voices rise above the broken field (G)  
No banners fly beneath the heavy rain (H)  
The meadow bows beneath the turning wheel (I)  
While dreams decay and vanish in their pain (H)  

Night falls without a sound, a hollow art (J)  
And leaves the world with one forsaken heart. (J)

---END-OF-SONNET---

evaluation: 0
'''
]

aggregate_examples = [
    '''Beneath the whisper of the waving grass (A)
The river sings a song of ancient lore (B)
Each golden moment holds a priceless mass (A)
Of value set beyond the richest store. (B)

The sunlight weaves a pattern soft and bright (C)
Across the meadows where the daisies sail, (D)
Yet even beauty feels a distant blight (C)
When love is locked within a silent jail. (D)

O time, who takes but seldom grants us grace, (E)
Within your tides do all our dreams entwine, (F)
Yet still the grass shall whisper of this place, (E)
A testament to moments lost in time. (F)

So hold the value of each fleeting breath, (G)
For life escapes more swiftly than its death. (G)

---END-OF-SONNET---

The jail of sorrow keeps my heart in thrall (A)
While grass grows wild beyond the broken gate. (B)
I find no value in these chains at all, (A)
Yet still I bear the burden of my fate. (B)

A lonesome wind creeps through the ruined hall (C)
And sings of dreams that once could not prevail, (D)
Now dust and silence answer every call, (C)
Their memories locked in an endless jail. (D)

But hope, like grass, persists beyond the stone (E)
And value shines where shadows used to tread; (F)
A broken heart can yet become its own, (E)
A phoenix rising from the dreams long dead. (F)

O jail, O grass, O value lost and won, (G)
Your cycle ends anew with each day's sun. (G)

---END-OF-SONNET---

I saw a meadow bright with golden flame (A)
And heard the jailer laughing in his cell; (B)
The value of the fields he could not name, (A)
For grass and sunlight meant no tales to tell. (B)

The sky was pale and heavy with the rain, (C)
While iron gates were rusted in the gale; (D)
A robin sang a melancholy strain (C)
Outside the damp and sorrow-scented jail. (D)

The grass grew thick and wove a velvet sheet (E)
Where value lay in every drop of dew, (F)
And in the mud the jailer dragged his feet, (E)
Unknowing that the grass would soon renew. (F)

The jail will fall, the grass will rise instead, (G)
For nature reclaims even iron and dread. (G)

---END-OF-SONNET---

Selected 2 Best sonnets:
1
2''',
'''Upon the hill where once the grass was green
The winds now whisper secrets of the vale,
A place where none but memories have been
And sorrow weeps behind a broken jail.

The stars above reflect in pools below
While tender dreams dissolve without a trace,
Yet hope still carves its message in the snow
And leaves the value written on its face.

The seasons turn with neither grief nor song
And grass will rise to meet the morning tide,
The heart must travel even when it's wrong,
And seek the dreams that time cannot deride.

We hold the broken past within our hands,
And plant new hopes upon the barren lands.

---END-OF-SONNET---

In fields of gold where bending grasses sway
I found a road that led beyond the stream,
A shining path that twisted night to day,
And led me farther from the jail of dream.

The value of a single step was small
Yet each one built a bridge of lasting stone;
The stars above seemed waiting for the call
To guide me far from everything I'd known.

The blades of grass were wet with morning's grace
And songs of sparrows filled the misty air;
They sang of freedom no jail could replace
Of journeys shaped by hope beyond despair.

I walked the road and felt the future start,
The grass and stars still singing in my heart.

---END-OF-SONNET---

The jail was hidden deep beneath the trees (A)
Where no soft grass could ever hope to grow. (B)
The shadows whispered warnings on the breeze, (A)
A chilling song the heart would never know. (B)

No value marked the time that passed away, (C)
The walls were cold and silent in their might, (D)
And grassless stones forever bleak and gray (C)
Reflected only sorrow in their sight. (D)

Yet through the cracks a single flower rose (E)
Defiant in the darkness and the pain, (F)
Its fragile beauty trembling as it grows, (E)
A symbol that the spirit will remain. (F)

Where grass may fail, and even value die, (G)
Still hope will blossom if it dares to try. (G)

---END-OF-SONNET---

Selected 2 best sonnets:
2
3'''
]

# React prompt for SonnetWriting
react = '''You are fluent in sonnet writing. You will be given a task that specifies the rhyme scheme and words that must be included. Your task is to write a sonnet that follows these requirements while maintaining poetic quality.

You should think step by step about the logical reasoning needed to write the sonnet:

1. First, analyze the rhyme scheme and required words
2. Then, plan how to incorporate the required words naturally
3. Next, develop a theme that connects these words meaningfully
4. Finally, write the sonnet following the required format

(Example)
Task: Write a sonnet with rhyme scheme ABAB CDCD EFEF GG and include the words: grass, value, jail

Thought: Let me analyze the requirements step by step:
1. Need to follow ABAB CDCD EFEF GG rhyme scheme
2. Must include "grass", "value", and "jail" naturally
3. Should create a theme connecting these words
4. Need to write in iambic pentameter

Thought: Let me plan the sonnet structure:
1. First quatrain: Introduce "grass" and set the scene
2. Second quatrain: Incorporate "jail" and develop theme
3. Third quatrain: Use "value" and reinforce theme
4. Couplet: Conclude with a meaningful message

Thought: Now I'll write the sonnet following this plan:

The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---
The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---END-OF-SONNET---

(End of example)

Remember, your task is to think step by step and write a sonnet that follows the required rhyme scheme and includes all specified words naturally. Do not add any explanation, comments, introduction or conclusion, you shall only return your sonnet followed by ---END-OF-SONNET---.

Task: {input}

Current reasoning:
{current_state}

---
'''

# Self-evaluation prompts for SonnetWriting
self_evaluate_step = '''You are evaluating a step in sonnet writing. Given the task requirements and the current step, determine if this step is correct and contributes to a good sonnet. Consider:
1. Does it follow the required rhyme scheme?
2. Does it maintain iambic pentameter?
3. Does it contribute to the overall theme?
4. Does it use the required words naturally?

Task: {input}

Previous steps:
{previous_steps}

Current step: {step}

Is this step correct and well-written? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete sonnet. Given the task requirements and the sonnet, determine if it meets all criteria. Consider:
1. Does it follow the required rhyme scheme?
2. Does it include all required words naturally?
3. Does it maintain iambic pentameter throughout?
4. Does it have a coherent theme and imagery?
5. Is it grammatically correct and poetic?

Task: {input}

Sonnet:
{sonnet}

Is this sonnet correct and well-written? Answer with a single word: Yes or No.
'''
```
