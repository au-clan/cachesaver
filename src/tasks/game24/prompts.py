# Updated
act = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next step:
14 + 2 = 16 (left: 8 8 16)

Example: 1 4 6
Possible next steps:
1 * 4 = 4 (left: 6 4)

Example: 1 3
Possible next step:
1 * 3 = 3 (left: 3)

Input: {input}
Possible next step:
'''

react = """
Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else.
Solve a task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be this type:

(1) Make24[a, b], which return how to generate 24 from a and b if possible
(2) Try[a, b], randomly try a and b to get result
(3) Finish

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

Example: 1 3
Thought 1: only two numbers left, I have to use these two.
Action 1: Make24[1, 3]
Observation 1: no way to get 24 from 1 and 3
Possible next step:
1 * 3 = 3 (left: 3)

Example: 1 4 6
Thought 1: there are three numbers, I can try any two of them.
Action 1: Try[4, 6]
Observation 1: 4 * 6 = 24
Possible next step:
4 * 6 = 24 (left 1 24)

Example: 2 8 8 14
Thought 1: there are four numbers, I can try any two of them which is not tried in previous Thought.
Action 1: Try[2, 14]
Observation 1: 2 + 14 = 16
Possible next step:
2 + 14 = 16 (left 8 8 16)

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Input: {input}
Possible next steps:"""


bfs = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list possible next steps as well as all the remaining numbers and nothing else.

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
Please select {n_select_sample} step from the proposed step list, which you believe can reach 24. Do not explain, just list the selected steps as well as all the remaining numbers and nothing else. See the examples for how you are expected to respond.
 
Example: 2 8 8 14
Proposed next steps:
(1) 8 / 2 = 4 (left: 4 8 14)
(2) 14 + 2 = 16 (left: 8 8 16)
Best Next Step:
(1) 8 / 2 = 4 (left: 4 8 14)

Example: 4 8 8
Proposed next steps:
(1) 8 * 8 = 64 (left: 4 64)
(2) 4 * 8 = 32 (left: 8 32)
Best Next Step:
(2) 4 * 8 = 32 (left: 8 32)

Example: 4 8 8
Proposed next steps:
(1) 4 * 8 = 32 (left: 8 32)
(2) 8 * 8 = 64 (left: 4 64)
(3) 4 - 8 = -4 (left: -4 8)
(4) 8 - 8 = 0 (left: 0 4)
(5) 8 / 4 = 2 (left: 2 8)
Best Next Step Set:
(1) 4 * 8 = 32 (left: 8 32)
(2) 8 * 8 = 64 (left: 4 64)
(5) 8 / 4 = 2 (left: 2 8)

Input: {state}
Proposed next steps:
{proposal}
Best Next Step:
'''

cot = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Return only the complete answer.

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