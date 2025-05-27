act = '''You are participating in a multiple-choice quiz. You will be given a paragraph, which  contains the information needed to answer the question.
After the paragraph you will be given your question together with the four choices.

(Example)

In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A.Civic Park is north of the administrative service area
B.The leisure area is southwest of the cultural area
C.The cultural district is in the northeast of the business district
D.The business district is southeast of the leisure area

---
Answer: a

(End of example)

Remember, your task is to select only one choise, which you think answers the question with respect to the paragraph given. Do not add any explanation, comments, introduction or conclusion, you shall only return your answer.

{paragraph}

{question}
{choises}'''

aggregate = '''You are participating in a multiple-choice quiz. You will be given a paragraph, which contains the information needed to answer the question.
After the paragraph you will be given your question together with the four choices.

You will be given possible answers to the given quiz, and it is your task to select the {k} best choices.

(Example)
In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A.Civic Park is north of the administrative service area
B.The leisure area is southwest of the cultural area
C.The cultural district is in the northeast of the business district
D.The business district is southeast of the leisure area

---
Possible answers:
(1) Answer: a
(2) Answer: a
(3) Answer: b
(4) Answer: c
(5) Answer: a

Selected answers:
(1) Answer: a
(2) Answer: a
(5) Answer: a

(End of example)

Remember, your task is to select exactly {k} answers, which you think is correct with respect to the question and paragraph. Do not add any explanation, comments, introduction, conclusion, or modification, you shall only return the indexes of your {k} selected answers.

{paragraph}

{question}
{choices}

---
Possible answers:
{actions}

Selected answers:'''

evaluate = '''You are the judge of a multiple-choice quiz. You will be given the paragraph, which contains the information needed to answer the question.
After the paragraph you will be given the question together with choices and a proposed answer. It is then your task to judge the answer by correctness ("incorrect", "plausible", or "correct").

{examples}

(End of examples)

Remember, your task is to judge the chosen answer by its correctness ("incorrect", "plausible", or "correct"). Do not add any explanation, comments, introduction or conclusion, you shall only return your judgement.

{paragraph}

{question}
{choices}
---
Answer: {answer}
---'''

evaluate_examples = [
    '''In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park.These four communities are designated as cultural area, leisure area, commercial area and administrative service area.It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A.Civic Park is north of the administrative service area
B.The leisure area is southwest of the cultural area
C.The cultural district is in the northeast of the business district
D.The business district is southeast of the leisure area
---
Answer: a
---
correct
''',
'''In recent years, graduate entrance examinations have continued to heat up.Correspondingly, a variety of postgraduate counseling classes have emerged, especially English and political counseling classes are almost a must for the postgraduates.Xiaozhuang, who has just started working, also intends to take the postgraduate entrance exam, so Xiaozhuang must take English tutoring classes
Which of the following can best strengthen the above argument
A.If you take an English tutoring class, you can pass the graduate entrance exam
B.Only those who intend to take the graduate entrance exam will participate in the English tutoring class
C.Even if you take an English tutoring class, you may not be able to pass the graduate entrance exam
D.If you do not participate in the English tutoring class, you cannot pass the graduate entrance exam
---
Answer: D
---
correct''',
'''Compared with small and medium-sized cities, especially small cities and towns, large cities have higher living costs, which inevitably limits the entry of rural population.Therefore, the development of large cities alone cannot actually achieve urbanization
Which of the following is the conclusion must be assumed
A.Urbanization is the only way for China's development
B.Simple development of large cities is not conducive to the promotion of urbanization
C.To achieve urbanization, the city must fully absorb the rural population
D.The attractiveness of large cities to the rural population in the outside world is significantly lower than that of small and medium-sized cities
---
Answer: a
---
incorrect'''
]

# React prompt for LogiQA
react = '''You are participating in a multiple-choice quiz. You will be given a paragraph, which contains the information needed to answer the question.
After the paragraph you will be given your question together with the four choices.

You should think step by step about the logical reasoning needed to answer the question:

1. First, analyze what information is given in the paragraph
2. Then, understand how this information relates to the question
3. Next, evaluate each choice against the given information
4. Finally, make a logical conclusion about which choice is correct

(Example)

In the planning of a new district in a township, it was decided to build a special community in the southeast, northwest, centered on the citizen park. These four communities are designated as cultural area, leisure area, commercial area and administrative service area. It is known that the administrative service area is southwest of the cultural area, and the cultural area is southeast of the leisure area.

Based on the above statement, which of the following can be derived?
A. Civic Park is north of the administrative service area
B. The leisure area is southwest of the cultural area
C. The cultural district is in the northeast of the business district
D. The business district is southeast of the leisure area

Thought: Let me analyze the given information step by step:
1. The paragraph describes relative positions of areas
2. We know: administrative service area is southwest of cultural area
3. We know: cultural area is southeast of leisure area
4. These areas are centered on the citizen park

Thought: Let me evaluate each choice:
A. Cannot be determined - Civic Park's position relative to administrative service area is not stated
B. Incorrect - This contradicts given info (cultural area is southeast of leisure area)
C. Cannot be determined - Business district's position relative to cultural district is not stated
D. Cannot be determined - Business district's position relative to leisure area is not stated

Thought: Based on the given information, none of the choices can be definitively derived. The paragraph only gives relative positions between administrative service area, cultural area, and leisure area.

---
Answer: a

(End of example)

Remember, your task is to think step by step and select only one choice that you think answers the question with respect to the paragraph given. Do not add any explanation, comments, introduction or conclusion, you shall only return your answer in the format "Answer: X" where X is one of a, b, c, or d.

{paragraph}

{question}
{choices}

Current reasoning:
{current_state}

---
Answer: '''

# Self-evaluation prompts for LogiQA
self_evaluate_step = '''You are evaluating a reasoning step in a logical reasoning task. Given the paragraph, question, choices, and the proposed reasoning step, determine if this step is correct and logical. Consider:
1. Does the reasoning follow from the given information?
2. Is the analysis of the choices accurate?
3. Are the logical connections valid?
4. Does it avoid making unsupported assumptions?

Paragraph: {paragraph}

Question: {question}
Choices: {choices}

Previous steps:
{previous_steps}

Current step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a logical reasoning task. Given the paragraph, question, choices, and the reasoning process, determine if the solution is correct. Consider:
1. Does the reasoning process follow logically from the given information?
2. Is the analysis of each choice thorough and accurate?
3. Are all logical connections valid and well-supported?
4. Does the final answer follow from the reasoning?
5. Are there any unsupported assumptions?

Paragraph: {paragraph}

Question: {question}
Choices: {choices}

Reasoning steps:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''