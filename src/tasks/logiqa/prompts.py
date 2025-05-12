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
Answer: a
Answer: a
Answer: b
Answer: c
Answer: a

Selected answers:
Answer: a
Answer: a
Answer: a

(End of example)

Remember, your task is to select exactly {k} possible answers, which you think is correct with respect to the question and paragraph. Do not add any explanation, comments, introduction, conclusion, or modification, you shall only return your selected answers.

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