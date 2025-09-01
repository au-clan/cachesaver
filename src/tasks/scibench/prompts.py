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