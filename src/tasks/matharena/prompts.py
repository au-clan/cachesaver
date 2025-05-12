# Step-by-step analysis prompt
act = '''Given a math problem, analyze it step by step. Each step should either analyze the problem, explain concepts, or provide the final answer.

Example Problem: Find the area of a circle with radius 5cm.
Step: Analyze[problem]
The problem asks for the area of a circle given its radius of 5cm.

Example Problem: Solve the equation 2x + 4 = 12
Step: Analyze[solution approach]
We need to isolate x by subtracting 4 from both sides and then dividing by 2.

Input Problem: {input}
Next step:
'''

aggregate = '''Given a math problem, and possible approaches to help you solve it. Select the {k} most relevant approaches to solve the problem.

Example Problem: Find the derivative of y = x² + 3x
Possible approaches:
Analyze[problem] Focus on identifying function components
Explain[math concepts] Review power rule and derivative rules
Analyze[solution approach] Apply derivative rules step by step
Explain[solution steps] Show each term's derivative
Finish[y' = 2x + 3]

Selected 3 approaches:
Analyze[problem] Focus on identifying function components
Explain[solution steps] Show each term's derivative
Finish[y' = 2x + 3]

Input Problem: {input}
Possible approaches:
{approaches}
Selected {k} approaches:
'''

# Brainstorming multiple approaches prompt
bfs = '''Given a math problem, list multiple possible approaches to solve it.

Example Problem: Find the derivative of y = x² + 3x
Possible approaches:
Analyze[problem] Focus on identifying function components
Explain[math concepts] Review power rule and derivative rules
Analyze[solution approach] Apply derivative rules step by step
Explain[solution steps] Show each term's derivative
Finish[y' = 2x + 3]

Input Problem: {input}
Possible approaches:
'''

# Complete solution chain-of-thought prompt
cot = '''Solve the given math problem step by step, showing your work and reasoning.

Example Problem: Solve 3x + 6 = 15
Steps:
Analyze[problem] Linear equation with one variable x
Explain[math concepts] Need to isolate x by using inverse operations
Analyze[solution approach] Subtract 6 from both sides, then divide by 3
Explain[solution steps] 3x = 9, then x = 3
Finish[3]

Input Problem: {input}
'''

# Evaluation prompt
evaluate = '''Evaluate if the given math problem can be solved using the available information. Respond with: "solvable", "needs more info", or "impossible".

Example Problem: Find x if 2x = 10
All information is present and equation can be solved
solvable

Example Problem: Find the area of a triangle
Missing base and height measurements
needs more info

Example Problem: Solve √(x+1) = -5
No real solution exists since square root cannot be negative
impossible

Input Problem: {input}
'''

# Solution verification prompt
evaluate_answer = '''Given a math problem and a proposed solution, determine if the solution is correct (correct/incorrect).

Example Problem: Solve 2x = 8
Proposed solution: x = 4
Judge:
correct

Example Problem: Find the area of a circle with radius 3
Proposed solution: A = 28.27
Judge:
incorrect

Input Problem: {input}
Proposed solution: {answer}
Judge:
'''