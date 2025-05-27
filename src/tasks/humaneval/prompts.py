SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."
SIMPLE_CHAT_INSTRUCTION_V2 = """You are an AI that only responds with only {lang} code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."""

aggregate_prompt = """You are a programming assistant, who is helping user to write efficient and correct codes. You will be given multiple implementations of the same function. You should choose the {k} best implementation based on the following criterias:
1. Correctness: The implementation should return the correct output.
2. Efficiency: The implementation should be efficient in terms of time and space complexity.
3. Readability: The implementation should be readable and understandable.
4. Style: The implementation should follow the style guide of the language.
5. Testability: The implementation should be testable.

Remember your task is to choose the {k} best implementation based on the above criterias. Make sure to keep each choise in their own code view, i.e., ```<implementation1>```, ```<implementation2>```, etc... Return only your choises and nothing else. Do not add any further thoughts or reasoning.
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

self_evaluation_prompt = """You are a programming assistant. Your job is to determine whether a candidate solution correctly implements a given specification. Answer only with "yes" or "no".

Function signature and docstring:
{prompt}

Implementation:
{implementation}

Do not include any further thoughts or reasoning, answer only with "yes" or "no".
"""