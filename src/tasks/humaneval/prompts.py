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

Action: ```python
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''
    # Input validation
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b
```

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