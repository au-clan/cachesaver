react = """Solve a question answering task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Run[code], which execute the code
(2) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

Prompt: "from typing import List"
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Run[code]
Observation 1: The code fails with mesage: xxxx

Prompt: "def is_palindrome(string: str) -> bool:"
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Run[code]
Observation 1: The codes run without fail

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Prompt: {code_prompt}
{current_state}"""
