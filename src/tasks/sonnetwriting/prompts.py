react = """Generate a sonnet with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be:

(1) Extend[prompt, append], which extend the prompt

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

Prompt: Write a sonnet with strict rhyme scheme ABAB CDCD EFEF GG, containing each of the following words verbatim: "grass", "value", and "jail".

Thought 1: From the keywords, this sonnet is related to jail, probably related to nature where there is grass
Action 1: Extend[prompt]
Observation 1: The generated sonnet strictly follows rhyme

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Prompt: {sonnet_prompt}
{current_state}"""
