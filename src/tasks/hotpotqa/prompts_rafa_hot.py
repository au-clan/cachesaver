#todo Note, is RAFA prompts 1:1 with the agentAct, it seems that way
#todo the reflect_prompt is an melt of rafa and evaluate prompt for hotpot
reflect_prompt='''Solve a question answering task with sequential Action steps. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
Here is an attempt answer:
{answer}
And we have the following feedback:
{feedback}
Using the feedback evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.'''
#this is the same as the reflect prompt...
value_reflect_prompt='''Solve a question answering task with sequential Action steps. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
Here is an attempt answer:
{answer}
And we have the following feedback:
{feedback}
Using the feedback evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.'''


#todo rafa plan is almost 1:1 with the act so we avoid using the cot prompt here because that agent doesnt use it ass well
propose_prompt = """Solve a question answering task with sequential Action steps. Action can be three types:

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


#for rafa the act agent is similar: AgentEvaluateGame24 just for hotpotqu, atleast for the evaluate_answer prompt
#todo these two prompts are the same as AgentEvaluateGame24 that is evaluate_answer and value_prompt. Consider using the AgentAct instead of this b...... these two queries
evaluate_answer = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.

Below some examples are give.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the latest available thought, action, and observation based on your reasoning analysis. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}

Evaluation:
'''

value_prompt=[
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