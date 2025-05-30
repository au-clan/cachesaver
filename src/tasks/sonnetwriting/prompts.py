act = '''You are fluent in sonnet writing and can only respond with your sonnet and nothing else.
Below, you will be given your task and keywords to include in your sonnet writing. Remember to return only your sonnet writing.

{input}'''

bfs = '''You are fluent in writing sonnets and must respond with only your sonnet texts—no explanations or extra comments. Write at least two full sonnets.

You will be given a prompt and keywords to include in your sonnets.

Each sonnet should be wrapped in triple backticks like this:
\`\`\`
<Sonnet here>
\`\`\`

Only return the sonnets, nothing else.

Task and keywords:
{input}'''


aggregate = '''There have been created some sonnet writings with respect to the following task:
{task}

Select the {k} best sonnet writings with respect to the task, without any modification to the sonnet writings. Do not add any explanation, comments, introduction or conclusion, your should only return the sonnets writings.
You select a sonnet writing by returning the number of that sonnet writing.

{examples}

(End of examples)

Remember, your task is to select the {k} best sonnet writings. Do not add any explanation, comments, introduction, conclusions or modifications.
You should only return the numbers of the selected sonnet writings.

{sonnets}'''

evaluate = '''You are given a sonnet writing below. Your task is to evaluate the sonnet based on the criterias:
    1. Follows the rhyme scheme: {rhyme_scheme}
    2. Contains these exact words: {words}

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the given sonnet between 0 and 10. Do not add any explanations, comments, introduction or conclusions.

{sonnet}
'''

examples_evaluate = [
    '''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

The river bends beside the morning grass (A)  
A shimmer dances on the silver tide (B)  
The hours like golden moments swiftly pass (A)  
And fortune's value flows with gentle pride (B)  

The jail of fear breaks open with the breeze (C)  
The open fields forgive the bitter rain (D)  
The grass revives beneath the waking trees (C)  
And songs of hope replace the cries of pain (D)  

Bright grasslands whisper secrets to the skies (E)  
The jail of winter thaws beneath the light (F)  
The value found in spring will never die (E)  
But bloom in endless fields beyond our sight (F)  

The grass will grow where broken dreams once fall (G)  
And value shines within the hearts of all. (G)

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

The mountain sighs beneath the winter snow
The river carves its path with solemn might
The flowers sleep beneath the frozen glow
Awaiting touch of spring’s returning light

No jail confines the wild and restless breeze
It tumbles past the valley’s sleeping door
The grass will rise when winter grants release
Yet value fades along the rocky shore

The colors blend beneath a paling sky
The stars retreat behind a drifting veil
Though grass returns, some dreams must say goodbye
And hearts once brave grow weary and grow pale

The seasons turn, and leave us wondering still
What value clings to hope, and what to will.

---END-OF-SONNET---

evaluation: 10 
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

Upon the cliffs the storm begins to roar
The candles flicker in the empty hall
The silver mist creeps underneath the door
A solemn hush descends upon the wall

No grassy fields are near this barren land
No value glimmers in the heavy rain
No jail can hold the fury in its hand
The waves collapse against the rocks in vain

The sailors cry into the endless dark
No flame to guide them through the cruel, wild gale
No harvest waits beyond the fading spark
Just shattered hopes imprisoned without bail

Their names are lost beneath the mourning wave
No grass, no value, only storms to brave.

---END-OF-SONNET---

evaluation: 10
''',
'''Required Rhyme Scheme: ABAB CDCD EFEF GG  
Required Words: grass, value, jail

The twilight burns across the shattered shore (A)  
Old lanterns flicker in the dying mist (B)  
The crows descend where empty houses mourn (C)  
A bitter memory clenched within a fist (B)  

The river moans beneath a sky of ash (D)  
Its course forgotten by the sleeping stone (E)  
The mountains tremble under thunder’s crash (D)  
The plains are silent, aching and alone (F)  

No voices rise above the broken field (G)  
No banners fly beneath the heavy rain (H)  
The meadow bows beneath the turning wheel (I)  
While dreams decay and vanish in their pain (H)  

Night falls without a sound, a hollow art (J)  
And leaves the world with one forsaken heart. (J)

---END-OF-SONNET---

evaluation: 0
'''
]

aggregate_examples = [
    '''Beneath the whisper of the waving grass (A)
The river sings a song of ancient lore (B)
Each golden moment holds a priceless mass (A)
Of value set beyond the richest store. (B)

The sunlight weaves a pattern soft and bright (C)
Across the meadows where the daisies sail, (D)
Yet even beauty feels a distant blight (C)
When love is locked within a silent jail. (D)

O time, who takes but seldom grants us grace, (E)
Within your tides do all our dreams entwine, (F)
Yet still the grass shall whisper of this place, (E)
A testament to moments lost in time. (F)

So hold the value of each fleeting breath, (G)
For life escapes more swiftly than its death. (G)

---END-OF-SONNET---

The jail of sorrow keeps my heart in thrall (A)
While grass grows wild beyond the broken gate. (B)
I find no value in these chains at all, (A)
Yet still I bear the burden of my fate. (B)

A lonesome wind creeps through the ruined hall (C)
And sings of dreams that once could not prevail, (D)
Now dust and silence answer every call, (C)
Their memories locked in an endless jail. (D)

But hope, like grass, persists beyond the stone (E)
And value shines where shadows used to tread; (F)
A broken heart can yet become its own, (E)
A phoenix rising from the dreams long dead. (F)

O jail, O grass, O value lost and won, (G)
Your cycle ends anew with each day's sun. (G)

---END-OF-SONNET---

I saw a meadow bright with golden flame (A)
And heard the jailer laughing in his cell; (B)
The value of the fields he could not name, (A)
For grass and sunlight meant no tales to tell. (B)

The sky was pale and heavy with the rain, (C)
While iron gates were rusted in the gale; (D)
A robin sang a melancholy strain (C)
Outside the damp and sorrow-scented jail. (D)

The grass grew thick and wove a velvet sheet (E)
Where value lay in every drop of dew, (F)
And in the mud the jailer dragged his feet, (E)
Unknowing that the grass would soon renew. (F)

The jail will fall, the grass will rise instead, (G)
For nature reclaims even iron and dread. (G)

---END-OF-SONNET---

Selected 2 Best sonnets:
1
2''',
'''Upon the hill where once the grass was green
The winds now whisper secrets of the vale,
A place where none but memories have been
And sorrow weeps behind a broken jail.

The stars above reflect in pools below
While tender dreams dissolve without a trace,
Yet hope still carves its message in the snow
And leaves the value written on its face.

The seasons turn with neither grief nor song
And grass will rise to meet the morning tide,
The heart must travel even when it's wrong,
And seek the dreams that time cannot deride.

We hold the broken past within our hands,
And plant new hopes upon the barren lands.

---END-OF-SONNET---

In fields of gold where bending grasses sway
I found a road that led beyond the stream,
A shining path that twisted night to day,
And led me farther from the jail of dream.

The value of a single step was small
Yet each one built a bridge of lasting stone;
The stars above seemed waiting for the call
To guide me far from everything I'd known.

The blades of grass were wet with morning's grace
And songs of sparrows filled the misty air;
They sang of freedom no jail could replace
Of journeys shaped by hope beyond despair.

I walked the road and felt the future start,
The grass and stars still singing in my heart.

---END-OF-SONNET---

The jail was hidden deep beneath the trees (A)
Where no soft grass could ever hope to grow. (B)
The shadows whispered warnings on the breeze, (A)
A chilling song the heart would never know. (B)

No value marked the time that passed away, (C)
The walls were cold and silent in their might, (D)
And grassless stones forever bleak and gray (C)
Reflected only sorrow in their sight. (D)

Yet through the cracks a single flower rose (E)
Defiant in the darkness and the pain, (F)
Its fragile beauty trembling as it grows, (E)
A symbol that the spirit will remain. (F)

Where grass may fail, and even value die, (G)
Still hope will blossom if it dares to try. (G)

---END-OF-SONNET---

Selected 2 best sonnets:
2
3'''
]

# React prompt for SonnetWriting
react = '''You are fluent in sonnet writing. You will be given a task that specifies the rhyme scheme and words that must be included. Your task is to write a sonnet that follows these requirements while maintaining poetic quality.

You should think step by step about the logical reasoning needed to write the sonnet:

1. First, analyze the rhyme scheme and required words
2. Then, plan how to incorporate the required words naturally
3. Next, develop a theme that connects these words meaningfully
4. Finally, write the sonnet following the required format

(Example)
Task: Write a sonnet with rhyme scheme ABAB CDCD EFEF GG and include the words: grass, value, jail

Thought: Let me analyze the requirements step by step:
1. Need to follow ABAB CDCD EFEF GG rhyme scheme
2. Must include "grass", "value", and "jail" naturally
3. Should create a theme connecting these words
4. Need to write in iambic pentameter

Thought: Let me plan the sonnet structure:
1. First quatrain: Introduce "grass" and set the scene
2. Second quatrain: Incorporate "jail" and develop theme
3. Third quatrain: Use "value" and reinforce theme
4. Couplet: Conclude with a meaningful message

Thought: Now I'll write the sonnet following this plan:

The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---
The river bends beside the morning grass (A)
A shimmer dances on the silver tide (B)
The hours like golden moments swiftly pass (A)
And fortune's value flows with gentle pride (B)

The jail of fear breaks open with the breeze (C)
The open fields forgive the bitter rain (D)
The grass revives beneath the waking trees (C)
And songs of hope replace the cries of pain (D)

Bright grasslands whisper secrets to the skies (E)
The jail of winter thaws beneath the light (F)
The value found in spring will never die (E)
But bloom in endless fields beyond our sight (F)

The grass will grow where broken dreams once fall (G)
And value shines within the hearts of all. (G)

---END-OF-SONNET---

(End of example)

Remember, your task is to think step by step and write a sonnet that follows the required rhyme scheme and includes all specified words naturally. Do not add any explanation, comments, introduction or conclusion, you shall only return your sonnet followed by ---END-OF-SONNET---.

Task: {input}

Current reasoning:
{current_state}

---
'''

# Self-evaluation prompts for SonnetWriting
self_evaluate_step = '''You are evaluating a step in sonnet writing. Given the task requirements and the current step, determine if this step is correct and contributes to a good sonnet. Consider:
1. Does it follow the required rhyme scheme?
2. Does it maintain iambic pentameter?
3. Does it contribute to the overall theme?
4. Does it use the required words naturally?

Task: {input}

Previous steps:
{previous_steps}

Current step: {step}

Is this step correct and well-written? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete sonnet. Given the task requirements and the sonnet, determine if it meets all criteria. Consider:
1. Does it follow the required rhyme scheme?
2. Does it include all required words naturally?
3. Does it maintain iambic pentameter throughout?
4. Does it have a coherent theme and imagery?
5. Is it grammatically correct and poetic?

Task: {input}

Sonnet:
{sonnet}

Is this sonnet correct and well-written? Answer with a single word: Yes or No.
'''