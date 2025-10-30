from langchain_core.prompts import PromptTemplate
from langchain import hub

# Basic Generation Prompt
prompt_template = hub.pull("rlm/rag-prompt")

# Query Rewriting Prompt
query_rewriting_text = """You are a helpful assistant that generates multiple search queries based on a single input query.

    Perform query expansion. If there are multiple common ways of phrasing a user question
    or common synonyms for key words in the question, make sure to return multiple versions
    of the query with the different phrasings.

    If there are acronyms or words you are not familiar with, do not try to rephrase them.

    Return 2 different versions of the question.
    
    Question: {question}
    """
query_rewriting_template = PromptTemplate.from_template(query_rewriting_text)

# Hyde Query Template
hyde_query_template = PromptTemplate(
        input_variables=["question"],
        template="""Given this question: '{question}'

        Please write a detailed, informative document that directly answers this question. 
        The document should be comprehensive and approximately 500 characters long.
        Write as if you're explaining this topic in a textbook or educational material.

        Document:"""
    )

# Query Decomposition Prompt
query_decompose_prompt = """
        You are a helpful assistant that prepares queries that will be sent to a search component.
        Sometimes, these queries are very complex.
        Your job is to simplify complex queries into multiple queries that can be answered
        in isolation to eachother.

        If the query is simple, then keep it as it is.
        Examples
        1. Query: Did Microsoft or Google make more money last year?
        Decomposed Questions: [Question(question='How much profit did Microsoft make last year?', answer=None), Question(question='How much profit did Google make last year?', answer=None)]
        2. Query: What is the capital of France?
        Decomposed Questions: [Question(question='What is the capital of France?', answer=None)]
        3. Query: {question}
        Decomposed Questions:
    """
query_decompose_template = PromptTemplate.from_template(query_decompose_prompt)