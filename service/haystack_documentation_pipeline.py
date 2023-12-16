from typing import Dict, Any, Callable

from haystack import Pipeline
from haystack.agents.base import ToolsManager
from haystack.nodes import PromptNode, SentenceTransformersRanker
from haystack.agents import Agent, Tool

from service.utils.memory_node import return_memory_node
from service.utils.prompts import agent_prompt
from service.utils.retriever import return_retriever


def resolver_function(
    query: str,
    agent: Agent,
    agent_step: Callable,
) -> Dict[str, Any]:
    """
    This function is used to resolve the parameters of the prompt template.
    :param query: the query
    :param agent: the agent
    :param agent_step: the agent step
    :return: a dictionary of parameters
    """
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }


def define_haystack_doc_searcher_tool() -> Tool:
    """
    Defines the tool for searching the Haystack documentation.
    :return: the Haystack documentation searcher tool
    """
    ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", top_k=5)
    retriever = return_retriever()
    haystack_docs = Pipeline()
    haystack_docs.add_node(component=retriever, name="retriever", inputs=["Query"])
    haystack_docs.add_node(component=ranker, name="ranker", inputs=["retriever"])

    return Tool(
        name="haystack_documentation_search_tool",
        pipeline_or_node=haystack_docs,
        description="Searches the Haystack documentation for information.",
        output_variable="documents",
    )


def return_haystack_documentation_agent(openai_key: str) -> Agent:
    """
    Returns an agent that can answer questions about the Haystack documentation.
    :param openai_key: the OpenAI key
    :return: the agent
    """

    agent_prompt_node = PromptNode(
        "gpt-3.5-turbo-16k",
        api_key=openai_key,
        stop_words=["Observation:"],
        model_kwargs={"temperature": 0.05},
        max_length=10000,
    )

    agent = Agent(
        agent_prompt_node,
        prompt_template=agent_prompt,
        prompt_parameters_resolver=resolver_function,
        memory=return_memory_node(openai_key),
        tools_manager=ToolsManager([define_haystack_doc_searcher_tool()]),
        final_answer_pattern=r"(?s)Final Answer\s*:\s*(.*)",
    )

    return agent
