import os

from haystack import Pipeline
from haystack.agents.base import ToolsManager
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode, WebRetriever, PreProcessor, Docs2Answers, DiversityRanker
from haystack.agents import Agent, Tool


def resolver_function(query, agent, agent_step):
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }


def return_retriever():
    preprocessor = PreProcessor(
        split_by="word",
        split_length=512,
        split_respect_sentence_boundary=True,
        split_overlap=20,
    )

    return WebRetriever(
        api_key=os.environ['SERPERDEV_API_KEY'],
        allowed_domains=["docs.haystack.deepset.ai"],
        mode="preprocessed_documents",
        preprocessor=preprocessor,
        top_search_results=20,
        top_k=10,
    )


def generate_pipeline():
    docs2answers = Docs2Answers()

    ranker = DiversityRanker(
        model_name_or_path="all-MiniLM-L6-v2",
        top_k=5,
        similarity="dot_product",
    )

    haystack_docs = Pipeline()
    haystack_docs.add_node(component=return_retriever(), name="retriever", inputs=["Query"])
    haystack_docs.add_node(component=ranker, name="ranker", inputs=["retriever"])
    return haystack_docs


def return_haystack_documentation_agent():
    openai_key = os.environ['OPENAI_KEY']

    haystack_search_tool = Tool(
        name="haystack_documentation_search_tool",
        pipeline_or_node=generate_pipeline(),
        description="Searches the Haystack documentation for answers to your questions.",
        output_variable="documents",
    )

    agent_prompt = """
    In the following conversation, a human user interacts with an AI Agent. This agent is specialises in answering questions about the Haystack documentation.
    The human user poses questions, and the AI Agent goes through several steps to provide well-informed answers. The good outcome of this conversation is of paramount importance for the wellbeing of the human.
    The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools.
    The AI Agent should always be focused on answering the human user's question avoiding any irrelevant information. The agent should provide as many examples as possible, especially code examples.
    The AI Agent should ignore its knowledge when answering the questions.
    The AI Agent has access to these tools:
    {tool_names_with_descriptions}

    The following is the previous conversation between a human and The AI Agent:
    {memory}

    AI Agent responses must start with one of the following:

    Thought: [the AI Agent's reasoning process]
    Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
    Observation: [tool's result]
    Final Answer: [final answer to the human user's question] (on a new line) ++##++##
    
    When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.

    The AI Agent should not ask the human user for additional information, clarification, or context.
    If the AI Agent cannot find a specific answer after exhausting available tools and approaches, it answers with Final Answer: inconclusive

    Question: {query}
    Thought:
    {transcript}
    """

    agent_prompt_node = PromptNode(
        "gpt-4",
        api_key=openai_key,
        stop_words=["Observation:"],
        model_kwargs={"temperature": 0.05},
        max_length=4096,
    )

    memory_node = return_memory_node(openai_key)

    agent = Agent(
        agent_prompt_node,
        prompt_template=agent_prompt,
        prompt_parameters_resolver=resolver_function,
        memory=memory_node,
        tools_manager=ToolsManager([haystack_search_tool]),
        final_answer_pattern=r"Final Answer\s*:\s*(.*)",
    )

    return agent


def return_memory_node(openai_key):
    memory_prompt_node = PromptNode('gpt-3.5-turbo', api_key=openai_key, max_length=2048)
    return ConversationSummaryMemory(memory_prompt_node)
