import os

from haystack import Pipeline
from haystack.agents.base import ToolsManager
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode, WebRetriever, PreProcessor, TopPSampler, Shaper, PromptTemplate
from haystack.agents import Agent, Tool
from haystack.pipelines import WebQAPipeline
from haystack_memory.memory import MemoryRecallNode
from haystack_memory.utils import MemoryUtils




def return_haystack_documentation_agent():
    openai_key = os.environ['OPENAI_KEY']
    serperdev_api_key = os.environ['SERPERDEV_API_KEY']

    agent_prompt_node = PromptNode(
        "gpt-3.5-turbo",
        api_key=openai_key,
        stop_words=["Observation:"],
        model_kwargs={"temperature": 0.05},
        max_length=8095,
    )

    preprocessor = PreProcessor(
        split_by="word",
        split_length=512,
        split_respect_sentence_boundary=True,
        split_overlap=20,
    )

    haystack_documentation_search = WebRetriever(
        api_key=serperdev_api_key,
        allowed_domains=["docs.haystack.deepset.ai"],
        mode="preprocessed_documents",
        preprocessor=preprocessor,
        top_search_results=10,
        top_k=5,
    )

    return Agent(
        prompt_node=agent_prompt_node,
        prompt_template='',
        prompt_parameters_resolver='resolver_function',
        tools_manager=ToolsManager(['haystack_documentation_search_tool']),
    )

    # https://haystack.deepset.ai/tutorials/25_customizing_agent


class CustomAgent:
    def __init__(self):
        self.agent, self.working_memory, self.sensory_memory = return_haystack_documentation_agent()
        self.memory_utils = MemoryUtils(
            working_memory=self.working_memory,
            sensory_memory=self.sensory_memory,
            agent=self.agent,
        )

    def run(self, query):
        return self.memory_utils.chat(query)
