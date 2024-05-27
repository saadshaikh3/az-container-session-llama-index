import os
from llama_index.core.settings import Settings
from llama_index.core.agent import AgentRunner, ReActAgent
from llama_index.core.tools.query_engine import QueryEngineTool
from app.engine.tools import ToolFactory
from app.engine.index import get_index
from llama_index.tools.azure_code_interpreter import (
    AzureCodeInterpreterToolSpec,
)
from llama_index.core import PromptTemplate


def get_chat_engine():
    system_prompt = os.getenv("SYSTEM_PROMPT")
    top_k = os.getenv("TOP_K", "3")
    tools = []

    # Add query tool if index exists
    index = get_index()
    if index is not None:
        query_engine = index.as_query_engine(similarity_top_k=int(top_k))
        query_engine_tool = QueryEngineTool.from_defaults(query_engine=query_engine)
        tools.append(query_engine_tool)

    # Add additional tools
    tools += ToolFactory.from_env()

    return AgentRunner.from_llm(
        llm=Settings.llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=True,
    )

def get_az_interpreter_chat_engine():

    POOL_MANAGEMENT_ENDPOINT = os.getenv("POOL_MANAGEMENT_ENDPOINT")

    azure_code_interpreter_spec = AzureCodeInterpreterToolSpec(
        pool_managment_endpoint=POOL_MANAGEMENT_ENDPOINT,
        local_save_path="./visual_outputs",
    )

    DATASET_PATH = os.getenv("DATASET_PATH")

    # Upload a sample dataset file of students performances
    res = azure_code_interpreter_spec.upload_file(
        local_file_path=DATASET_PATH
    )

    # Create the ReActAgent and inject the tools defined in the AzureDynamicSessionsToolSpec
    agent = ReActAgent.from_tools(
        azure_code_interpreter_spec.to_tool_list(), llm=Settings.llm, verbose=True
    )

    #modifying the ReActAgent's prompts a bit in the Additional section

    react_system_header_str = """\

    You are designed to help with a variety of tasks, from answering questions \
        to providing summaries to other types of analyses.

    ## Tools
    You have access to a wide variety of tools. You are responsible for using
    the tools in any sequence you deem appropriate to complete the task at hand.
    This may require breaking the task into subtasks and using different tools
    to complete each subtask.

    You have access to the following tools:
    {tool_desc}


    ## Output Format
    To answer the question, please use the following format.

    ```
    Thought: I need to use a tool to help me answer the question.
    Action: tool name (one of {tool_names}) if using a tool.
    Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
    ```

    Please ALWAYS start with a Thought.

    Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

    If this format is used, the user will respond in the following format:

    ```
    Observation: tool response
    ```

    You should keep repeating the above format until you have enough information
    to answer the question without using any more tools. At that point, you MUST respond
    in the one of the following two formats:

    ```
    Thought: I can answer without using any more tools.
    Answer: [your answer here]
    ```

    ```
    Thought: I cannot answer the question with the provided tools.
    Answer: Sorry, I cannot answer your query.
    ```

    ## Additional Rules
    - If the user query talks about dataset, then you are supposed to look into the uploaded file present in the local dataset folder.
    - In case of file modification, use the same name of the modified file as the input file's

    ## Current Conversation
    Below is the current conversation consisting of interleaving human and assistant messages.

    """
    react_system_prompt = PromptTemplate(react_system_header_str)
    agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
    agent.reset()

    if len(res) != 0:
        return agent, azure_code_interpreter_spec