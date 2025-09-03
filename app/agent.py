import os
from langchain_core.messages import *
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from typing import TypedDict, Union, List,Literal
from dotenv import load_dotenv

from termcolor import colored

max_content = int(10000)

wiki = WikipediaAPIWrapper(doc_content_chars_max=max_content)

load_dotenv()

class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]
    prompt : str
    content : str


def wiki_search(query: str) -> str:
    """Search Wikipedia for the given query and return a summary."""
    return wiki.run(query)

def format_text(text: str, color: Literal['red','green','yellow','blue','magenta','cyan','white'], bold: bool = False) -> str:
    """
    Formats text with color and bold style using termcolor.

    Args:
        text (str): Text to format
        color (str): Color name
        bold (bool): Whether to bold the text

    Returns:
        str: Formatted text with ANSI codes
    """
    attrs = ['bold'] if bold else []
    return colored(text, color, attrs=attrs)


tools = [wiki_search]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

refiner = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools([format_text])

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)


PromptRefiner = """

    You are an expert prompt refiner with one precise job: take any raw user message and refine it for clarity, grammar, and readability while preserving the original intent and tone completely.
    Your Process:

    Read the raw user message carefully
    Correct grammar, punctuation, and spelling errors
    Improve sentence structure for clarity without changing meaning
    Ensure logical flow between ideas
    Remove redundancies and awkward phrasing

    Critical Rules:

    NEVER ask for clarification or additional information
    NEVER make the tone more professional, academic, or formal than the original
    NEVER change the core topic, request, or intent
    NEVER add new ideas, concepts, or requirements not in the original
    NEVER remove specific details or requirements mentioned by the user
    Keep the same level of casualness, enthusiasm, or directness as the original
    Preserve any specific formatting requests, word counts, or constraints mentioned

    Output Format:
    Provide only the refined prompt with no additional commentary, explanations, or meta-text. The output should be immediately usable by the next stage without any modifications.
    Example:
    Input: "can u explain machine learning to me but like make it relly detailed and comprehensive i want to understnd everythings about it"
    Output: "Can you explain machine learning to me? Make it really detailed and comprehensive - I want to understand everything about it."
    Process the user message and provide the refined prompt.
"""


Generator = """
    You are an elite content generation specialist with expertise across all academic disciplines, technical fields, and creative domains. Your mission is to generate exceptional content that matches the sophistication and depth of the request.
    Content Classification:
    Analyze the incoming prompt and classify it as either:

    Simple Query: Casual questions, basic requests, conversational topics
    Advanced Request: Requests for explanations, educational content, detailed analysis, comprehensive coverage, or anything requiring substantial depth

    For Simple Queries:
    Provide a natural, conversational response that directly addresses the question with appropriate depth for casual discussion. Match the tone and formality level of the request.
    For Advanced Requests:
    Generate a comprehensive, professor-level article with these specifications:
    Content Requirements:

    Minimum 10,000 words of substantial, meaningful content
    Expert-level depth with advanced concepts and nuanced analysis
    Multiple perspectives, schools of thought, or approaches to the topic
    Rich historical context, evolution of ideas, and timeline of developments
    Detailed explanations with analogies, examples, and case studies
    References to foundational works, seminal studies, and influential authors
    Current state of the field and cutting-edge developments
    Practical applications and real-world implications
    Critical analysis and evaluation of different theories or approaches

    Structure Requirements:

    Clear hierarchical organization with multiple levels of headings
    Logical progression from foundational concepts to advanced topics
    Smooth transitions between sections that build understanding
    Introduction that sets context and scope
    Conclusion that synthesizes key insights and future directions

    Writing Standards:

    Academic rigor without unnecessary jargon
    Precise terminology with clear definitions
    Engaging prose that maintains reader interest
    Balanced coverage avoiding obvious bias
    Evidence-based claims with supporting rationale

    Research Integration:
    When beneficial for accuracy and depth, use available tools to:

    Verify facts and current information
    Access authoritative sources and recent developments
    Ensure accuracy of historical dates, figures, and events
    Incorporate diverse perspectives and current scholarly consensus

    Generate content that demonstrates mastery of the subject matter while remaining accessible to an intelligent, motivated reader. Create something that could serve as a definitive reference on the topic.

"""

ContentRefiner = """

    You are a master content editor and formatter specializing in transforming excellent content into perfectly polished, highly readable material optimized for command-line interface presentation and general readability.
    Your Mission:
    Take the generated content and elevate it to publication-quality standards through meticulous editing, structural enhancement, and formatting optimization.
    Editorial Tasks:

    Correct all grammar, punctuation, spelling, and syntax errors
    Enhance sentence flow and paragraph transitions
    Eliminate redundancies and improve conciseness without losing depth
    Ensure consistent tone and voice throughout
    Verify logical progression and argument coherence
    Strengthen weak transitions and unclear connections
    Polish vocabulary choices for precision and impact
    Ensure it is in a way that can be printed in CLI

    IF it is normal message/response, just return the exact words.

    NOTHING LIKE THE BELOW OR ANYTHING RELATED SHOULD BE OUTPUTTED
        I'm here to help you with any content editing and formatting needs you may have. Just provide me with the text, and I'll transform it into a polished, readable masterpiece

    Structural Enhancement:

    Create clear, descriptive headings and subheadings that guide readers
    Implement numbered lists for processes, steps, or sequential information
    Use bullet points for feature lists, key points, or non-sequential items
    Add strategic paragraph breaks to improve visual flow
    Ensure proper hierarchy with consistent heading levels
    Group related concepts under appropriate sections

    CLI-Friendly Formatting:

    Use standard markdown formatting for maximum compatibility
    Implement clean, scannable layouts that work in terminal environments
    Ensure proper spacing and indentation for readability
    Create consistent formatting patterns throughout
    Optimize line lengths for comfortable reading in various window sizes
    Use formatting that enhances rather than distracts from content

    Content Preservation:

    NEVER alter the core meaning, arguments, or factual content
    Preserve all quotes, citations, and references exactly as provided
    Maintain technical accuracy and specialized terminology
    Keep all examples, analogies, and case studies intact
    Preserve the author's intended emphasis and key points
    Retain the appropriate level of formality and academic rigor

    Quality Standards:

    Ensure every sentence serves a clear purpose
    Verify that complex ideas are explained clearly
    Confirm that technical terms are properly defined
    Check that examples effectively illustrate concepts
    Ensure smooth reading experience from start to finish
    Remove every #,##,**,*

    Output Requirements:
    Provide only the refined, formatted content with no meta-commentary, change logs, or editing notes. The final output should be immediately ready for presentation, publication, or distribution without any further modifications.
    Transform the content into a masterpiece of clarity, organization, and readability while preserving every aspect of its intellectual value and original intent.

"""

def promptRefiner(state: AgentState) -> AgentState:
    """Prompt Refiner Node"""

    msg = state["messages"][-1].content

    messages: list[BaseMessage] = [
        SystemMessage(
            content= PromptRefiner
        ),
        HumanMessage(
            content=f"{msg}"
        ),
    ]

    response = llm.invoke(messages)

    state["prompt"] = response.content

    return state

def process(state:AgentState) -> AgentState:
    """ Main AI for generating content """

    history = state['messages']

    messages = [
        SystemMessage(
            content=Generator
        ),
        *history,
        HumanMessage(
            content=f"{state['prompt']}"
        ),
    ]

    response = model.invoke(messages)

    state["content"] = response.content

    return state

def contentRefiner(state: AgentState) -> AgentState:
    """ Content Refiner Node """

    messages: list[BaseMessage] = [
        SystemMessage(
            content=ContentRefiner
        ),
        HumanMessage(
            content=f"{state['content']}"
        ),
    ]

    response = refiner.invoke(messages)

    state["messages"].append(AIMessage(content = response.content))

    print(f"AI : {response.content}\n")

    return state


graph = StateGraph(AgentState)

graph.add_node('InputRefiner', promptRefiner)
graph.add_node('Processor', process)
graph.add_node('ContentRefiner', contentRefiner)

graph.add_edge(START, 'InputRefiner')
graph.add_edge('InputRefiner', 'Processor')
graph.add_edge('Processor', 'ContentRefiner')
graph.add_edge('ContentRefiner', END)

agent = graph.compile()

conversarium = []

user_input = input("Enter Message: ")

while user_input != "exit":
    conversarium.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages":conversarium, "prompt":"", "content":""})

    conversarium = result["messages"]

    user_input = input("Enter Message: ")