import os
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph
from tools import ImageCaptionTool, ObjectDetectionTool, OCRTool

# --- Define EchoTool using BaseTool (replace with your image tools as needed) ---
class EchoToolInput(BaseModel):
    input: str = Field(description="The string to echo back.")

class EchoTool(BaseTool):
    name: str = "Echo"
    description: str = "Echoes back the input string."
    args_schema = EchoToolInput

    def _run(self, input: str):
        return f"Echo: {input}"

    def _arun(self, input: str):
        raise NotImplementedError("This tool does not support async")

# --- State schema for LangGraph ---
class AgentState(BaseModel):
    messages: List[BaseMessage]

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in a .env file or as an environment variable.")
    st.stop()

# --- Initialize tools and LLM ---
tools = [
    ImageCaptionTool(),
    ObjectDetectionTool(),
    OCRTool(),
    # EchoTool(),  # Remove for production
]
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

# --- Create a prompt and bind tools ---
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an assistant that can use tools to help the user. "
     "You have access to tools for image captioning, object detection, and text extraction (OCR). "
     "When given an image path, use the appropriate tool. "
     "Use the OCR tool when the user asks about text in an image. "
     "For follow-up or creative questions, use your previous answers and your own reasoning. "
     "Only use a tool if you need to analyze the image again."
    ),
    MessagesPlaceholder(variable_name="messages"),
])
agent_chain = prompt | llm.bind_tools(tools)

# --- Build LangGraph node ---
def agent_node(state):
    result = agent_chain.invoke({"messages": state.messages})
    return AgentState(messages=state.messages + [result])

# --- Build the graph ---
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
runnable_graph = graph.compile()

# --- Streamlit UI ---
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
    }
    /* Hide the entire file row (name, size, and x button) under the uploader */
    [data-testid="stFileUploaderFile"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Chat With an Image")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "clear_input" not in st.session_state:
    st.session_state["clear_input"] = False
if "input_processed" not in st.session_state:
    st.session_state["input_processed"] = False
    st.session_state["user_input"] = ""

# --- Fix rerun loop and clear input ---
if st.session_state.get("clear_input", False):
    st.session_state["user_input"] = ""
    st.session_state["clear_input"] = False
    st.rerun()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file and "image_path" not in st.session_state:
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getbuffer())
        st.session_state["image_path"] = tmp.name

if "image_path" in st.session_state:
    st.image(st.session_state["image_path"], use_container_width=True)

    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        user_input = st.text_input(
            " ",  # Use a single space to avoid Streamlit warning
            key="user_input",
            placeholder="Ask a question about your image...",
            label_visibility="collapsed"
        )
    with col2:
        ask_clicked = st.button(
            "Ask",
            use_container_width=True,
            type="primary"  # Makes the button blue in Streamlit >=1.25
        )
    with col3:
        clear_clicked = st.button(
            "Clear",
            use_container_width=True
        )

    # Handle Clear chat button
    if clear_clicked:
        st.session_state["messages"] = []
        st.session_state["clear_input"] = True  # Set flag instead of directly clearing input
        st.rerun()

    # Initialize submitted before using it
    submitted = False
    if ask_clicked:
        submitted = True
    elif user_input and not st.session_state.get("input_processed", False):
        submitted = True

    if submitted and user_input:
        if len(st.session_state["messages"]) == 0:
            question = f"{user_input}\nImage path: {st.session_state['image_path']}"
        else:
            question = user_input
        st.session_state["messages"].append(HumanMessage(content=question))
        st.session_state["input_processed"] = True  # Prevent double submit
        state = AgentState(messages=st.session_state["messages"])
        try:
            result = runnable_graph.invoke(state)
            messages = getattr(result, "messages", None)
            if messages is None:
                messages = result["messages"]
            last_msg = messages[-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool = next(t for t in tools if t.name == tool_name)
                    try:
                        tool_output = tool._run(**tool_args)
                        st.session_state["messages"].append(AIMessage(content=tool_output))
                        st.session_state["clear_input"] = True
                        st.session_state["input_processed"] = False
                        st.rerun()
                    except Exception as e:
                        st.session_state["messages"].append(AIMessage(content=f"Tool {tool_name} failed with error: {e}"))
                        st.session_state["clear_input"] = True
                        st.session_state["input_processed"] = False
                        st.rerun()
            else:
                st.session_state["messages"].append(AIMessage(content=last_msg.content))
                st.session_state["clear_input"] = True
                st.session_state["input_processed"] = False
                st.rerun()
        except Exception as e:
            st.session_state["messages"].append(AIMessage(content=f"Agent error: {e}"))
            st.session_state["clear_input"] = True
            st.session_state["input_processed"] = False
            st.rerun()
    elif not user_input:
        st.session_state["input_processed"] = False

else:
    st.info("Please upload an image to start chatting.")

# --- Display chat history in a clean format ---
for msg in reversed(st.session_state["messages"]):
    if isinstance(msg, HumanMessage):
        # Remove 'Image path: ...' from display
        lines = msg.content.splitlines()
        filtered_lines = [line for line in lines if not line.strip().startswith("Image path:")]
        display_content = "\n".join(filtered_lines)
        st.markdown(
            f"<div style='background:#e6f7ff;padding:8px;border-radius:6px;margin-bottom:4px'><b>You:</b> {display_content}</div>",
            unsafe_allow_html=True
        )
    elif isinstance(msg, AIMessage):
        st.markdown(
            f"<div style='background:#f6ffed;padding:8px;border-radius:6px;margin-bottom:4px'><b>Assistant:</b> {msg.content}</div>",
            unsafe_allow_html=True
        )
