"""Python file to serve as the frontend"""
import streamlit as st
import os
import faiss, pickle
from streamlit_chat import message
#from langchain import PromptTemplate
from langchain.agents import Tool
#from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.prompts import load_prompt
#from langchain.agents import initialize_agent
#from gpt_index import GPTSimpleVectorIndex

openai.api_key = os.getenv("OPENAI_API_KEY")

#Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
prompt=load_prompt("prompt.json")
chain = ChatVectorDBChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)
#index = GPTSimpleVectorIndex.load_from_disk('simple_index.json')
#tools = [
#    Tool(
#        name = "Aifa",
#        func=lambda q: str(index.query(q)),
#        description="useful for when you want answers to medical queries. The input to this tool should be a complete english sentence.",
#        return_direct=True
#    ),
#]
#memory = ConversationBufferMemory(memory_key="chat_history")
#llm=OpenAI(temperature=0)
#chain = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Ask Aifa 🧠", page_icon=":brain:")
html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    \n*Aifa* is a smart bot that answers medical queries in a **simple language**.
    \n\n**Do not** use *Aifa* as a substitute for professional medical advice.
    \n\n\nComing Soon: _Sources_!
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.64)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    \n*Aifa* has been trained on a large corpus of medical text and can provide accurate responses. 
    \nSimply type your question in the text box and hit enter to get a response. 
    You can also download the output as txt.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.64)"),unsafe_allow_html=True)
    st.markdown("""
    <a href = "mailto:abc@example.com?subject = Feedback&body = Message">Send Feedback</a>
    """,
    unsafe_allow_html=True,
    )

st.header("Chat with Aifa 🧠")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("Start typing below and click enter ⏎", disabled=False, placeholder="What are beta blockers?", key="input")
    return input_text

hide="""
<style>
footer{
	visibility: hidden;
    position: relative;
}
.viewerBadge_container__1QSob{
    visibility: hidden;
}
#MainMenu{
	visibility: hidden;
}
<style>
"""
st.markdown(hide, unsafe_allow_html=True)

st.markdown(
    """
    <style>
        iframe[width="220"] {
            position: fixed;
            bottom: 60px;
            right: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

user_input = get_text()

if len(st.session_state["generated"]) < 2 :
    log = str(st.session_state["generated"].copy())
else: 
    log = str(st.session_state["generated"][-2:])

chat_history = [(user_input, log)]
if user_input:
    result = chain({"question":user_input, "chat_history": chat_history})
    output = f"{result['answer']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")