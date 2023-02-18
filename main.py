"""Python file to serve as the frontend"""
import streamlit as st
import os
import faiss, pickle
import openai
from streamlit_chat import message
from streamlit.components.v1 import html
from langchain.agents import Tool
from langchain import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.prompts import load_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")

#Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index
#prompt=load_prompt("prompt.json")
chain = ChatVectorDBChain.from_llm(llm=OpenAI(temperature=0), vectorstore=store)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Ask Aifa 🧠", page_icon=":brain:")
html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """
button = """
<script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="fazeen" data-color="#5F7FFF" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#ffffff" data-coffee-color="#FFDD00" ></script>
"""

with st.sidebar:
    st.markdown("""
    # About 
    \n*Aifa* is a smart bot that answers medical queries in a **simple language**.
    \n\n**Do not** use *Aifa* as a substitute for professional medical advice.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.64)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work
    \n*Aifa* has been trained on a large corpus of medical text and can provide accurate responses. 
    \nSimply type your question in the text box and hit enter to get a response.
    \nFeel free to ask follow up questions
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.64)"),unsafe_allow_html=True)
    st.markdown("""
    <a href = "mailto:abc@example.com?subject = Feedback&body = Message">Send Feedback</a>
    """,
    unsafe_allow_html=True,
    )

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    st.markdown("""
    # Ask Aifa 🧠
    \n Try asking Aifa a medical question!
    """)
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
html(button, height=70, width=220)
st.markdown(
    """
    <style>
        iframe[width="220"] {
            position: fixed;
            bottom: 24px;
            right: 24px;
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
