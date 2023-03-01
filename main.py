"""Python file to serve as the frontend"""
import streamlit as st
import os
import faiss, pickle
import openai
from streamlit_chat import message
from streamlit.components.v1 import html
from langchain.agents import Tool
from langchain import OpenAI
from langchain.prompts import load_prompt
from langchain.chains import VectorDBQA
from langchain.chains.question_answering import load_qa_chain

openai.api_key = os.getenv("OPENAI_API_KEY")

#Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

prompt = load_prompt('prompt.json')

qa_chain = load_qa_chain(llm=OpenAI(temperature=0.1), chain_type="stuff",
                        prompt=prompt)

chain = VectorDBQA(combine_documents_chain=qa_chain, vectorstore=store, k=4)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Ask Aifa!", page_icon=":brain:")
html_temp = """
                <div style="background-color:{};padding:1px">
                
                </div>
                """

with st.sidebar:
    st.markdown("""
    # About 
    \n*Aifa* is here to provide you with accurate and easy-to-understand responses in **natural language**.
    \n\nDisclaimer: **Do not** use *Aifa* as a substitute for professional medical advice.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.64)"),unsafe_allow_html=True)
    st.markdown("""
    # How does it work?
    \n*Aifa* has been trained on a large corpus of medical text and can provide accurate responses. 
    \nFrom diseases to treatment options, medication side effects, and much more, Aifa has got you covered.
    \nFeel free to ask follow up questions to Aifa.
    """)
    st.markdown(html_temp.format("rgba(55, 53, 47, 0.64)"),unsafe_allow_html=True)
    st.markdown("""
    <a href = "mailto:fazeen.nasser@outlook.com?subject = Feedback&body = Message">Send Feedback</a>
    """,
    unsafe_allow_html=True,
    )
    st.markdown("""
    <a href="https://www.buymeacoffee.com/fazeen" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 35px;width: 110px ;" ></a>
    """,
    unsafe_allow_html=True,
    )

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    st.markdown("""
    ## Curious about your health? Aifa has answers!
    """)
    input_text = st.text_input("Refresh the page to reset the conversation...", disabled=False, placeholder="Start typing a medical question here and press enter ⏎", key="input")
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

user_input = get_text()

if len(st.session_state["generated"]) < 1:
    prev_a = [""]
else:
    prev_a = st.session_state["generated"][-1:]

if len(st.session_state["past"]) < 1:
    prev_q = [""]
else:
    prev_q = st.session_state["past"][-1:]

query = '\n'.join([f"Q: {prev_q[0]}\nA: {prev_a[0]}\nQ: {user_input}\nA: "])
if user_input:
    result = chain({"query": query})
    output = f"{result['result']}"

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
