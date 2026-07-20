"""Streamlit chat UI. Renders only — no scoring, safety, or KB logic here."""

import streamlit as st

from chatbot import Chatbot

st.set_page_config(page_title="Knowledge-Base Chatbot", page_icon="📚")


@st.cache_resource
def load_chatbot() -> Chatbot:
    # Cached so the knowledge base and any model load once, not every rerun.
    return Chatbot()


def render_response(response: dict) -> None:
    st.markdown(response["message"])
    primary = response["primary"]
    st.markdown(f"**[{primary['title']}]({primary['link']})** · {primary['format']}")
    st.markdown(primary["summary"])
    if primary.get("when_this_helps"):
        st.markdown(
            "**When this helps:**\n"
            + "\n".join(f"- {line}" for line in primary["when_this_helps"])
        )
    if response["related"]:
        links = " · ".join(
            f"[{r['title']}]({r['link']})" for r in response["related"]
        )
        st.markdown(f"**Related:** {links}")


bot = load_chatbot()

st.title("Knowledge-Base Chatbot")
st.caption(
    "This bot does not give personal medical advice. It only points you to "
    "resources in its knowledge base."
)

if "history" not in st.session_state:
    st.session_state.history = []  # ("user", str) or ("assistant", dict)

# Streamlit reruns this script on every interaction: replay stored history
# first, then handle new input.
for role, payload in st.session_state.history:
    with st.chat_message(role):
        if role == "user":
            st.markdown(payload)
        else:
            render_response(payload)

if query := st.chat_input("Ask a question about the program..."):
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    response = bot.answer(query)
    st.session_state.history.append(("assistant", response))
    with st.chat_message("assistant"):
        render_response(response)
