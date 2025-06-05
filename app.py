import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os, json
from gradio.themes.base import Base
from gradio.themes.utils import colors

load_dotenv()
llm = ChatOpenAI(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.0,
)
embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL"))
index = FAISS.load_local("index", embeddings=embeddings, allow_dangerous_deserialization=True)  # uses stored index+vecs
retriever = index.as_retriever(search_kwargs={"k": 5})

SYSTEM_PROMPT = """You are techemotion, an AI assistant that answers only from the provided context. If unsure, say you don't know.\n\n"""
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "Context:\n{context}\n\n"),
    ("human", "{question}"),
])

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | PROMPT
    | llm
)

def chat_fn(message, history):
    """Gradio Chatbot API – stateless except for UI history"""
    result = chain.invoke(message)
    return str(result.content)

class TechEmotionTheme(Base):
    def __init__(self):
        super().__init__(
            primary_hue=colors.cyan,
            secondary_hue=colors.cyan,
            neutral_hue=colors.gray,
        )
        super().set(
            body_background_fill="#000000",
            body_text_color="#ffffff",
            block_background_fill="#000000",
            block_title_text_color="#1febf3",
            input_background_fill="#000000",
            button_primary_background_fill="#1febf3",
            button_primary_text_color="#000000",
            button_primary_background_fill_hover="#ffffff",
            button_primary_text_color_hover="#000000",
        )

tech_emotion_theme = TechEmotionTheme()

custom_head = """
<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap\">
<style> body, .gradio-container { font-family: 'Inter', sans-serif; } </style>
"""

with gr.Blocks(
    theme=gr.themes.Ocean(),
    css="""
    body, .gradio-container { background: #181A1B !important; color: #FFFFFF !important; }
    footer {visibility: hidden !important;}
    #input-row { align-items: center !important; }
    #send-btn { height: 48px !important; min-width: 48px !important; display: flex; align-items: center; justify-content: center; }
    """,
    head=custom_head
) as demo:
    gr.Markdown("# Tech.Emotion Summit 2025 Virtual Assistant - built with ❤️ by Neosperience")
    chatbot = gr.Chatbot(value=[], label="Assistant", type="messages")
    with gr.Row(equal_height=True, elem_id="input-row"):
        msg = gr.Textbox(
            placeholder="Ask me everything about the event, agenda, speakers...",
            lines=1,
            show_label=False,
            container=True,
            scale=8
        )
        send_btn = gr.Button("✈️", scale=1, elem_id="send-btn")

    def respond(user_msg, chat_hist):
        if not user_msg.strip():
            return chat_hist, ""
        answer = chat_fn(user_msg, chat_hist)
        chat_hist = chat_hist + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": answer}]
        return chat_hist, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    send_btn.click(respond, [msg, chatbot], [chatbot, msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)