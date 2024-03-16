from flask import Flask, render_template, request, session
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask_session import Session
import pickle

app = Flask(__name__)
app.secret_key = 'super_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

@app.route("/process", methods=['POST'])
def process():
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    session["conversation"] = pickle.dumps(get_conversation_chain(vectorstore))
    return render_template('process.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_question = request.form['user_question']
        conversation_chain = pickle.loads(session["conversation"])
        response = conversation_chain({'question': user_question})
       
        chat_history = session.get("chat_history", [])
        chat_history.extend(response['chat_history'])
        print(chat_history)
        session["chat_history"] = chat_history
        
        return render_template('chat.html', chat_history=chat_history)
    else:
        chat_history = session.get("chat_history", [])
        return render_template('chat.html', chat_history=chat_history)

@app.route("/")
def index():
    load_dotenv()
    # if "conversation" not in session:
    session["conversation"] = None
    # if "chat_history" not in session:
    session["chat_history"] = []
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
