import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
import pickle
from dotenv import load_dotenv
import os


if "messages" not in st.session_state.keys():  # Inicializa o histórico de mensagens do chat
    st.session_state.messages = []


# Carregar as variáveis de dentro do .env
load_dotenv()


def load_data():
    
    # Carregamento do PDF e/ou do .pkl

    with st.spinner(text="Carregando – aguarde! Isso deve levar de 1 a 2 minutos."):

        # Se o arquivo de Embbeding já existe então carrega
        if (os.path.exists(f"{'ProcuradoriaGeral'}.pkl") and os.path.getsize(f"{'ProcuradoriaGeral'}.pkl") > 0):
            with open(f"{'ProcuradoriaGeral'}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
                return vectorStore
        # Se o arquivo de Embbeding não existe então cria
        else:

            # O pdf é carregado diretamente da pasta local
            pdf = PdfReader('ProcuradoriaGeral.pdf')

            if pdf is not None:
                # Ler o arquivo recebido
                pdf_reader = pdf

                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # TODO: É necessário que o texto analisado seja dividido em partes para que o modelo LLM \
                #        consiga utilizar o conteúdo por completo.

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=20,
                    length_function=len
                )

                chunk = text_splitter.split_text(text=text)
                store_nome = 'ProcuradoriaGeral'

                # TODO: Para evitar erros de EOFError que está relacionado a sobreescrita e arquivo vazio é necessário fazer primeiro
                # a verificação se o arquivo já existe e se não está vazio, caso a ordem seja outra é possível que o programa fique estagnado em bug.

                if not os.path.exists(f"{store_nome}.pkl"):
                    # TODO: Transformar os chunks em embeddings
                    embeddings = OpenAIEmbeddings()

                    # TODO: Armazenar os embeddings em um banco de vetores
                    vectorStore = FAISS.from_texts(chunk, embedding=embeddings)

                    with open(f"{store_nome}.pkl", "wb") as f:
                        pickle.dump(vectorStore, f)

                else:
                    st.write("O arquivo está vazio")


def conversation():

    # Determinando o comportamento de conversação da IA
    template = """You are an AI assistant for answering questions about the most recent state of Vestibular da Unicamp 2024.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "Hmm, não tenho certeza." Don't try to make up an answer.
    If the question is not about the most recent Vestibular da Unicamp de 2024, politely inform them that you are tuned to only answer questions only about the most recent Vestibular da Unicamp de 2024.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="human_input", return_messages=True)
    # temperatura foi mantida como padrão 0, o motivo: a tarefa ter um aspecto determinístico seguindo o que está escrito no edital
    llm = OpenAI(temperature=0.2)
    chain = load_qa_chain(
        llm=llm, chain_type="stuff", memory=memory, prompt=prompt)
    return chain


def main():
    st.header('Tire suas dúvidas sobre o vestibular da UNICAMP - 2024')

    if query := st.chat_input(
            "Escreva sua pergunta sobre o vestibular", key='widget'):
        st.session_state.messages.append(
            {"role": "user", "content": query})

    add_vertical_space(1)

    data_loaded = load_data()

    if query:
        # o valor de k foi aumentado para 5, pois atribui mais exemplos que auxiliaram obter uma melhor resposta
        docs = data_loaded.similarity_search(query=query, k=5)

        chain = conversation()
        # método get_openai_callback utilizado para cálculo de quanto foi gasto a cada uso da OpenAI API 
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, human_input=query)
            print(cb)

        # Se a última mensagem não é da assistente então gera uma nova resposta
        for message in st.session_state.messages:  # Mostra as mensagens prévias
            with st.chat_message(message["role"]):
                st.write(message["content"])
        # Mostra a mensagem de resposta 
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    # Adiciona a resposta-resposta ao histórico
                    st.session_state.messages.append(message)


#st.markdown(
#    "<p1 style='text-align: left; color: grey;'>Desenvolvido por [Evair](https://github.com/ver0z)</h1>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
