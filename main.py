from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from gtts import gTTS
from dotenv import load_dotenv

import speech_recognition as sr
import pyttsx3
import platform
import os
import tempfile
import subprocess
import sys

# Caminho do índice a ser criado ou consultado
INDEX_PATH = "manual_index"

# Carrega .env e resgata a chave
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Checa se o índice ja existe e evita criar ele novamente
if os.path.exists(INDEX_PATH):
    print("Carregando índice existente...")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("Gerando índice FAISS...")
    loader = PyPDFLoader("manual_fastback.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    ),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def ouvir_comando():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n Fale agora:")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        audio = recognizer.listen(source)

    try:
        texto = recognizer.recognize_google(audio, language="pt-BR")
        print(f"Você disse: {texto}")
        return texto
    except sr.UnknownValueError:
        print("Não entendi o que foi dito.")
        return ""
    except sr.RequestError as error:
        print(f"Erro no serviço de reconhecimento: {error}")
        return ""
    
def responder_pergunta(pergunta):
    print("Buscando resposta...")
    resposta = qa_chain.invoke({"query": pergunta})
    return resposta["result"]

def falar(texto):
    print(f"Resposta: {texto}\n")

    try:
        # Cria um arquivo temporário com o áudio
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
            # Uso do gTTS para a voz soar mais natural
            tts = gTTS(text=texto, lang="pt-br")
            tts.save(temp_audio.name)

            # Tenta tocar com mpg123 (mais comum no Ubuntu)
            try:
                subprocess.run(
                    ["mpg123", "-q", "-d", "1", temp_audio.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except FileNotFoundError:
                # Fallback: tenta com ffplay (instalado com ffmpeg)
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-af", "atempo=1.5", temp_audio.name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

    except Exception as e:
        print(f"[ERRO NO TTS] {e}")

if __name__ == "__main__":
    print("\nAgente inteligente pronto! Diga algo como:")
    print("'Como trocar o óleo do motor?' ou 'Qual é a capacidade do tanque?'")
    print("Diga 'sair' para encerrar.\n")

    while True:
        comando = ouvir_comando()
        if not comando:
            continue

        if comando.lower() in ["sair", "encerrar", "parar", "tchau"]:
            falar("Encerrando. Até mais!")
            print("Encerrando...")
            break

        resposta = responder_pergunta(comando)
        falar(resposta)


