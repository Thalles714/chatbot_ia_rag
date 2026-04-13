import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ================================
# 🔐 CONFIGURAÇÃO
# ================================
load_dotenv()

# OBS: certifique-se que colocou sua chave no .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================
# 📄 CARREGAR DADOS
# ================================
with open("data.txt", "r", encoding="utf-8") as f:
    texto = f.read()

# ================================
# 🧠 CRIAR BANCO VETORIAL
# ================================
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="cardapio")

# ================================
# ✂️ DIVIDIR TEXTO EM PARTES
# ================================
def dividir_texto(texto, tamanho=500):
    return [texto[i:i+tamanho] for i in range(0, len(texto), tamanho)]

chunks = dividir_texto(texto)

# ================================
# 📌 GERAR EMBEDDINGS E SALVAR
# ================================
for i, chunk in enumerate(chunks):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    )
    
    collection.add(
        documents=[chunk],
        embeddings=[emb.data[0].embedding],
        ids=[str(i)]
    )

# ================================
# 🔍 BUSCA (RAG)
# ================================
def buscar_contexto(pergunta):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta
    )

    resultado = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=3
    )

    return "\n".join(resultado["documents"][0])

# ================================
# 🤖 GERAR RESPOSTA
# ================================
def responder(pergunta):
    contexto = buscar_contexto(pergunta)

    prompt = f"""
Você é um atendente de churrascaria focado em delivery.

Use apenas essas informações:
{contexto}

Pergunta:
{pergunta}

Responda de forma:
- Natural
- Direta
- Persuasiva (leve o cliente a pedir)
"""

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return resposta.choices[0].message.content

# ================================
# 💬 LOOP DE CHAT
# ================================
print("🤖 Chatbot iniciado! (digite 'sair' para encerrar)\n")

while True:
    pergunta = input("Você: ")

    if pergunta.lower() == "sair":
        print("Encerrando...")
        break

    resposta = responder(pergunta)

    print(f"Bot: {resposta}\n")