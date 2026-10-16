# Chatbot de cardápio com RAG — protótipo de estudo

Aplicação de terminal que consulta um cardápio em `data.txt`, recupera trechos com ChromaDB e usa a API OpenAI para gerar uma resposta. Projeto educacional, sem interface web ou autenticação de usuários.

## Executar

Requer Python 3.11 ou 3.12 e acesso à API OpenAI.

```sh
python -m venv .venv
```

Ative o ambiente virtual antes de instalar as dependências: no Windows, `.venv\Scripts\activate`; no macOS/Linux, `source .venv/bin/activate`.

```sh
python -m pip install -r requirements.txt
```

Copie `.env.example` para `.env`, preencha `OPENAI_API_KEY` e revise `data.txt`. Execute:

```sh
python app.py
```

O conteúdo e o `.env` são resolvidos a partir da pasta do script, independentemente do diretório do terminal. Digite `sair` ou encerre com Ctrl+C/Ctrl+D.

## Funcionamento e limites

- Ao executar o programa, os trechos do arquivo são enviados à OpenAI para embeddings. O índice Chroma é temporário e é reconstruído a cada execução.
- Cada pergunta envia uma consulta para embeddings e os trechos recuperados para a geração. Essas chamadas podem gerar cobrança; consulte os limites da sua conta.
- Não use documentos ou dados pessoais sem autorização. A presença de um índice local não significa que os dados ficam apenas no computador.
- As regras ficam em uma mensagem de sistema e os documentos/perguntas em uma mensagem de usuário. Isso não garante resistência completa a prompt injection nem precisão factual.
- Importar `app.py` não realiza chamadas de rede. O protótipo não deve ser publicado como API aberta sem controles adicionais.

## Testes offline

```sh
python -m unittest discover -s tests -v
```

Os testes substituem os clientes externos e não consomem créditos. Verificam recuperação em coleção pequena/vazia, separação de mensagens e divisão do conteúdo.
