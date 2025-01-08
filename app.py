### Importar libs

# Bibliotecas padrão do Python
import os
import json
import re
import subprocess
import time
import typing_extensions
import urllib
from datetime import datetime, timedelta
from html import unescape
from timeit import default_timer as timer
from typing import List, Tuple
import requests

# Bibliotecas de análise de dados e machine learning
import numpy as np
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Bibliotecas para gráficos e interfaces
import altair as alt
import gradio as gr
import streamlit as st
#from streamlit import chat_message as message

# Bibliotecas relacionadas ao Google Colab e Cloud
from google.cloud import aiplatform
from google.colab import auth as google_auth
from google.colab import files

# Trabalho com Neo4j e Graph Data Science
from graphdatascience import GraphDataScience
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector, SearchType

# Bibliotecas de processamento de linguagem natural e IA
from langchain.chains import GraphCypherQAChain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from langchain.llms import VertexAI, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate,)
from langchain.callbacks import get_openai_callback
from tqdm import tqdm
import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.language_models import TextGenerationModel

# Configuração de ambiente
from dotenv import load_dotenv
import IPython

# Imprime o diretório de trabalho atual
print("Diretório de trabalho atual:", os.getcwd())

# Ativa arquivo de configuracao
env_file = '/content/ws.env'
load_dotenv('es.env', override=True)

# Adiciona Configuracoes
# Neo4j
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

aura_ds_value = os.getenv('AURA_DS')
if aura_ds_value is not None:
  AURA_DS = aura_ds_value.lower() == 'true'
else:
  AURA_DS = False

# AI
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
LLM_CYPHER = os.getenv('LLM_CYPHER')
LLM_QA = os.getenv('LLM_QA')
GPROJECT_ID = os.getenv('GPROJECT_ID')
GLOCATION = os.getenv('GLOCATION')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

vertexai.init(project=GPROJECT_ID, location=GLOCATION)

def load_cypher_llm(LLM_CYPHER: str):
    if LLM_CYPHER == "gpt-4":
        print("LLM_CYPHER: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif LLM_CYPHER == "gpt-4-turbo":
        print("LLM_CYPHER: Using GPT-4-Turbo")
        return ChatOpenAI(temperature=0, model_name="gpt-4-turbo", streaming=True)
    elif LLM_CYPHER == "gpt-4o":
        print("LLM_CYPHER: Using GPT-4o")
        return ChatOpenAI(temperature=0, model_name="gpt-4o", streaming=True)
    elif LLM_CYPHER == "gpt-3.5-turbo":
        print("LLM_CYPHER: Using GPT-3.5 turbo")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif LLM_CYPHER == "code-bison@001":
        print("LLM_CYPHER: Using Google code-bison@001")
        return VertexAI(temperature=0, model_name=LLM_CYPHER, max_output_tokens=2048)
    print("LLM_CYPHER: Using GPT-3.5")
    #Caso nada fornecido
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

cypher_llm = load_cypher_llm(LLM_CYPHER)

def load_qa_llm(LLM_QA: str):
    if LLM_QA == "gpt-4":
        print("LLM_QA: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    if LLM_QA == "gpt-4-turbo":
        print("LLM_QA: Using GPT-4-Turbo")
        return ChatOpenAI(temperature=0, model_name="gpt-4-turbo", streaming=True)
    if LLM_QA == "gpt-4o":
        print("LLM_QA: Using GPT-4o")
        return ChatOpenAI(temperature=0, model_name="gpt-4o", streaming=True)
    elif LLM_QA == "gpt-3.5-turbo":
        print("LLM_QA: Using GPT-3.5 Turbo")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif LLM_QA == "text-bison":
        print("LLM_QA: Using Text Bison")
        return VertexAI(temperature=0, model_name='text-bison', max_output_tokens=2048)
    print("LLM_QA: Using GPT-3.5 Turbo")
    #Caso nada fornecido
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

qa_llm = load_qa_llm(LLM_QA)

CYPHER_GENERATION_TEMPLATE = """Você é um especialista em tradução de Cypher para Neo4j que entende a pergunta em português do Brasil e a converte para Cypher estritamente baseado no esquema Neo4j fornecido e seguindo as instruções abaixo:
1. Gere uma consulta Cypher compatível APENAS para a versão 5 do Neo4j
2. Não use as palavras-chave EXISTS, SIZE no Cypher. Use alias ao usar a palavra-chave WITH
3. Por favor, não use os mesmos nomes de variáveis para diferentes nós e relações na consulta.
4. Use apenas nós e relações mencionados no esquema
5. Sempre envolva a saída do Cypher dentro de 3 crases
6. Sempre faça uma busca insensível a maiúsculas/minúsculas e difusa por qualquer propriedade relacionada à pesquisa. Ex.: para buscar um nome de empresa, use `toLower(c.name) contains 'neo4j'`
8. Sempre use aliases para se referir ao nó na consulta
9. 'Resposta' NÃO é uma palavra-chave do Cypher. Resposta nunca deve ser usada em uma consulta.
10. Por favor, gere apenas uma consulta Cypher por pergunta.
11. Cypher NÃO é SQL. Portanto, não misture e combine as sintaxes.
12. Toda consulta Cypher sempre começa com a palavra-chave MATCH.

Schema:
{schema}

Samples:
Question: Quantos clients existem hoje na base de dados no total ?
Answer: MATCH (c:Client)
RETURN count(c)

Question: Qual a quantidade de clients por fabricante na base de dados ?
Answer: MATCH (c:Client)
RETURN c.clientManufac AS Fabricante, COUNT(*) AS Quantity

Question: Como posso obter o nome canônico do cliente, a chave do client o fabricante, o último BSSID conectado, a força do sinal, o total de snapshots incluindo o primeiro e o último, e a data do último snapshot para cada client, ordenados pelo timestamp do último snapshot?
Answer: MATCH (client:Client)-[:ASSOCIATED_TO_CLIENT_SNAPSHOT_FIRST]->(firstSnapshot:`Client Snapshot`)
MATCH (client)-[:ASSOCIATED_TO_CLIENT_SNAPSHOT_LAST]->(lastSnapshot:`Client Snapshot`)
OPTIONAL MATCH (firstSnapshot)-[r:NEXT_CLIENT_SNAPSHOT*]->(intermediateSnapshot)
WITH client, lastSnapshot,
     COUNT(DISTINCT intermediateSnapshot) AS totalIntermediateSnapshots
RETURN
    client.clientCname AS ClientCanonicalName,
    client.clientKey AS ClientKey,
    client.clientManufac AS Manufacturer,
    lastSnapshot.csnapLbssid AS LastBSSID,
    lastSnapshot.csnapSignalstr AS SignalStrength,
    totalIntermediateSnapshots + 1 AS TotalSnapshots,  // +1 to include the last snapshot in the count
    lastSnapshot.csnapTimestamp AS LastSnapshotTimestamp
ORDER BY LastSnapshotTimestamp DESC;

Question: Quantos access points existem hoje na base de dados no total ?
Answer: MATCH (ap:`Access Point`)
return count(ap)

Question: Qual a quantidade de access points por fabricante na base de dados ?
Answer: MATCH (ap:`Access Point`)
RETURN ap.apManufac AS Fabricante, COUNT(*) AS Quantity

Question: Como posso obter o nome canônico do access point, a chave do access point o fabricante, o último BSSID conectado, a força do sinal, o tipo de criptografia, o total de snapshots incluindo o primeiro e o último, e a data do último snapshot para cada access point, ordenados pelo timestamp do último snapshot?
Answer: MATCH (ap:`Access Point`)-[:ASSOCIATED_TO_ACCESS_POINT_SNAPSHOT_FIRST]->(firstSnapshot:`Access Point Snapshot`)
MATCH (ap)-[:ASSOCIATED_TO_ACCESS_POINT_SNAPSHOT_LAST]->(lastSnapshot:`Access Point Snapshot`)
OPTIONAL MATCH (firstSnapshot)-[r:NEXT_ACCESS_POINT_SNAPSHOT*]->(intermediateSnapshot)
// Adiciona uma etapa para buscar o apCrypt do último snapshot
OPTIONAL MATCH (lastSnapshot)-[:ASSOCIATED_TO_ACCESS_POINT_SECURITY_POSTURE]->(apSecPosture:`Access Point Security Posture`)
WITH ap, lastSnapshot, apSecPosture,
     COUNT(DISTINCT intermediateSnapshot) AS totalIntermediateSnapshots
RETURN
    ap.apCname AS ApCanonicalName,
    ap.apKey AS ApKey,
    ap.apManufac AS Manufacturer,
    lastSnapshot.apsnapLbssid AS LastBSSID,
    lastSnapshot.apsnapSignalstr AS SignalStrength,
    apSecPosture.apCrypt AS Encryption, // Adiciona a coluna de Encryption
    totalIntermediateSnapshots + 1 AS TotalSnapshots,  // +1 to include the last snapshot in the count
    lastSnapshot.apsnapTimestamp AS LastSnapshotTimestamp
ORDER BY LastSnapshotTimestamp DESC;

Question: Quantos WIFI devices existem hoje na base de dados no total ?
Answer: MATCH (device:`WIFI Device`)
return count(device)

Question: Qual a quantidade de WIFI devices por fabricante na base de dados ?
Answer: MATCH (device:`WIFI Device`)
RETURN device.deviceManufac AS Fabricante, COUNT(*) AS Quantity

Question: Como posso obter o nome canônico do WIFI Device, a chave do dispositivo, o fabricante, o último BSSID ao qual foi conectado, a força do sinal, o total de snapshots do dispositivo (incluindo o primeiro e o último), e a data do último snapshot, ordenados pelo timestamp do último snapshot?
Answer: MATCH (device:`WIFI Device`)-[:ASSOCIATED_TO_WIFI_DEVICE_SNAPSHOT_FIRST]->(firstSnapshot:`WIFI Device Snapshot`)
MATCH (device)-[:ASSOCIATED_TO_WIFI_DEVICE_SNAPSHOT_LAST]->(lastSnapshot:`WIFI Device Snapshot`)
OPTIONAL MATCH (firstSnapshot)-[r:NEXT_WIFI_DEVICE_SNAPSHOT*]->(intermediateSnapshot)
WITH device, lastSnapshot,
     COUNT(DISTINCT intermediateSnapshot) AS totalIntermediateSnapshots
RETURN
    device.deviceCname AS DeviceCanonicalName,
    device.deviceKey AS DeviceKey,
    device.deviceManufac AS Manufacturer,
    lastSnapshot.devicesnapLbssid AS LastBSSID,
    lastSnapshot.devicesnapSignalstr AS SignalStrength,
    totalIntermediateSnapshots + 1 AS TotalSnapshots,  // +1 to include the last snapshot in the count
    lastSnapshot.devicesnapTimestamp AS LastSnapshotTimestamp
ORDER BY LastSnapshotTimestamp DESC;

Question: Quantos WIFI Bridge existem hoje na base de dados no total ?
Answer: MATCH (brd:`WIFI Bridge`)
return count(brd)

Question: Qual a quantidade de WIFI Bridge por fabricante na base de dados ?
Answer: MATCH (Bridge:`WIFI Bridge`)
RETURN Bridge.brdManufac AS Fabricante, COUNT(*) AS Quantity

Question: Como posso obter o nome canônico da WIFI Bridge, a chave da bridge, o fabricante, o último BSSID ao qual foi conectada, a força do sinal, o total de snapshots da bridge (incluindo o primeiro e o último), e a data do último snapshot, ordenados pelo timestamp do último snapshot?
Answer: MATCH (Bridge:`WIFI Bridge`)-[:ASSOCIATED_TO_WIFI_BRIDGE_SNAPSHOT_FIRST]->(firstSnapshot:`WIFI Bridge Snapshot`)
MATCH (Bridge)-[:ASSOCIATED_TO_WIFI_BRIDGE_SNAPSHOT_LAST]->(lastSnapshot:`WIFI Bridge Snapshot`)
OPTIONAL MATCH (firstSnapshot)-[r:NEXT_WIFI_BRIDGE_SNAPSHOT*]->(intermediateSnapshot)
WITH Bridge, lastSnapshot,
     COUNT(DISTINCT intermediateSnapshot) AS totalIntermediateSnapshots
RETURN
    Bridge.brdCname AS BridgeCanonicalName,
    Bridge.brdKey AS BridgeKey,
    Bridge.brdManufac AS Manufacturer,
    lastSnapshot.brdsnapLbssid AS LastBSSID,
    lastSnapshot.brdsnapSignalstr AS SignalStrength,
    totalIntermediateSnapshots + 1 AS TotalSnapshots,  // +1 to include the last snapshot in the count
    lastSnapshot.brdsnapTimestamp AS LastSnapshotTimestamp
ORDER BY LastSnapshotTimestamp DESC;

Question: Quantos SSID existem hoje na base de dados no total ?
Answer: MATCH (ssid:SSID)
return count(ssid)

Question: Como posso obter o nome, o hash, o tipo de criptografia do SSID, junto com o total de dispositivos que estão sondando, anunciando e respondendo ao SSID, e a data do último snapshot de SSID, ordenados pelo timestamp do último snapshot?
Answer: MATCH (ssid:SSID)-[:ASSOCIATED_TO_SSID_SNAPSHOT_LAST]->(lastSnapshot:`SSID Snapshot`)
OPTIONAL MATCH (lastSnapshot)<-[:PROBING_FOR_SSID_WIFI_DEVICE|:PROBING_FOR_SSID_CLIENT|:PROBING_FOR_SSID_WIFI_BRIDGE|:PROBING_FOR_SSID_AP]-(probingDevice)
OPTIONAL MATCH (lastSnapshot)<-[:ADVERTISING_FOR_SSID_WIFI_DEVICE|:ADVERTISING_FOR_SSID_CLIENT|:ADVERTISING_FOR_SSID_WIFI_BRIDGE|:ADVERTISING_FOR_SSID_AP]-(advertisingDevice)
OPTIONAL MATCH (lastSnapshot)<-[:RESPONDING_FOR_SSID_WIFI_DEVICE|:RESPONDING_FOR_SSID_CLIENT|:RESPONDING_FOR_SSID_WIFI_BRIDGE|:RESPONDING_FOR_SSID_AP]-(respondingDevice)
WITH ssid, lastSnapshot,
     COUNT(DISTINCT probingDevice) AS TotalProbingDevices,
     COUNT(DISTINCT advertisingDevice) AS TotalAdvertisingDevices,
     COUNT(DISTINCT respondingDevice) AS TotalRespondingDevices
RETURN
    ssid.ssidName AS SSIDName,
    ssid.ssidHash AS SSIDHash,
    ssid.ssidEncryptionitem AS SSIDEncryption,
    TotalProbingDevices,
    TotalAdvertisingDevices,
    TotalRespondingDevices,
    lastSnapshot.ssidsnapTimestamp AS LastSnapshotTimestamp
ORDER BY LastSnapshotTimestamp DESC;

Question: Quantos grupos de comunidade diferentes de tudo relacionado a client, ssid, wifi device, wifi bridge e ssids existem hoje na base de dados no total ?
Answer: MATCH (n)
WHERE n.wccFullvalue IS NOT NULL
RETURN COUNT(DISTINCT n.wccFullvalue) AS distinctWccValues

Question: Qual a quantidade de grupos de comunidades diferentes por grupo de comunidade tem na base de dados ?
Answer: MATCH (n)
WHERE n.wccFullvalue IS NOT NULL
RETURN n.wccFullvalue AS Value, COUNT(n) AS NodeCount
ORDER BY NodeCount DESC

Question: Quantas redes inseguras relacionadas a access point eu tenho no total em minha base de dados ?
Answer: MATCH (ap:`Access Point`)
-[:ASSOCIATED_TO_ACCESS_POINT_SNAPSHOT_FIRST|ASSOCIATED_TO_ACCESS_POINT_SNAPSHOT_LAST]
->(apsnap:`Access Point Snapshot`)
-[:`ASSOCIATED_TO_ACCESS_POINT_SECURITY_POSTURE`]
->(aps:`Access Point Security Posture`)
WHERE aps.apCrypt IN ['WEP', 'Open']
RETURN aps.apCrypt AS EncryptionType, COUNT(DISTINCT ap) AS AccessPointCount

Question: Qual e a lista de redes inseguras relacionadas a access point que eu tenho em minha base de dados ?
Answer: MATCH (ap:`Access Point`)
-[:ASSOCIATED_TO_ACCESS_POINT_SNAPSHOT_FIRST]
->(firstSnapshot:`Access Point Snapshot`)
-[:ASSOCIATED_TO_ACCESS_POINT_SECURITY_POSTURE]
->(apFirstSecPosture:`Access Point Security Posture`)
MATCH (ap)-[:ASSOCIATED_TO_ACCESS_POINT_SNAPSHOT_LAST]
->(lastSnapshot:`Access Point Snapshot`)
-[:ASSOCIATED_TO_ACCESS_POINT_SECURITY_POSTURE]
->(apSecPosture:`Access Point Security Posture`)
WHERE apSecPosture.apCrypt IN ['WEP', 'Open']
OPTIONAL MATCH (firstSnapshot)-[r:NEXT_ACCESS_POINT_SNAPSHOT*]
->(intermediateSnapshot)
WITH ap, firstSnapshot, lastSnapshot, apSecPosture,
     COUNT(DISTINCT intermediateSnapshot) AS totalIntermediateSnapshots
RETURN
    ap.apCname AS ApCanonicalName,
    ap.apKey AS ApKey,
    ap.apManufac AS Manufacturer,
    lastSnapshot.apsnapLbssid AS LastBSSID,
    lastSnapshot.apsnapSignalstr AS SignalStrength,
    apSecPosture.apCrypt AS Encryption, // Encryption type of the last snapshot
    totalIntermediateSnapshots + 1 AS TotalSnapshots, // +1 to include the first snapshot in the count
    lastSnapshot.apsnapTimestamp AS LastSnapshotTimestamp
ORDER BY LastSnapshotTimestamp DESC;

Question: {question}
Answer:
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=['schema','question'], validate_template=True, template=CYPHER_GENERATION_TEMPLATE
)

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
    )

### STREAMLIT

def query_graph(user_input):
    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        verbose=True,
        return_intermediate_steps=True
        )
    result = chain(user_input)
    return result


st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])

with title_col:
    st.title("Cybersecurity LLM Chatbot")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

user_input = st.text_input("Entre sua pergunta", key="input")

cypher_query = None
database_results = None

if user_input:
    with st.spinner("Processando sua pergunta ..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        try:
            result = query_graph(user_input)

            intermediate_steps = result["intermediate_steps"]
            cypher_query = intermediate_steps[0]["query"]
            database_results = intermediate_steps[1]["context"]

            answer = result["result"]
            st.session_state.system_msgs.append(answer)
        except Exception as e:
            st.write("Falha em processar a pergunta. Por favor tente novamente.")
            print(e)

    st.write(f"Tempo Decorrido: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Display the chat history
    #with col1:
    #    if st.session_state["system_msgs"]:
    #        for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
    #            message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
    #            message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col1:
        if "system_msgs" in st.session_state and st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                st.write("Assistant: " + st.session_state["system_msgs"][i], key=str(i) + "_assistant")
                st.write("User: " + st.session_state["user_msgs"][i], key=str(i) + "_user", is_user=True)

    with col2:
        if cypher_query:
            st.text_area("Ultima Query Cypher", cypher_query, key="_cypher", height=240)

    with col3:
        if database_results:
            st.text_area("Ultimos resultados do Neo4j DB", database_results, key="_database", height=240)