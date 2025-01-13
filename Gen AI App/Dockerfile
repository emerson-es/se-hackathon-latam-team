# Use uma imagem base do Python
FROM python:3.9

# Copie os arquivos para dentro do container
COPY . ./

# Instale as dependências necessárias
RUN pip install -r requirements.txt

# Exponha a porta 8080
EXPOSE 8080

# Defina a variável de ambiente PORT
ENV PORT 8080

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
