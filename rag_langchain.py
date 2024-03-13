
import argparse
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os
from openai import OpenAI
import json
import logging
import time 
import requests
import paho.mqtt.client as mqtt
from weather_api import get_weather_data


import paho.mqtt.client as mqtt
import json
import logging
import time 
from weather_api import get_weather_data

# Define the broker address and topic
broker_address = "mqtt.eclipseprojects.io"
topic = "Spirulina_Edge"
received_data = None

# Define a callback function to handle received messages
def on_message(client, userdata, msg):
    global received_data
    # Decode and print the received message
    received_data = json.loads(msg.payload.decode())
    print("Received data:", received_data)
    




os.environ["TOKENIZERS_PARALLELISM"] = "false" # workaround for HuggingFace/tokenizers

     

def main():
    global received_data
    # Create an MQTT client instance
    client = mqtt.Client()

    # Set the callback function for received messages
    client.on_message = on_message

    # Connect to the MQTT broker
    client.connect(broker_address)

    # Subscribe  to the specified topic
    client.subscribe(topic)
    client.loop_start()
    docs_dir="./handbook/"
    persist_dir="./handbook_faiss"
    embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    print(f"Embedding: {embedding.model_name}")


    if os.path.exists(persist_dir): 
        print(f"Loading FAISS index from {persist_dir}")
        vectorstore = FAISS.load_local(persist_dir, embedding, allow_dangerous_deserialization=True)

        print("done.")
    else:
        print(f"Building FAISS index from documents in {docs_dir}")
        loader = DirectoryLoader(docs_dir,
            loader_cls=Docx2txtLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.docx"  # which files get loaded
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )
        frags = text_splitter.split_documents(docs)

        print(f"Poplulating vector store with {len(docs)} docs in {len(frags)} fragments")
        vectorstore = FAISS.from_documents(frags, embedding)
        print(f"Persisting vector store to: {persist_dir}")
        vectorstore.save_local(persist_dir)
        print(f"Saved FAISS index to {persist_dir}")

    # Be sure your local model suports a large context size for this
    llm = ChatOpenAI(
         #base_url="http://localhost:1234/v1",
        api_key="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",
        temperature=0.6
        
   )
   # Example: reuse your existing OpenAI setup




    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory.load_memory_variables({})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
 
  

       # Start a REPL loop
    while True:
        if received_data:
            user_input = json.dumps(received_data)
            received_data = None  # Reset received_data to None after using it
        else:
            user_input = input("Enter your input: ")
        if user_input=="exit":
            break
        memory.chat_memory.add_user_message(user_input)
        result = qa_chain({"question": user_input})
        response = result["answer"]
        memory.chat_memory.add_ai_message(response)
        print("AI:", response)
          # Obtenir et afficher la recommandation
    #     get_recommendation(response, vectorstore, embedding)
     # Stop MQTT client loop when exiting the while loop
        client.loop_stop()


      

if __name__ == "__main__":
    main()

# import os
# import json
# import time 
# import logging
# import argparse
# import requests
# import paho.mqtt.client as mqtt
# from weather_api import get_weather_data

# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models.openai import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader, Docx2txtLoader
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.faiss import FAISS
    

# class MQTTClient:
#     def __init__(self, broker_address, topic):  #Initialise un client MQTT avec l'adresse du courtier et le sujet spécifiés.
#         self.broker_address = broker_address
#         self.topic = topic
#         self.received_data = None
#         self.client = mqtt.Client()
#         self.client.on_message = self.on_message

#     def on_message(self, client, userdata, msg):
#         self.received_data = json.loads(msg.payload.decode())
#         print("Received data:", self.received_data)

#     def connect(self):
#         self.client.connect(self.broker_address)
#         self.client.subscribe(self.topic)
#         self.client.loop_start()

#     def disconnect(self):
#         self.client.loop_stop()

# class ChatSystem:
#     def __init__(self, embedding_model="all-mpnet-base-v2", api_key="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", temperature=0.6):
#         self.embedding = SentenceTransformerEmbeddings(model_name=embedding_model)
#         self.llm = ChatOpenAI(api_key=api_key, temperature=temperature)
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         self.memory.load_memory_variables({})
#         self.qa_chain = None

#     def build_qa_chain(self, docs_dir="./handbook/", persist_dir="./handbook_faiss"):
#         if os.path.exists(persist_dir): 
#             print(f"Loading FAISS index from {persist_dir}")
#             vectorstore = FAISS.load_local(persist_dir, self.embedding, allow_dangerous_deserialization=True)
#             print("done.")
#         else:
#             print(f"Building FAISS index from documents in {docs_dir}")
#             loader = DirectoryLoader(docs_dir,
#                 loader_cls=Docx2txtLoader,
#                 recursive=True,
#                 silent_errors=True,
#                 show_progress=True,
#                 glob="**/*.docx"  # which files get loaded
#             )
#             docs = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=500,
#                 chunk_overlap=75
#             )
#             frags = text_splitter.split_documents(docs)
#             print(f"Populating vector store with {len(docs)} docs in {len(frags)} fragments")
#             vectorstore = FAISS.from_documents(frags, self.embedding)
#             print(f"Persisting vector store to: {persist_dir}")
#             vectorstore.save_local(persist_dir)
#             print(f"Saved FAISS index to {persist_dir}")

#         self.qa_chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             memory=self.memory,
#             retriever=vectorstore.as_retriever()
#         )

  
#     def chat(self):
    
#         while True:
#             user_input = input("Enter your input: ")
#             if user_input == "exit":
#                     break
#             self.memory.chat_memory.add_user_message(user_input)
#             result = self.qa_chain({"question": user_input})
#             response = result["answer"]
#             self.memory.chat_memory.add_ai_message(response)
#             print("AI:", response)

# def main():
#     broker_address = "mqtt.eclipseprojects.io"
#     topic = "Spirulina_Edge"
#     mqtt_client = MQTTClient(broker_address, topic)
#     mqtt_client.connect()

#     chat_system = ChatSystem()
#     chat_system.build_qa_chain()
#     chat_system.chat()

#     mqtt_client.disconnect()
   

# if __name__ == "__main__":
#     main()
