from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from flask_cors import CORS
import pickle
import io
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://pdf-chat-bot-mocha.vercel.app"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/members")

def members():
    return "hiiii"



@app.route('/extract_text', methods=['POST'])

def extract_text():
    uploaded_file = request.files.get('file')

    if uploaded_file is None:
        return jsonify({'error': 'No file selected'}), 400

    # Ensure the file is a PDF
    if not uploaded_file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400

    try:
        # Read the PDF file content
        pdf_content = uploaded_file.read()

        # Create a BytesIO object to make it file-like
        pdf_file = io.BytesIO(pdf_content)

        # Process the PDF buffer using PdfReader
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Return the text as JSON
        return jsonify({'text': text})
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/give_ans', methods=['POST'])
def give_ans():
    try:
        data = request.get_json()

        # Extract the necessary data from the request
        pdf_text = data.get('pdfText', '')
        input_text = data.get('input_text', '')
        
        # Your existing logic for processing PDF and generating a response
        # I'm assuming your original process_pdf function is loaded from a separate module
        # Replace the following line with your actual processing logic
        response = process_pdf(pdf_text, input_text)

        return jsonify({'result': response})
    except Exception as e:
        print(f"Error processing input: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def process_pdf(text, query):
    load_dotenv()
    
    # Your existing logic for processing the PDF text and generating a response
    # I'm assuming you have a function like this; replace the following lines with your actual logic
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)

    store_name = "your_store_name"  # Replace with your logic to generate a store name

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI(model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)

    return response
# def process_pdf(pdf_path):

#     load_dotenv()
#     with open(pdf_path, 'rb') as pdf:
#         pdf_reader = PdfReader(pdf)        
        
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
        
#         text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 length_function=len
#                 )
#         chunks = text_splitter.split_text(text=text)


#         store_name = pdf.name[:-4]

#         embeddings = OpenAIEmbeddings()
#         VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
#         with open(f"{store_name}.pkl", "wb") as f:
#             pickle.dump(VectorStore, f)


#         query="what is the Date of audit and withdrawl?"
#         if query:
#             docs = VectorStore.similarity_search(query=query, k=3)

#             llm = OpenAI()
#             chain = load_qa_chain(llm=llm, chain_type="stuff")
#             with get_openai_callback() as cb:
#                 response = chain.run(input_documents=docs, question=query)
#                 print(cb)
            

    
#     return response



    
