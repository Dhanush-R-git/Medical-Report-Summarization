from flask_pymongo import PyMongo
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pdf_generator import create_pdf
import logging

app = Flask(__name__)

# Configure MongoDB connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/reportdb"
mongo = PyMongo(app)

# Initialize HuggingFace model
HUGGINGFACEHUB_API_TOKEN = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=1024, temperature=0.3, token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))

# Define the prompt template
prompt_template = """
You are a medical report generator AI.Your task is to create comprehensive medical reports based on the provided patient details and medical history.Follow this structured format for each patient report:
**Patient Details:* - Patient Name:[Insert Patient Name here]-
Patient ID: [Insert Patient ID here]
Joining Date (Reg.Date & Time): [Insert Reg.Date & Time here] 
Test Report: [Insert Test Report details here]
Age/Sex: [Insert Age/Sex here] Impression:
[Insert Impression here]
**Medical History:*
     - Summary: 
           - Relevant past illnesses: [Summarize relevant past illnesses]
             Chronic conditions: [List any chronic conditions]
             Significant medical events: [Mention surgeries or other significant medical events]
**Symptoms and Diagnosis:* - Symptoms: 
Primary symptoms: [Summarize primary symptoms]
Onset and progression: [Mention onset and progression of symptoms]
Diagnosis: - [Provide diagnosis made by healthcare professionals]
**Treatment and Recommendations:* - Treatment: -
Medications:[List medications administered]- Therapies or procedures: [Summarize therapies or procedures performed]- Recommendations: - Follow-up care: [Mention any follow-up care]- Lifestyle adjustments: [Include lifestyle adjustments advised]*Lab Reports:* - Key findings: [Summarize key findings from blood tests, urine analysis, etc. mentioned here]
Significant results: [Highlight significant results or abnormalities]
**Current Status:* - Status: - Current condition:[Describe current condition]- Progress or ongoing issues: [Mention progress or ongoing issues]
above mentioned heading must be in the output and also the summary should be structed and readable way words within 1500 words
Note: if suppose the content not there in the given input i want to show not content is specified in the input file"{context}"
"""

# Initialize PromptTemplate and LLMChain
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context"]
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Route for the login page
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    credentials = {
        'tharunps': {
            '123456': '/upload_doctor_report',
            '456789': '/upload_scan_report',
            '789123': '/upload_blood_report',
            '012345': '/create_patient',
            '001200': '/index'
        }
    }

    if username in credentials and password in credentials[username]:
        return redirect(credentials[username][password])
    else:
        return 'Invalid username or password', 403

# Routes for uploading reports
@app.route('/upload_doctor_report', methods=['GET', 'POST'])
def upload_doctor_report():
    if request.method == 'POST':
        patient_id = request.form['patientId']
        report_file = request.files['reportFile']
        if not patient_id or not report_file:
            return jsonify({'message': 'Patient ID and report file are required'}), 400

        patient = mongo.db.patients.find_one({'patient_id': patient_id})
        if not patient:
            return jsonify({'message': 'Patient ID not found. Please create the patient first.'}), 400

        file_path = os.path.join('uploads', 'doctor_reports', report_file.filename)
        report_file.save(file_path)
        
        mongo.db.patients.update_one(
            {'patient_id': patient_id},
            {'$set': {'doctor_report': file_path}},
            upsert=True
        )
        
        return jsonify({'message': 'Doctor report uploaded successfully'})
    return render_template('upload_doctor_report.html')

@app.route('/upload_scan_report', methods=['GET', 'POST'])
def upload_scan_report():
    if request.method == 'POST':
        patient_id = request.form['patientId']
        report_file = request.files['reportFile']
        if not patient_id or not report_file:
            return jsonify({'message': 'Patient ID and report file are required'}), 400

        patient = mongo.db.patients.find_one({'patient_id': patient_id})
        if not patient:
            return jsonify({'message': 'Patient ID not found. Please create the patient first.'}), 404
        
        file_path = os.path.join('uploads', 'scan_reports', report_file.filename)
        report_file.save(file_path)
        
        mongo.db.patients.update_one(
            {'patient_id': patient_id},
            {'$set': {'scan_report': file_path}},
            upsert=True
        )
        
        return jsonify({'message': 'Scan report uploaded successfully'})
    return render_template('upload_scan_report.html')

@app.route('/upload_blood_report', methods=['GET', 'POST'])
def upload_blood_report():
    if request.method == 'POST':
        patient_id = request.form['patientId']
        report_file = request.files['reportFile']
        if not patient_id or not report_file:
            return jsonify({'message': 'Patient ID and report file are required'}), 400

        patient = mongo.db.patients.find_one({'patient_id': patient_id})
        if not patient:
            return jsonify({'message': 'Patient ID not found. Please create the patient first.'}), 404
        
        file_path = os.path.join('uploads', 'blood_reports', report_file.filename)
        report_file.save(file_path)
        
        mongo.db.patients.update_one(
            {'patient_id': patient_id},
            {'$set': {'blood_report': file_path}},
            upsert=True
        )
        
        return jsonify({'message': 'Blood report uploaded successfully'})
    return render_template('upload_blood_report.html')

@app.route('/create_patient', methods=['GET', 'POST'])
def create_patient():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        address = request.form['address']
        phone = request.form['phone']
        
        patient_id = mongo.db.patients.count_documents({}) + 1
        patient_id = f'PAT{patient_id:04}'
        
        mongo.db.patients.insert_one({
            'patient_id': patient_id,
            'name': name,
            'email': email,
            'address': address,
            'phone': phone,
            'doctor_report': None,
            'scan_report': None,
            'blood_report': None
        })
        
        return jsonify({'message': 'Patient created successfully', 'patient_id': patient_id})
    return render_template('create_patient.html')

# Upload and summary route
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

logging.basicConfig(level=logging.DEBUG)

@app.route('/upload', methods=['POST'])
def upload_file():
    patient_id = request.form.get('patientId')
    if not patient_id:
        return jsonify({'message': 'Patient ID is required'}), 400

    patient = mongo.db.patients.find_one({'patient_id': patient_id})
    if not patient:
        return jsonify({'message': 'Patient ID not found. Please create the patient first.'}), 404

    doctor_report_path = patient.get('doctor_report')
    scan_report_path = patient.get('scan_report')
    blood_report_path = patient.get('blood_report')

    documents = []

    def load_pdf_documents(file_path):
        try:
            loader = PyPDFLoader(file_path)
            return loader.load_and_split()
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return []

    if doctor_report_path:
        documents.extend(load_pdf_documents(doctor_report_path))

    if scan_report_path:
        documents.extend(load_pdf_documents(scan_report_path))

    if blood_report_path:
        documents.extend(load_pdf_documents(blood_report_path))

    if not documents:
        return jsonify({'message': 'No reports found for the given patient ID.'}), 404

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.debug("Embeddings model loaded successfully.")
    except Exception as e:
        logging.error(f"Error initializing embeddings: {e}")
        return jsonify({'message': 'Error with embeddings model.'}), 500
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    
    try:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./database_db1"
        )
        vectordb.persist()
        logging.debug("Vector database created and persisted successfully.")
    except Exception as e:
        logging.error(f"Error creating or persisting vector database: {e}")
        return jsonify({'message': 'Error with vector database.'}), 500

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        response = qa_chain("Create a full medical summary that covers the patient's history, diagnosis, treatment, and current status")
        pdf_summary = response["result"]

        pdf_path = os.path.join('uploads', 'summary.pdf')
        create_pdf(pdf_summary, pdf_path)
        logging.debug("Summary PDF generated successfully.")
    except Exception as e:
        logging.error(f"Error in QA chain or PDF generation: {e}")
        return jsonify({'message': 'Error generating summary.'}), 500
    
    return render_template('index.html', summary=pdf_summary)

@app.route('/download')
def download_file():
    return send_file('uploads/summary.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
