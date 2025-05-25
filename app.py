from flask_pymongo import PyMongo
from flask import Flask, request, render_template, send_file, redirect, session, url_for, jsonify
import os, smtplib
import pdfplumber
from email.message import EmailMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pdf_generator import create_pdf
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')  # Replace with a strong secret key

# Configure MongoDB connection
app.config["MONGO_URI"] = os.getenv('MONGO_URI')
mongo = PyMongo(app)

# Initialize HuggingFace model
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Mail config
app.config.update(
    MAIL_SERVER   = os.getenv('MAIL_SERVER', 'localhost'),
    MAIL_PORT     = int(os.getenv('MAIL_PORT', 25)),
    MAIL_USERNAME = os.getenv('MAIL_USERNAME'),
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD'),
    MAIL_USE_TLS  = os.getenv('MAIL_USE_TLS', 'False').lower() in ('1','true','yes')
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="summarization",
    max_length=1024, 
    temperature=0.3, 
    token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )

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

# --- Authentication & session management ---
# Route for the login page
@app.route('/', methods=['GET'])
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    credentials = {
        'medsum': {
            '123456': '/upload_doctor_report',
            '456789': '/upload_scan_report',
            '789123': '/upload_blood_report',
            '012345': '/create_patient',
            '001200': '/index'
        }
    }

    if username in credentials and password in credentials[username]:
        session['logged_in'] = True
        session['user'] = username
        session['role'] = 'doctor'  # only doctors reach these routes
        return redirect(credentials[username][password])
    else:
        return 'Invalid username or password', 403
    
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

def doctor_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in') or session.get('role') != 'doctor':
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated


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
@doctor_required
def index():
    # fetch all experts to show in panel
    experts = list(mongo.db.experts.find({}, {'_id': 0}))
    return render_template('index.html', experts=experts, summary=None, patient_id='')

logging.basicConfig(level=logging.DEBUG)

@app.route('/upload', methods=['POST'])
@doctor_required
def upload_file():
    experts = list(mongo.db.experts.find({}, {'_id': 0}))
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

        pdf_path = os.path.join('uploads', f'summary_{patient_id}_{int(datetime.utcnow().timestamp())}.pdf')
        create_pdf(pdf_summary, pdf_path)
        logging.debug("Summary PDF generated successfully.")
    except Exception as e:
        logging.error(f"Error in QA chain or PDF generation: {e}")
        return jsonify({'message': 'Error generating summary.'}), 500
    # persist summary record in DB
    mongo.db.summaries.insert_one({
        'patient_id': patient_id,
        'summary': pdf_summary,
        'pdf_path': pdf_path,
        'timestamp': datetime.utcnow()
    })
    
    return render_template('index.html', summary=pdf_summary, experts=experts, patientId=patient_id)

@app.route('/download/<path:filename>')
@doctor_required
def download_file(filename):
    return send_file(filename, as_attachment=True)

# --- New Medical Summaries page ---
@app.route('/medical_summaries', methods=['GET', 'POST'])
@doctor_required
def medical_summaries():
    summaries = None
    if request.method == 'POST':
        pid = request.form.get('patientId')
        raw = mongo.db.summaries.find({'patient_id': pid}).sort('timestamp', -1)

        summaries = []
        for s in raw:
            ts = s.get('timestamp')
            if ts:
                date_str = ts.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = 'N/A'
            summaries.append({
                'date': date_str,
                'excerpt': s.get('summary', '')[:100] + ('…' if s.get('summary') and len(s.get('summary')) > 100 else ''),
                'pdf_path': s.get('pdf_path', '')
            })

    return render_template('medical_summaries.html', summaries=summaries)

@app.route('/send_expert_email', methods=['POST'])
@doctor_required
def send_expert_email():
    data         = request.json or {}
    patient_id   = data.get('patientId')
    expert_email = data.get('expertEmail')
    summary_text = data.get('summaryText')

    if not (patient_id and expert_email and summary_text):
        return jsonify({'message': 'Missing data'}), 400

    # Build email
    msg = EmailMessage()
    msg['Subject'] = f"MediSum: Analysis request for Patient {patient_id}"
    msg['From']    = app.config['MAIL_USERNAME'] or 'no-reply@medisum.com'
    msg['To']      = expert_email
    msg.set_content(
        f"Dear Specialist,\n\n"
        f"You have been requested to review the medical summary for patient {patient_id}.\n\n"
        f"--- Summary ---\n{summary_text}\n\n"
        f"Please reply with your expert analysis.\n\n"
        "Regards,\nMediSum Team"
    )

    try:
        # Connect to real SMTP server
        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'], timeout=10)
        if app.config['MAIL_USE_TLS']:
            server.starttls()
        if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)
        server.quit()
        return jsonify({'message': 'Email sent successfully'})
    except Exception as e:
        logging.error(f"Email error: {e}")
        return jsonify({'message': 'Failed to send email: ' + str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)