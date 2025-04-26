import os
import uuid
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, session, abort
from flask_sqlalchemy import SQLAlchemy
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

MUNICIPAL_COMPLAINT_URL = "https://www.yourmunicipalcorporation.gov.in/complaints"

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'supersecretkey')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///complaints.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SANITATION_KEYWORDS = {
    'en': ['garbage', 'trash', 'waste', 'sewage', 'drainage', 'pollution'],
    'hi': ['कचरा', 'गंदगी', 'नाली', 'सीवेज', 'प्रदूषण'],
    'ta': ['குப்பை', 'கழிவு', 'கழிவுநீர்', 'மாசுபாடு'],
    'bn': ['আবর্জনা', 'বর্জ্য', 'নিকাশী', 'দূষণ'],
    'ur': ['کوڑا', 'گندگی', 'سیوریج', 'آلودگی']
}

sanitation_map = {483: 'garbage', 568: 'sewage', 605: 'waste', 849: 'pollution'}

DECOMPOSABLE_MAP = {
    937: 'green', 954: 'green', 970: 'green', 951: 'green',
    927: 'green', 949: 'green', 966: 'green', 987: 'green', 991: 'green'
}

NON_DECOMPOSABLE_MAP = {
    566: 'blue', 504: 'blue', 654: 'blue', 717: 'blue',
    883: 'blue', 892: 'blue', 479: 'blue', 507: 'blue', 605: 'blue'
}

class Complaint(db.Model):
    id = db.Column(db.String(8), primary_key=True)
    issue = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    language = db.Column(db.String(10), nullable=False)
    complaint_text = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default="Filed")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    bin_type = db.Column(db.String(10), default="unknown")
    waste_type = db.Column(db.String(20), default="unknown")

def analyze_image(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    _, pred = torch.max(output, 1)
    class_id = pred.item()
    bin_type = NON_DECOMPOSABLE_MAP.get(class_id, DECOMPOSABLE_MAP.get(class_id, 'unknown'))
    waste_type = 'decomposable' if bin_type == 'green' else 'non-decomposable'
    if bin_type == 'unknown':
        waste_type = 'unknown'
    return {
        'detected_issue': sanitation_map.get(class_id, 'other'),
        'bin_type': bin_type,
        'waste_type': waste_type,
        'class_id': class_id
    }

def process_text_helper(text, default_lang='en'):
    lower = text.lower()
    for lang, keywords in SANITATION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return {'intent': 'file_complaint', 'language': lang}
    return {'intent': 'unknown', 'language': default_lang}

@app.route('/')
def index():
    from flask import render_template_string
    return render_template_string(INDEX_HTML)

@app.route('/process_text', methods=['POST'])
def process_text_route():
    data = request.get_json() or {}
    result = process_text_helper(data.get('text',''))
    session['detected_issue'] = result['intent']
    return jsonify(result)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['image']
    fname = secure_filename(file.filename)
    uid = uuid.uuid4().hex[:8]
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', f"{uid}_{fname}")
    file.save(path)
    analysis = analyze_image(path)
    session.update({
        'detected_issue': analysis['detected_issue'],
        'bin_type': analysis['bin_type'],
        'waste_type': analysis['waste_type']
    })
    return jsonify(analysis)

@app.route('/set_location', methods=['POST'])
def set_location():
    data = request.get_json() or {}
    session['location'] = data.get('location','')
    return jsonify({'status': 'ok'})

@app.route('/generate_application', methods=['POST'])
def generate_application():
    loc = session.get('location', '___________')
    issue = session.get('detected_issue', 'accumulation of garbage')
    template = f"""
To  
The Municipal Commissioner,  
[Name of Municipal Corporation]  
[City Name]  

Subject: Complaint regarding {issue} at {loc}

Respected Sir/Madam,

I, ______________________________________ (Name),  
residing at ______________________________________ (Full Address),  
Mobile No.: ______________________,  
Email: __________________________,  

wish to bring to your kind notice that there is an accumulation of garbage at the above-mentioned location.  
This not only causes foul odor and breeding of disease-carrying vectors, but also poses a serious health and environmental hazard to local residents.

Kindly take immediate action to clear this waste and arrange for regular disposal.  

Thank you for your prompt attention.

Date: ________________            Signature: ________________  
""".strip()
    return jsonify({
        'application_template': template,
        'municipal_link': MUNICIPAL_COMPLAINT_URL
    })

@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json() or {}
    lang = data.get('language', 'en')
    issue = session.get('detected_issue', 'unspecified')
    loc = session.get('location', 'unspecified')
    bin_type = session.get('bin_type', 'unknown')
    waste_type = session.get('waste_type', 'unknown')
    cid = uuid.uuid4().hex[:8]
    templates = {
        'en': "Complaint: {issue}, Location: {loc}",
        'hi': "शिकायत: {issue}, स्थान: {loc}",
        'ta': "புகார்: {issue}, இடம்: {loc}",
        'bn': "অভিযোগ: {issue}, অবস্থান: {loc}",
        'ur': "شکایت: {issue}, مقام: {loc}"
    }
    complaint_text = templates.get(lang, templates['en']).format(issue=issue, loc=loc)
    comp = Complaint(
        id=cid, issue=issue, location=loc, language=lang,
        complaint_text=complaint_text, bin_type=bin_type, waste_type=waste_type
    )
    db.session.add(comp)
    db.session.commit()
    return jsonify({
        'complaint_text': complaint_text,
        'submission_link': f'/complaint/{cid}',
        'bin_type': bin_type,
        'waste_type': waste_type
    })

@app.route('/complaint/<cid>')
def view_complaint(cid):
    comp = Complaint.query.get_or_404(cid)
    return jsonify({
        'id': comp.id,
        'issue': comp.issue,
        'location': comp.location,
        'language': comp.language,
        'complaint_text': comp.complaint_text,
        'status': comp.status,
        'timestamp': comp.timestamp.isoformat(),
        'bin_type': comp.bin_type,
        'waste_type': comp.waste_type
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5400, debug=True)
