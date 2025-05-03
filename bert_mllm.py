import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/home/Desktop/client.json"
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this
import werkzeug
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import io
import json
import pandas as pd
import torch
import re
from tkinter import Tk, filedialog
from google.cloud import vision
from google.cloud.vision_v1 import types
from transformers import BertTokenizer, BertForSequenceClassification
import subprocess
import ollama

class FormProcessor:
    def __init__(self, output_dir: str = "bert_mllm"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.bert_model_path = "./bert_model"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model_path)
        self.model.eval()

        self.central_corpus_path = "central_corpus.xlsx"
        if os.path.exists(self.central_corpus_path):
            self.corpus_df = pd.read_excel(self.central_corpus_path)
        else:
            self.corpus_df = pd.DataFrame(columns=["Admission No", "Patient Name", "Hospital Name", "Prescription", "Observations", "Date Visited"])

    def select_image_gui(self):
        root = Tk()
        root.withdraw()
        return filedialog.askopenfilename(title="Select Medical Form Image")

    def extract_text_google_vision(self, image_path: str):
        print("🔍 Vision processing...")
        client = vision.ImageAnnotatorClient()
        with io.open(image_path, "rb") as f:
            content = f.read()

        response = client.text_detection(image=vision.Image(content=content))
        lines = [annotation.description for annotation in response.text_annotations[1:]]
        raw_text = "\n".join(lines)
        with open(os.path.join(self.output_dir, "raw_extracted_text.txt"), "w") as f:
            f.write(raw_text)
        print("✅ Text extracted.")
        return raw_text.strip(), lines

    def extract_with_ollama(self, full_text: str) -> dict:
        prompt = f"""
        Extract patient's full name and admission number (e.g., 21JE0796) from the following:

        {full_text}

        Provide output in JSON format:
        {{"Patient Name": ..., "Admission No": ...}}
        """
        result = subprocess.run(["ollama", "run", "llama3", prompt], stdout=subprocess.PIPE, text=True)
        try:
            json_text = re.search(r'\{.*?\}', result.stdout, re.DOTALL)
            return json.loads(json_text.group(0)) if json_text else {"Patient Name": "", "Admission No": ""}
        except Exception:
            return {"Patient Name": "", "Admission No": ""}

    def extract_fixed_fields(self, lines: list, raw_text: str) -> dict:
        fields = {"Admission No": "", "Patient Name": "", "Hospital Name": "", "Date Visited": ""}
        fields.update(self.extract_with_ollama(raw_text))

        for line in lines:
            if "INSTITUTE" in line or "HEALTH CENTRE" in line:
                fields["Hospital Name"] = line.strip()
                break

        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', raw_text)
        if match:
            fields["Date Visited"] = f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"

        if not re.match(r'\d{2}JE\d{4,5}', fields["Admission No"], re.IGNORECASE):
            fields["Admission No"] = self.generate_unique_id()

        return fields

    def clean_with_ollama(self, raw_text: str) -> list:
        print("🧹 Cleaning with Ollama...")
        ollama_prompt = f"""
        The following text is from a medical form. Extract only:
        - Prescriptions (medicines + dosages/frequency on the same line)
        - Observations (symptoms/conditions)

        Do NOT include any numbering, full forms, or extra phrases. Keep acronyms like BP, ECG, etc., as-is.
        Do not write anything except the lines themselves.

        Input:
        ---
        {raw_text}
        ---
        Output:
        """
        response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": ollama_prompt}])
        lines = response['message']['content'].strip().split("\n")
        return [re.sub(r"^\d+[\).]\s*", "", line.strip()) for line in lines if line.strip()]

    def classify_with_bert(self, cleaned_lines: list) -> dict:
        print("🧠 Classifying using BERT...")
        label_map = {0: "Prescription", 1: "Observation", 2: "Other"}
        result = {key: [] for key in ["Prescription", "Observation"]}

        inputs = self.tokenizer(cleaned_lines, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1).tolist()

        for text, label_id in zip(cleaned_lines, preds):
            if label_id == 1 and re.search(r'\bmg\b|\bml\b|\btablet\b|\bdose\b|\bcap\b|\bx\d\b', text, re.IGNORECASE):
                result["Prescription"].append(text)
            elif label_map[label_id] in result:
                result[label_map[label_id]].append(text)

        return result

    def generate_unique_id(self):
        existing = set(self.corpus_df["Admission No"].astype(str))
        from random import randint
        while True:
            new_id = f"21JE{randint(10000, 99999)}"
            if new_id not in existing:
                return new_id

    def update_corpus(self, fields: dict, classified: dict):
        entry = {
            "Admission No": fields["Admission No"],
            "Patient Name": fields["Patient Name"],
            "Hospital Name": fields["Hospital Name"],
            "Prescription": " | ".join(classified["Prescription"]),
            "Observations": " | ".join(classified["Observation"]),
            "Date Visited": fields["Date Visited"]
        }

        self.corpus_df = pd.concat([self.corpus_df, pd.DataFrame([entry])], ignore_index=True)

        self.corpus_df["Date Visited"] = pd.to_datetime(self.corpus_df["Date Visited"], format="%d/%m/%Y", errors="coerce")
        self.corpus_df.sort_values(by=["Admission No", "Date Visited"], inplace=True)
        self.corpus_df["Date Visited"] = self.corpus_df["Date Visited"].dt.strftime("%d/%m/%Y")

        self.corpus_df.to_excel(self.central_corpus_path, index=False)

    def save_classified_txt(self, classified: dict):
        for category in ["Prescription", "Observation"]:
            with open(os.path.join(self.output_dir, f"{category.lower()}.txt"), "w") as f:
                f.write("\n".join(classified[category]))


app = Flask(__name__)
# Configure CORS
CORS(
    app,
    resources={r"/process": {"origins": "*"}}
    # resources={
    #     r"/process": {
    #         "origins": ["https://*.ngrok-free.app"],
    #         "methods": ["POST"],
    #         "allow_headers": ["Content-Type","Accept"]
    #     }
    # }
)

# Configure upload folder (create this folder in your project directory)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process', methods=['POST'])
def process_form():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    
    file = request.files['image']
    
    # If user does not select file, browser submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        try:
            # Process the image using your existing class
            processor = FormProcessor()
            
            # Instead of using select_image_gui, we use the uploaded file path
            raw_text, lines = processor.extract_text_google_vision(image_path)
            fields = processor.extract_fixed_fields(lines, raw_text)
            cleaned_lines = processor.clean_with_ollama(raw_text)
            classified = processor.classify_with_bert(cleaned_lines)
            
            processor.save_classified_txt(classified)
            processor.update_corpus(fields, classified)
            
            # Clean up: remove the uploaded file after processing
            os.remove(image_path)
            
            return jsonify({
                'status': 'success',
                'admission_no': fields['Admission No'],
                'classified_data': classified,
                'fields': fields
            })
            
        except Exception as e:
            # Clean up in case of error
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/', methods=['GET'])
def get_res():
    return jsonify({'result': 'Welcome to API'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)
#     def run(self):
#         image_path = self.select_image_gui()
#         if not image_path:
#             print(" No image selected.")
#             return

#         raw_text, lines = self.extract_text_google_vision(image_path)
#         fields = self.extract_fixed_fields(lines, raw_text)
#         cleaned_lines = self.clean_with_ollama(raw_text)
#         classified = self.classify_with_bert(cleaned_lines)

#         self.save_classified_txt(classified)
#         self.update_corpus(fields, classified)

#         print(f"🎉 Process complete for Admission No: {fields['Admission No']}")

# if __name__ == "__main__":
#     processor = FormProcessor()
#     processor.run()



