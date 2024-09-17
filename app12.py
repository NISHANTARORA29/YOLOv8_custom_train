import os
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import cv2
from PIL import Image
import numpy as np
import streamlit as st
from ultralytics import YOLO
from fpdf import FPDF
from io import BytesIO
import tempfile

# Load YOLO model
yolo_model = YOLO('/Users/nishantarora/Desktop/intern proj/runs/detect/Model_for_angles/weights/best.pt')  # Update this path

# Define the classes your model was trained on
desired_classes = ['machine1', 'machine2', 'machine3', 'machine4', 'machine8-L1', 'machine8-L2', 'machine8-L3', 
                   'machine8-L3-001', 'machine8-L2-001', 'machine8-L1-001', 'machine8-L2-002', 
                   'machine1-001', 'machine8-L1-002', 'machine4-001']

# Load QA model and tokenizer
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Define class colors for bounding boxes
class_colors = {
    'machine1': (255, 0, 0), 'machine2': (0, 255, 0), 'machine3': (0, 0, 255), 'machine4': (255, 255, 0),
    'machine8-L1': (255, 128, 0), 'machine8-L2': (0, 255, 128), 'machine8-L3': (128, 0, 255),
    'machine8-L3-001': (128, 64, 0), 'machine8-L2-001': (64, 128, 0), 'machine8-L1-001': (0, 128, 64),
    'machine8-L2-002': (64, 0, 128), 'machine1-001': (128, 128, 0), 'machine8-L1-002': (0, 64, 128),
    'machine4-001': (128, 0, 64)
}

# Function to detect objects in group images and extract individual machine images
def detect_objects_in_group(image_path):
    image = Image.open(image_path)
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(image_rgb)
    detected_labels = []
    machine_count = {}
    extracted_machines = []

    for result in results[0].boxes:
        class_id = int(result.cls)
        label = yolo_model.names[class_id]
        if label in desired_classes:
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            width, height = x2 - x1, y2 - y1
            color = class_colors.get(label, (255, 255, 255))
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_rgb, f"{label} {width}x{height}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            detected_labels.append(label)
            machine_count[label] = machine_count.get(label, 0) + 1

            # Extract the detected machine and save it as a separate image
            machine_image = image_rgb[y1:y2, x1:x2]
            machine_image_pil = Image.fromarray(machine_image)
            extracted_machines.append((label, machine_image_pil))

    annotated_image = Image.fromarray(image_rgb)
    return annotated_image, detected_labels, machine_count, extracted_machines

# Function to answer questions based on a text file
def answer_question_from_text(text_file_path, question):
    with open(text_file_path, 'r') as file:
        context = file.read()
    inputs = qa_tokenizer.encode_plus(question, context, max_length=512, truncation=True, return_tensors='pt')
    outputs = qa_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    return qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

# Streamlit application
st.title("Group Image Object Detection and Q&A Bot")
uploaded_group_images = st.file_uploader("Upload group images (1-10 images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_group_images:
    results_text = ""

    for i, uploaded_image in enumerate(uploaded_group_images):
        st.image(uploaded_image, caption=f"Uploaded Group Image {i+1}", use_column_width=True)
        st.write(f"Detecting objects in Image {i+1}...")

        detected_image, detected_objects, machine_count, extracted_machines = detect_objects_in_group(uploaded_image)

        if detected_objects:
            results_text += f"Image {i+1} - Detected Objects: {', '.join(detected_objects)}\n"
            results_text += f"Machine Counts: {machine_count}\n\n"

            # Display annotated image
            st.image(detected_image, caption=f"Annotated Image {i+1}", use_column_width=True)

            for label, machine_image in extracted_machines:
                st.image(machine_image, caption=f"{label} in Image {i+1}", use_column_width=True)

            for detected_object in set(detected_objects):
                st.write(f"Ask a question about the {detected_object} in Image {i+1}:")
                question = st.text_input(f"Your question about the {detected_object} in Image {i+1}:", key=f"{detected_object}_{i}")
                if question:
                    text_file_path = f"{detected_object}.txt"
                    answer = answer_question_from_text(text_file_path, question)
                    st.write(f"Answer: {answer}")
                    results_text += f"Question about {detected_object}: {question}\nAnswer: {answer}\n\n"
    
    # Add summary text
    results_text += "\nSummary:\n"
    for i, uploaded_image in enumerate(uploaded_group_images):
        results_text += f"Image {i+1}: {len(detected_objects)} objects detected, including {', '.join(detected_objects)}.\n"

    # Offer download options: PDF, CSV, and TXT
    st.download_button(
        label="Download TXT",
        data=results_text,
        file_name="detection_results.txt",
        mime="text/plain"
    )

    # PDF Download Option
    def generate_pdf_with_images(results_text):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, txt="Group Image Object Detection Summary", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 10, results_text)
        pdf_output = BytesIO()
        pdf_str = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_str)
        pdf_output.seek(0)
        return pdf_output

    pdf_data = generate_pdf_with_images(results_text)
    st.download_button(
        label="Download PDF",
        data=pdf_data,
        file_name="detection_results.pdf",
        mime="application/pdf"
    )

    csv_data = results_text.replace("\n", ",")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="detection_results.csv",
        mime="text/csv"
    )
