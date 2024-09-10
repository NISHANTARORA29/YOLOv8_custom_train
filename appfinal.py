import os
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import cv2
from PIL import Image
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Load your trained YOLO model
yolo_model = YOLO('/Users/nishantarora/Desktop/intern proj/runs/detect/custom_yolo_advance/weights/best.pt')  # Update this path

# Define the classes your model was trained on
desired_classes = ['machine1', 'machine2', 'machine3', 'machine4', 'machine5', 'machine6', 'machine7', 'machine8-L1', 
                   'machine8-L2', 'machine8-L3', 'machine9', 'machine10', 'machine11', 'machine12']

# Load the BERT QA model and tokenizer
qa_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

class_colors = {
    'machine1': (255, 0, 0), 'machine2': (0, 255, 0), 'machine3': (0, 0, 255), 'machine4': (255, 255, 0),
    'machine5': (255, 0, 255), 'machine6': (0, 255, 255), 'machine7': (128, 0, 128),
    'machine9': (128, 128, 128), 'machine10': (64, 0, 64), 'machine11': (0, 64, 64), 'machine12': (64, 64, 0),
    'machine8-L1': (255, 128, 0), 'machine8-L2': (0, 255, 128), 'machine8-L3': (128, 0, 255)
}

def detect_objects_in_group(image_path):
    image = Image.open(image_path)
    image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(image_rgb)
    detected_labels = []
    machine_count = {}

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

    return Image.fromarray(image_rgb), detected_labels, machine_count

def answer_question_from_text(text_file_path, question):
    with open(text_file_path, 'r') as file:
        context = file.read()
    inputs = qa_tokenizer.encode_plus(question, context, max_length=512, truncation=True, return_tensors='pt')
    outputs = qa_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    return qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

st.title("Group Image Object Detection and Q&A Bot")
uploaded_group_images = st.file_uploader("Upload group images (1-10 images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_group_images:
    results_text = ""

    for i, uploaded_image in enumerate(uploaded_group_images):
        st.image(uploaded_image, caption=f"Uploaded Group Image {i+1}", use_column_width=True)
        st.write(f"Detecting objects in Image {i+1}...")

        detected_image, detected_objects, machine_count = detect_objects_in_group(uploaded_image)
        st.image(detected_image, caption=f"Detected Image {i+1} with Annotations", use_column_width=True)

        if detected_objects:
            results_text += f"Image {i+1} - Detected Objects: {', '.join(detected_objects)}\n"
            results_text += f"Machine Counts: {machine_count}\n\n"

            for detected_object in set(detected_objects):
                st.write(f"Ask a question about the {detected_object} in Image {i+1}:")
                question = st.text_input(f"Your question about the {detected_object} in Image {i+1}:", key=f"{detected_object}_{i}")
                if question:
                    text_file_path = f"{detected_object}.txt"
                    answer = answer_question_from_text(text_file_path, question)
                    st.write(f"Answer: {answer}")
                    results_text += f"Question about {detected_object}: {question}\nAnswer: {answer}\n\n"

    st.download_button(
        label="Download Results",
        data=results_text,
        file_name="detection_results.txt",
        mime="text/plain"
    )

