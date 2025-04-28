import chromadb
from chromadb.utils import embedding_functions
import requests
import csv
import json
import os
import gradio as gr

# ตั้งค่า ChromaDB และ Embedding
chroma_client = chromadb.PersistentClient(path="./hr_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_or_create_collection(
    name="hr_documents",
    embedding_function=embedding_func
)

# โหลดข้อมูลจาก CSV
def load_data_from_csv(file_path):
    documents = []
    metadata_list = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row.get('Question', '').strip()
            if question:
                documents.append(question)
                metadata_list.append({
                    "type": row.get('Type', ''),
                    "question": question,
                    "answer": row.get('Answer', ''),
                    "category": row.get('Category', ''),
                    "subcategory": row.get('Subcategory', ''),
                    "detail": row.get('Detail', ''),
                    "eligible_group": row.get('EligibleGroup', '')
                })
    return documents, metadata_list

# เพิ่มข้อมูลลง ChromaDB
def add_data_to_chroma(documents, metadata_list):
    if collection.count() == 0:
        ids = [f"doc_{i}" for i in range(len(documents))]
        collection.add(documents=documents, ids=ids, metadatas=metadata_list)
        print("เพิ่มข้อมูลลง ChromaDB เรียบร้อย!")
    else:
        print("ข้อมูลมีอยู่ใน ChromaDB แล้ว")

# ค้นหาคำถามที่ใกล้เคียง
def search_data(query, documents, metadata_list, category=None, n_results=1):
    filter_condition = {"category": category} if category else None
    results = collection.query(query_texts=[query], n_results=n_results, where=filter_condition)
    matched_doc = results["documents"][0][0]
    distance = results["distances"][0][0]
    
    if distance > 0.6:
        print(f"Debug: ไม่พบคำถามที่ใกล้เคียง (distance: {distance})")
        return None, None, None, None
    
    for doc, meta in zip(documents, metadata_list):
        if doc == matched_doc:
            print(f"Debug: จับคู่กับคำถาม '{doc}' (distance: {distance})")
            return meta["answer"], meta["category"], meta["detail"], meta["eligible_group"]
    return None, None, None, None

# เรียก Typhoon API
def call_typhoon(query, answer=None, category=None, detail=None, eligible_group=None):
    endpoint = "https://api.opentyphoon.ai/v1/chat/completions"
    api_key = "sk-qIhiUd3uyyCa2zPHGjwK5Jq7iejtjNDaFPuaIeC0kVNRtiuW"
    
    if answer:
        prompt = f"""ตอบคำถาม: {query}
        อิงข้อมูลจาก dataset:
        - คำตอบ: {answer}
        - หมวดหมู่: {category or 'ไม่ระบุ'}
        - รายละเอียด: {detail or 'ไม่มี'}
        - กลุ่มที่มีสิทธิ์: {eligible_group or 'ไม่ระบุ'}
        - ตอบเป็นภาษาไทย สุภาพ เรียบเรียงให้เป็นธรรมชาติ ใช้ข้อมูลนี้เท่านั้น"""
    else:
        prompt = f"""ตอบคำถาม: {query}
        - ตอบเป็นภาษาไทย สุภาพ เป็นธรรมชาติ
        - สามารถตอบแบบทั่วไปได้ตามความรู้ทั่วไป"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8"
    }
    payload = {
        "model": "typhoon-v2-8b-instruct",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.05,
        "min_p": 0,
        "messages": [
            {"role": "system", "content": "คุณคือผู้ช่วยฝ่ายทรัพยากรบุคคลที่ให้ข้อมูลจาก dataset หรือตอบแบบทั่วไปได้ถ้าไม่มีข้อมูล"},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, verify=False)
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        if answer:
            return f"คำตอบจากข้อมูล: {answer} (หมวดหมู่: {category or 'ไม่ระบุ'}, รายละเอียด: {detail or 'ไม่มี'})"
        return "ขออภัยครับ ผมไม่สามารถเชื่อมต่อระบบได้ โปรดลองใหม่ภายหลัง"

# โหลดข้อมูลครั้งแรก
file_path = "hr_data.csv"
documents, metadata_list = load_data_from_csv(file_path)
add_data_to_chroma(documents, metadata_list)

# ฟังก์ชันสำหรับ chatbot ใน Gradio
def chatbot_interface(user_input, history):
    if not user_input.strip():
        return "กรุณาพิมพ์คำถามครับ", history
    
    # เรียก chatbot
    response = call_typhoon(user_input, *search_data(user_input, documents, metadata_list))
    
    # อัปเดตประวัติการสนทนา
    history.append((user_input, response))
    return "", history  # ล้างช่อง input และส่งประวัติกลับ

# สร้าง UI ด้วย Gradio
with gr.Blocks(title="HR Assistant - มหาวิทยาลัยธรรมศาสตร์") as demo:
    gr.Markdown("# HR Assistant - มหาวิทยาลัยธรรมศาสตร์")
    gr.Markdown("สวัสดีครับ! ผมคือผู้ช่วยฝ่ายทรัพยากรบุคคล พร้อมตอบคำถามเกี่ยวกับสวัสดิการและข้อมูลทั่วไป")
    
    # ส่วนแชท
    chatbot = gr.Chatbot(label="การสนทนา")
    user_input = gr.Textbox(placeholder="พิมพ์คำถามของคุณที่นี่...", label="คำถาม")
    submit_btn = gr.Button("ส่งคำถาม")
    
    # การทำงานเมื่อกดปุ่ม
    submit_btn.click(
        fn=chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )
    
    # ล้างประวัติ
    clear_btn = gr.Button("ล้างการสนทนา")
    clear_btn.click(lambda: ("", []), inputs=None, outputs=[user_input, chatbot])

# รันแอป
demo.launch(share=True)