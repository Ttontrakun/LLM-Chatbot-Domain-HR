import chromadb
from chromadb.utils import embedding_functions
import requests
import csv
import sys
import io
import json
import os
import warnings

# ปิด warning เกี่ยวกับ SSL
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# บังคับ encoding เป็น UTF-8
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ตั้งค่า Chroma Client
chroma_client = chromadb.PersistentClient(path="./welfare_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_or_create_collection(
    name="welfare_documents",
    embedding_function=embedding_func
)

# โหลดข้อมูลจาก CSV
def load_data_from_csv(file_path):
    documents = []
    metadata_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            doc_type = row.get('Type', '').strip()
            question = row.get('Question', '').strip()
            answer = row.get('Answer', '').strip()
            detail = row.get('Detail', '').strip()
            
            if doc_type == "Q&A" and question:
                documents.append(question)
            elif doc_type == "Detail" and detail:
                documents.append(detail)
            else:
                continue
            
            metadata_list.append({
                "type": doc_type,
                "question": question,
                "answer": answer,
                "category": row.get('Category', ''),
                "subcategory": row.get('Subcategory', ''),
                "detail": detail,
                "eligible_group": row.get('EligibleGroup', '')
            })
    return documents, metadata_list

def add_data_to_chroma(documents, metadata_list):
    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids, metadatas=metadata_list)

def search_data(query, documents, metadata_list, category=None, n_results=1):
    filter_condition = {"category": category} if category else None
    results = collection.query(query_texts=[query], n_results=n_results, where=filter_condition)
    matched_doc = results["documents"][0][0]
    distance = results["distances"][0][0]  # ระยะห่าง (ความคล้ายกัน)
    
    # ถ้าระยะห่างมากเกินไป (เช่น > 0.5) ถือว่าไม่เกี่ยวข้อง
    if distance > 0.5:
        return None, None, None, None
    
    for doc, meta in zip(documents, metadata_list):
        if doc == matched_doc:
            return meta["answer"], meta["category"], meta["detail"], meta["eligible_group"]
    return None, None, None, None

def call_typhoon(query, answer=None, category=None, detail=None, eligible_group=None):
    endpoint = "https://api.opentyphoon.ai/v1/chat/completions"
    api_key = "sk-qIhiUd3uyyCa2zPHGjwK5Jq7iejtjNDaFPuaIeC0kVNRtiuW"
    
    if answer:  # ถ้ามีข้อมูลจาก dataset
        prompt = f"""คุณคือ Assistant ที่เชี่ยวชาญด้านสวัสดิการ ช่วยตอบคำถามโดยอิงข้อมูล:
        หมวดหมู่: {category or 'ไม่ระบุ'}
        ข้อมูลหลัก: {answer}
        รายละเอียดเพิ่มเติม: {detail or 'ไม่มี'}
        กลุ่มบุคลากรที่มีสิทธิ์: {eligible_group or 'ไม่ระบุ'}
        คำถาม: {query}
        - ตอบเป็นภาษาไทยอย่างสุภาพและเป็นธรรมชาติ
        - อิง 'ข้อมูลหลัก' และ 'รายละเอียดเพิ่มเติม' เป็นหลัก
        - ห้ามเพิ่มข้อมูลที่ไม่เกี่ยวข้อง"""
    else:  # ถ้าไม่มีข้อมูลจาก dataset
        prompt = f"""คุณคือ Assistant ที่มีความรู้ทั่วไป ช่วยตอบคำถาม:
        คำถาม: {query}
        - ตอบเป็นภาษาไทยอย่างสุภาพและเป็นธรรมชาติ
        - ถ้าไม่รู้คำตอบ ให้บอกว่า "ขออภัยครับ ผมไม่ทราบข้อมูลในเรื่องนี้"
        - ห้ามเดาคำตอบมั่ว ๆ"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
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
            {"role": "system", "content": "คุณคือผู้ช่วยที่ให้ข้อมูลอย่างถูกต้องและสุภาพ"},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"

def chatbot(query, documents, metadata_list):
    matched_answer, category, detail, eligible_group = search_data(query, documents, metadata_list)
    response = call_typhoon(query, matched_answer, category, detail, eligible_group)
    return response

def interactive_chat():
    file_path = "welfare_data.csv"
    documents, metadata_list = load_data_from_csv(file_path)
    add_data_to_chroma(documents, metadata_list)
    
    print("สวัสดีครับ! ผมคือ Assistant ด้านสวัสดิการและความรู้ทั่วไป (ใช้ Typhoon v2-8b)")
    print("พิมพ์คำถามของคุณ หรือพิมพ์ 'ออก' เพื่อจบการสนทนา")
    
    while True:
        question = input("คำถาม: ")
        if question.strip().lower() == "ออก":
            print("ขอบคุณที่ใช้บริการครับ สวัสดีครับ!")
            break
        if not question.strip():
            print("กรุณาพิมพ์คำถามครับ")
            continue
        
        response = chatbot(question, documents, metadata_list)
        print("คำตอบ:", response)
        print("-" * 50)

if __name__ == "__main__":
    interactive_chat()