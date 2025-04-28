import csv
import io
import sys
import os
import requests
from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import time
from requests.exceptions import ConnectionError

# ปิดคำเตือน SSL และตั้งค่าให้ requests ไม่ใช้ SSL verification
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
nltk.download('punkt')

# ตั้งค่า stdout และ stdin ให้รองรับ UTF-8
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

# Dataset คำถามเกี่ยวกับการลา (จากโค้ดใหม่)
dataset = [
    {"Question": "ลาป่วยได้กี่วัน?", "Answer": "ไม่เกิน 60 วันทําการต่อปี (มีเงินเดือน), เกิน 60 ถึง 120 วันอาจได้หรือไม่ได้เงิน, เกิน 120 วันไม่มีเงิน ใบรับรองแพทย์ต้องแนบถ้าลาเกิน 30 วัน"},
    {"Question": "ลาคลอดบุตรได้กี่วัน?", "Answer": "ไม่เกิน 90 วันต่อครรภ์ ได้เงินเดือน 45 วันจากมหาวิทยาลัย + 45 วันจากประกันสังคม"},
    {"Question": "ลาเพื่อดูแลบุตรและภรรยาหลังคลอดได้กี่วัน?", "Answer": "ไม่เกิน 15 วันทําการต่อครั้ง ได้เงินเดือนเต็ม"},
    {"Question": "ลากิจส่วนตัวได้กี่วัน?", "Answer": "ไม่เกิน 45 วันทําการต่อปี (มีเงินเดือน), ปีแรกไม่เกิน 15 วัน, ลาเลี้ยงบุตร 150 วันไม่มีเงิน"},
    {"Question": "ลาพักผ่อนประจำปีได้กี่วัน?", "Answer": "10 วันทําการต่อปี สะสมได้สูงสุด 20 วัน (หรือ 30 วันถ้าทำงานเกิน 10 ปี) ได้เงินเดือนเต็ม"},
    {"Question": "ลาอุปสมบทหรือฮัจย์ได้กี่วัน?", "Answer": "ไม่เกิน 120 วัน ครั้งเดียวในชีวิต ได้เงินเดือนเต็ม"},
    {"Question": "ลาปฏิบัติธรรมตามมติคณะรัฐมนตรีได้กี่วัน?", "Answer": "30-90 วัน ครั้งเดียวในชีวิต ได้เงินเดือนเต็ม"},
    {"Question": "ลาติดตามคู่สมรสได้กี่วัน?", "Answer": "ไม่เกิน 365 วัน ครั้งเดียวในชีวิต ไม่ได้เงินเดือน"},
    {"Question": "ลาฟื้นฟูสมรรถภาพด้านอาชีพได้กี่วัน?", "Answer": "ไม่เกิน 12 เดือนตามหลักสูตร ครั้งเดียว ได้เงินเดือนเต็ม"},
    {"Question": "ลาตรวจเลือกหรือเตรียมพลได้กี่วัน?", "Answer": "ตามหมายเรียก ได้เงินเดือนเต็ม"}
]

def call_pathumma(query, context, retries=3, delay=5):
    endpoint = "https://api.aiforthai.in.th/textqa/completion"
    api_key = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJSbXRkR1BoYSIsImlhdCI6MTc0MjUyNDA2NCwibmJmIjoxNzQyNTIzNzY0LCJleHAiOjE3NDI2MTA0NjQsImV4cGlyZUlnIjo4NjQwMCwiY3R4Ijp7ImNsaWVudGlkIjoiNWI0NGM0ZDRkMWMwODY5ZmE0YjNlZDkyZTFmNzgzYjgiLCJ1c2VyaWQiOiJVMzcyODM5Nzk1MzI3In0sImlzcyI6ImNlcjp1c2VydG9rZW6ifQ.fI5Ma5g5hjb3FUYXrVSo3_mqNTWGyyTVs9pYtZGtkAyuYvjOb_SHSfTPFBKr6GCnShaKI9jPm9Tv7m_ApC3NEA"
    
    # Prompt แบบสั้นกระชับจากโค้ดเก่า
    prompt = f"""ตอบคำถาม: {query}
    อิงข้อมูลจาก dataset:
    - คำตอบ: {context}
    - ตอบเป็นภาษาไทย สั้น กระชับ ตรงประเด็น ใช้ข้อมูลนี้เท่านั้น
    - ห้ามเริ่มด้วย "สวัสดีครับ" หรือคำทักทาย"""
    
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    for attempt in range(retries):
        try:
            res = requests.post(endpoint, json=payload, headers=headers, verify=False, timeout=30)
            res.raise_for_status()
            return res.json().get("content", "เกิดข้อผิดพลาด: ไม่มี content ใน response").strip()
        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                wait_time = delay * (attempt + 1)
                print(f"429 Too Many Requests - รอ {wait_time} วินาทีก่อนลองใหม่ (ครั้งที่ {attempt + 1}/{retries})")
                time.sleep(wait_time)
            elif res.status_code == 401:
                return "เกิดข้อผิดพลาด: API Key ไม่ถูกต้องหรือไม่มีสิทธิ์เข้าใช้งาน"
            else:
                return f"เกิดข้อผิดพลาด HTTP: {str(e)}"
        except ConnectionError as e:
            wait_time = delay * (attempt + 1)
            print(f"Connection Error (10054) - รอ {wait_time} วินาทีก่อนลองใหม่ (ครั้งที่ {attempt + 1}/{retries})")
            time.sleep(wait_time)
        except requests.exceptions.Timeout:
            wait_time = delay * (attempt + 1)
            print(f"Timeout Error - รอ {wait_time} วินาทีก่อนลองใหม่ (ครั้งที่ {attempt + 1}/{retries})")
            time.sleep(wait_time)
        except Exception as e:
            return f"เกิดข้อผิดพลาดอื่นๆ: {str(e)}"
    return "เกิดข้อผิดพลาด: เกินจำนวนครั้งที่ลองใหม่"

def evaluate_response(generated_response, reference_response):
    cleaned_response = generated_response.strip()
    reference_tokens = word_tokenize(reference_response, engine="newmm")
    generated_tokens = word_tokenize(cleaned_response, engine="newmm")
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference_tokens], generated_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_scores = scorer.score(reference_response, generated_response)
    rouge_l_f1 = rouge_scores['rougeL'].fmeasure
    
    P, R, F1 = bert_score([cleaned_response], [reference_response], lang="th", verbose=False)
    bertscore_f1 = F1.item()
    
    return {"bleu": bleu_score, "rouge_l": rouge_l_f1, "bertscore": bertscore_f1}

def test_dataset():
    total_bleu, total_rouge, total_bert = 0, 0, 0
    count = 0
    
    results = []
    
    for item in dataset:
        test_q = item["Question"]
        ref_answer = item["Answer"]
        pathumma_response = call_pathumma(test_q, ref_answer)
        metrics = evaluate_response(pathumma_response, ref_answer)
        
        print(f"\nคำถาม: {test_q}")
        print(f"คำตอบอ้างอิง: {ref_answer}")
        print(f"Pathumma Response: {pathumma_response}")
        print(f"Metrics: BLEU: {metrics['bleu']:.4f}, ROUGE-L: {metrics['rouge_l']:.4f}, BERTScore: {metrics['bertscore']:.4f}")
        
        total_bleu += metrics['bleu']
        total_rouge += metrics['rouge_l']
        total_bert += metrics['bertscore']
        count += 1
        
        results.append({
            "Question": test_q,
            "Reference Answer": ref_answer,
            "Pathumma Response": pathumma_response,
            "BLEU": metrics['bleu'],
            "ROUGE-L": metrics['rouge_l'],
            "BERTScore": metrics['bertscore']
        })
        time.sleep(5)  # Delay 5 วินาที
    
    # บันทึกผลลัพธ์ลง CSV
    with open("pathumma_evaluation_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Question", "Reference Answer", "Pathumma Response", "BLEU", "ROUGE-L", "BERTScore"])
        writer.writeheader()
        writer.writerows(results)
    
    # คำนวณค่าเฉลี่ย
    print(f"\nค่าเฉลี่ยจาก {count} คำถาม:")
    print(f"Average BLEU: {total_bleu/count:.4f}")
    print(f"Average ROUGE-L: {total_rouge/count:.4f}")
    print(f"Average BERTScore: {total_bert/count:.4f}")
    print("ผลลัพธ์ถูกบันทึกใน pathumma_evaluation_results.csv")

if __name__ == "__main__":
    test_dataset()