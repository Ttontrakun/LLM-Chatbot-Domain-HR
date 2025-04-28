import chromadb
from chromadb.utils import embedding_functions
import requests
import csv
import sys
import io
import os
import time
from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

nltk.download('punkt')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ตั้งค่า ChromaDB
chroma_client = chromadb.PersistentClient(path="./welfare_db")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_or_create_collection(
    name="welfare_documents",
    embedding_function=embedding_func
)

def load_data_from_csv(file_path):
    documents = []
    metadata_list = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                doc_type = row.get('Type', '').strip()
                question = row.get('Question', '').strip()
                answer = row.get('Answer', '').strip()
                
                if doc_type == "Q&A" and question:
                    documents.append(question)
                elif doc_type == "Detail" and answer:
                    documents.append(answer)
                else:
                    continue
                
                metadata_list.append({
                    "type": doc_type,
                    "question": question,
                    "answer": answer,
                    "category": row.get('Category', ''),
                    "subcategory": row.get('Subcategory', ''),
                    "detail": row.get('Detail', ''),
                    "eligible_group": row.get('EligibleGroup', '')
                })
        print(f"โหลดข้อมูลจาก {file_path} เรียบร้อย จำนวน {len(documents)} รายการ")
    except FileNotFoundError:
        print(f"ไม่พบไฟล์ {file_path} ใช้ dataset 50 ข้อในโค้ดแทน")
        default_dataset = [
            {"Question": "พนักงานมหาวิทยาลัยได้รับสิทธิ์กองทุนประกันสังคมอย่างไร?", "Answer": "ได้รับตามกฎหมาย"},
            {"Question": "พนักงานมหาวิทยาลัยเข้าร่วมกองทุนสำรองเลี้ยงชีพได้อย่างไร?", "Answer": "มีสิทธิ์เข้าร่วม"},
            {"Question": "สวัสดิการเกี่ยวกับ ตรวจสุขภาพประจำปี คืออะไร?", "Answer": "ตรวจฟรีที่โรงพยาบาลศิริราช/รพ.ธรรมศาสตร์"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินช่วยเหลือค่ารักษาพยาบาล คืออะไร?", "Answer": "ไม่เกิน 20,000 บาท/ปี (สะสมได้สูงสุด 40,000 บาท)"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินช่วยเหลือกรณีติดโควิด-19 คืออะไร?", "Answer": "ค่าใช้จ่ายเกี่ยวกับประกันโรคระบาด, วัคซีน และค่ารักษาพยabal"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินช่วยเหลือเยี่ยมผู้ป่วย คืออะไร?", "Answer": "กระเช้าเยี่ยมมูลค่า 500 บาท/ราย"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินช่วยเหลือค่าดอกเบี้ยบ้าน คืออะไร?", "Answer": "450 บาท/เดือน (สูงสุด 120 เดือน)"},
            {"Question": "สวัสดิการเกี่ยวกับ โครงการเงินกู้เพื่อที่อยู่อาศัย คืออะไร?", "Answer": "กู้เงินกับ ธ.อ.ส. และออมสิน โดยหักจากเงินเดือน"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินอุดหนุนการศึกษาบุตร คืออะไร?", "Answer": "อนุบาล - ป.6: 1,500 บาท/ปี, ม.1 - ม.6 (ปวช.3): 2,000 บาท/ปี"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินสนับสนุนศึกษาภาษาต่างประเทศ คืออะไร?", "Answer": "ไม่เกิน 50,000 บาท/5 ปี"},
            {"Question": "สวัสดิการเกี่ยวกับ ประกันชีวิต / อุบัติเหตุ / สุขภาพ (กลุ่ม) คืออะไร?", "Answer": "ได้รับตามสิทธิ์"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินช่วยเหลือภัยพิบัติ คืออะไร?", "Answer": "ไม่เกิน 10,000 บาท"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินช่วยเหลืองานศพ คืออะไร?", "Answer": "ข้าราชการ 3,500 บาท, คู่สมรส/บิดา/มารดา 3,000 บาท, บุตร 2,000 บาท"},
            {"Question": "สวัสดิการเกี่ยวกับ ค่าพวงหรีด คืออะไร?", "Answer": "มหาวิทยาลัย 700 บาท, คณะ 1,000 บาท"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินทดแทนการเสียชีวิต/บาดเจ็บจากงาน คืออะไร?", "Answer": "ได้รับในกรณีไม่ได้รับเงินจากประกันสังคม"},
            {"Question": "สวัสดิการเกี่ยวกับ เงินสนับสนุนสื่อการสอน คืออะไร?", "Answer": "ไม่เกิน 50,000 บาท/5 ปี"},
            {"Question": "สวัสดิการเกี่ยวกับ ของขวัญเกษียณอายุ คืออะไร?", "Answer": "ทำงานครบ 60 ปี: 20,000 บาท, ทำงานเกิน 25 ปีแล้วย้ายออก: 10,000 บาท"},
            {"Question": "สวัสดิการเกี่ยวกับ ของที่ระลึกโอกาสพิเศษ คืออะไร?", "Answer": "ไม่เกิน 1,000 บาท/ชิ้น"},
            {"Question": "พนักงานมหาวิทยาลัยได้รับสิทธิ์กองทุนประกันสังคมหรือไม่?", "Answer": "ได้รับสิทธิ์ตามกฎหมายที่เกี่ยวข้อง"},
            {"Question": "พนักงานมหาวิทยาลัยสามารถเบิกค่ารักษาพยาบาลได้เท่าไหร่?", "Answer": "สามารถเบิกได้สูงสุด 20,000 บาท/ปี และสะสมได้สูงสุด 40,000 บาท"},
            {"Question": "มหาวิทยาลัยช่วยค่าดอกเบี้ยที่อยู่อาศัยให้พนักงานหรือไม่?", "Answer": "ใช่ มหาวิทยาลัยช่วยเหลือเดือนละ 450 บาท นาน 10 ปี"},
            {"Question": "เงินอุดหนุนการศึกษาบุตรของพนักงานมหาวิทยาลัยมีกี่ระดับ?", "Answer": "อนุบาล - ป.6: 1,500 บาท/ปี, ม.1 - ม.6 (ปวช.3): 2,000 บาท/ปี"},
            {"Question": "กรณีเสียชีวิต บุคลากรจะได้รับเงินช่วยเหลือหรือไม่?", "Answer": "ได้รับเงินช่วยเหลือ 3,500 บาทสำหรับบุคลากร และ 3,000 บาทสำหรับคู่สมรสหรือบิดามารดา"},
            {"Question": "พนักงานมหาวิทยาลัยมีสิทธิ์เข้าร่วมกองทุนสำรองเลี้ยงชีพหรือไม่?", "Answer": "มีสิทธิ์เข้าร่วมได้ แต่เฉพาะพนักงานมหาวิทยาลัยเท่านั้น"},
            {"Question": "มหาวิทยาลัยมีสวัสดิการด้านประกันสุขภาพหรือไม่?", "Answer": "มี โดยเป็นประกันสุขภาพกลุ่มที่ครอบคลุมค่ารักษาพยabal"},
            {"Question": "พนักงานที่ได้รับผลกระทบจากโควิด-19 จะได้รับความช่วยเหลืออย่างไร?", "Answer": "ได้รับเงินช่วยเหลือค่าประกันโรคระบาด ค่าวัคซีน และค่ารักษาพยabalในส่วนที่เกินจากสิทธิเดิม"},
            {"Question": "พนักงานที่เกษียณอายุจะได้รับเงินช่วยเหลือหรือไม่?", "Answer": "ได้รับ โดยผู้ที่ทำงานครบ 60 ปีจะได้รับเงินช่วยเหลือสูงสุด 20,000 บาท"},
            {"Question": "พนักงานสามารถขอเงินช่วยเหลือภัยพิบัติได้หรือไม่?", "Answer": "สามารถขอได้ โดยได้รับเงินช่วยเหลือสูงสุด 10,000 บาท"},
            {"Question": "มหาวิทยาลัยให้ค่าพวงหรีดกรณีเสียชีวิตเท่าไหร่?", "Answer": "มหาวิทยาลัยให้ 700 บาท และคณะให้สูงสุด 1,000 บาท"},
            {"Question": "พนักงานมหาวิทยาลัยสามารถกู้เงินเพื่อที่อยู่อาศัยได้หรือไม่?", "Answer": "สามารถกู้ได้ผ่านโครงการของ ธ.อ.ส. และธนาคารออมสิน โดยหักจากเงินเดือน"},
            {"Question": "เงินช่วยเหลือด้านสื่อการสอนสำหรับอาจารย์มีเงื่อนไขอย่างไร?", "Answer": "อาจารย์สามารถเบิกค่าอุปกรณ์การเรียนการสอนได้สูงสุด 50,000 บาทต่อ 5 ปี"},
            {"Question": "สวัสดิการช่วยเหลือกรณีประสบภัยธรรมชาติมีอะไรบ้าง?", "Answer": "พนักงานที่ได้รับผลกระทบจากอุทกภัย วาตภัย หรืออัคคีภัย สามารถรับเงินช่วยเหลือได้สูงสุด 10,000 บาท"},
            {"Question": "พนักงานมหาวิทยาลัยได้รับสิทธิ์กองทุนประกันสังคมหรือไม่?", "Answer": "ได้รับสิทธิ์ตามกฎหมายที่เกี่ยวข้อง"},
            {"Question": "พนักงานมหาวิทยาลัยสามารถเข้าร่วมกองทุนสำรองเลี้ยงชีพได้หรือไม่?", "Answer": "สามารถเข้าร่วมได้"},
            {"Question": "พนักงานมหาวิทยาลัยสามารถเบิกค่ารักษาพยาบาลได้เท่าไหร่?", "Answer": "สามารถเบิกได้สูงสุด 20,000 บาท/ปี และสะสมได้สูงสุด 40,000 บาท"},
            {"Question": "มหาวิทยาลัยช่วยค่าดอกเบี้ยที่อยู่อาศัยให้พนักงานหรือไม่?", "Answer": "ใช่ มหาวิทยาลัยช่วยเหลือเดือนละ 450 บาท นาน 10 ปี"},
            {"Question": "เงินอุดหนุนการศึกษาบุตรของพนักงานมหาวิทยาลัยมีอะไรบ้าง?", "Answer": "อนุบาล - ป.6: 1,500 บาท/ปี, ม.1 - ม.6 (ปวช.3): 2,000 บาท/ปี"},
            {"Question": "พนักงานมหาวิทยาลัยได้รับประกันชีวิตและประกันอุบัติเหตุหรือไม่?", "Answer": "ได้รับสิทธิ์ประกันชีวิต, ประกันอุบัติเหตุ และประกันสุขภาพกลุ่ม"},
            {"Question": "มหาวิทยาลัยมีเงินช่วยเหลือกรณีภัยพิบัติหรือไม่?", "Answer": "สามารถขอได้ โดยได้รับเงินช่วยเหลือสูงสุด 10,000 บาท"},
            {"Question": "กรณีเสียชีวิต บุคลากรจะได้รับเงินช่วยเหลือหรือไม่?", "Answer": "ได้รับเงินช่วยเหลือ 3,500 บาทสำหรับบุคลากร และ 3,000 บาทสำหรับคู่สมรสหรือบิดามารดา"},
            {"Question": "พนักงานสามารถขอเงินช่วยเหลือค่าครองชีพได้หรือไม่?", "Answer": "สามารถขอรับได้ 3,000 บาท/เดือน หากมีผลการประเมินระดับดีขึ้นไป"},
            {"Question": "พนักงานสามารถเบิกค่าชุดเครื่องแต่งกายได้หรือไม่?", "Answer": "สามารถเบิกได้สูงสุด 4,000 บาท/ปี (ลูกจ้างประจำ 2,000 บาท)"},
            {"Question": "พนักงานมหาวิทยาลัยสายวิชาการมีหลักเกณฑ์การต่อสัญญาจ้างอย่างไร?", "Answer": "สัญญาที่ 1: 2 ปี, สัญญาที่ 2: 3 ปี, สัญญาที่ 3: 5 ปี, สัญญาที่ 4: จนเกษียณอายุ"},
            {"Question": "พนักงานมหาวิทยาลัยสายวิชาการที่มีปริญญาเอกต่อสัญญาที่ 3 ได้อย่างไร?", "Answer": "ต้องได้ตำแหน่ง ผศ. หรือยื่นขอตำแหน่ง ผศ. ภายใน 6 เดือนก่อนครบสัญญาที่ 2 มิฉะนั้นสัญญาจะสิ้นสุด"},
            {"Question": "พนักงานมหาวิทยาลัยสายวิชาการที่มีปริญญาเอกต่อสัญญาที่ 4 ได้อย่างไร?", "Answer": "ต้องได้ตำแหน่งไม่ต่ำกว่า รศ. ถ้ายังไม่ได้ให้ขยายสัญญาที่ 3 อีกครั้งละ 1 ปี สูงสุด 2 ครั้ง หากครบแล้วยังไม่ได้ รศ. สัญญาจะสิ้นสุด"},
            {"Question": "ลาป่วยได้กี่วัน?", "Answer": "ไม่เกิน 60 วันทําการต่อปี (มีเงินเดือน), เกิน 60 ถึง 120 วันอาจได้หรือไม่ได้เงิน, เกิน 120 วันไม่มีเงิน ใบรับรองแพทย์ต้องแนบถ้าลาเกิน 30 วัน"},
            {"Question": "ลาคลอดบุตรได้กี่วัน?", "Answer": "ไม่เกิน 90 วันต่อครรภ์ ได้เงินเดือน 45 วันจากมหาวิทยาลัย + 45 วันจากประกันสังคม"},
            {"Question": "ลาเพื่อดูแลบุตรและภรรยาหลังคลอดได้กี่วัน?", "Answer": "ไม่เกิน 15 วันทําการต่อครั้ง ได้เงินเดือนเต็ม"},
            {"Question": "ลากิจส่วนตัวได้กี่วัน?", "Answer": "ไม่เกิน 45 วันทําการต่อปี (มีเงินเดือน), ปีแรกไม่เกิน 15 วัน, ลาเลี้ยงบุตร 150 วันไม่มีเงิน"},
            {"Question": "ลาพักผ่อนประจำปีได้กี่วัน?", "Answer": "10 วันทําการต่อปี สะสมได้สูงสุด 20 วัน (หรือ 30 วันถ้าทำงานเกิน 10 ปี) ได้เงินเดือนเต็ม"}
        ]
        for item in default_dataset:
            documents.append(item["Question"])
            metadata_list.append({
                "type": "Q&A",
                "question": item["Question"],
                "answer": item["Answer"],
                "category": "",
                "subcategory": "",
                "detail": "",
                "eligible_group": ""
            })
    return documents, metadata_list[:50]  # จำกัด 50 ข้อ

def add_data_to_chroma(documents, metadata_list):
    if collection.count() == 0:
        ids = [f"doc_{i}" for i in range(len(documents))]
        collection.add(documents=documents, ids=ids, metadatas=metadata_list)
        print("เพิ่มข้อมูลลง ChromaDB เรียบร้อย!")
    else:
        print("ข้อมูลมีอยู่ใน ChromaDB แล้ว")

def call_typhoon(query, reference_answer, retries=3, delay=5):
    endpoint = 'https://api.opentyphoon.ai/v1/chat/completions'
    
    prompt = f"""ตอบคำถาม: {query}
    อิงข้อมูลจาก dataset:
    - คำตอบ: {reference_answer}
    - ตอบเป็นภาษาไทย สั้น กระชับ ตรงประเด็น ใช้ข้อมูลนี้เท่านั้น
    - ห้ามเริ่มด้วย "สวัสดีครับ" หรือคำทักทาย"""
    
    headers = {
        "Authorization": "Bearer sk-wfFmJzJCZwHeoYrx21xUEtC3fDiRRXLPBIwuAFxSxqF34KvH",
    }
    payload = {
        "model": "typhoon-v2-8b-instruct",
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "คุณคือ HR Assistant ที่ให้ข้อมูลจาก dataset"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.05,
        "min_p": 0
    }
    
    for attempt in range(retries):
        try:
            res = requests.post(endpoint, json=payload, headers=headers, verify=False)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if res.status_code == 429:
                wait_time = delay * (attempt + 1)
                print(f"429 Too Many Requests - รอ {wait_time} วินาทีก่อนลองใหม่ (ครั้งที่ {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                return f"เกิดข้อผิดพลาด: {str(e)}"
        except Exception as e:
            return f"เกิดข้อผิดพลาด: {str(e)}"
    return "เกิดข้อผิดพลาด: เกินจำนวนครั้งที่ลองใหม่ (429 Too Many Requests)"

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
    file_path = "welfare_data.csv"
    documents, metadata_list = load_data_from_csv(file_path)
    add_data_to_chroma(documents, metadata_list)
    
    results = []
    
    for item in metadata_list:
        query = item["question"]
        ref_answer = item["answer"]
        
        typhoon_response = call_typhoon(query, ref_answer)
        metrics = evaluate_response(typhoon_response, ref_answer)
        
        print(f"\nคำถาม: {query}")
        print(f"คำตอบอ้างอิง: {ref_answer}")
        print(f"Typhoon Response: {typhoon_response}")
        print(f"Metrics: BLEU: {metrics['bleu']:.4f}, ROUGE-L: {metrics['rouge_l']:.4f}, BERTScore: {metrics['bertscore']:.4f}")
        
        results.append({
            "Question": query,
            "Reference Answer": ref_answer,
            "Typhoon Response": typhoon_response,
            "BLEU": metrics['bleu'],
            "ROUGE-L": metrics['rouge_l'],
            "BERTScore": metrics['bertscore']
        })
        time.sleep(5)  # Delay 5 วินาที
    
    with open("typhoon_evaluation_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Question", "Reference Answer", "Typhoon Response", "BLEU", "ROUGE-L", "BERTScore"])
        writer.writeheader()
        writer.writerows(results)
    
    total_bleu = sum(r["BLEU"] for r in results)
    total_rouge = sum(r["ROUGE-L"] for r in results)
    total_bert = sum(r["BERTScore"] for r in results)
    count = len(results)
    
    print(f"\nค่าเฉลี่ยจาก {count} คำถาม:")
    print(f"Average BLEU: {total_bleu/count:.4f}")
    print(f"Average ROUGE-L: {total_rouge/count:.4f}")
    print(f"Average BERTScore: {total_bert/count:.4f}")
    print("ผลลัพธ์ถูกบันทึกใน typhoon_evaluation_results.csv")

if __name__ == "__main__":
    test_dataset()