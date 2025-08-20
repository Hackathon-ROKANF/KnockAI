import os
import re
import fitz  # PyMuPDF
from datetime import datetime
import pandas as pd
import json
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from scipy import stats  # 백분위 계산을 위해 추가
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("❌ [경고] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. API 호출이 실패합니다.")


# --- 1. PDF 분석 로직 (기존과 동일) ---
def korean_to_int(kstr):
    if not kstr: return 0
    kstr = str(kstr).replace(",", "").replace(" ", "").strip()
    if kstr.isdigit(): return int(kstr)
    num_map = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    unit_map = {'십': 10, '백': 100, '천': 1000}
    large_unit_map = {'만': 10000, '억': 100000000, '조': 1000000000000}
    total_sum, temp_sum, current_num = 0, 0, 0
    for char in kstr:
        if char in num_map:
            current_num = num_map[char]
        elif char in unit_map:
            temp_sum += (current_num if current_num else 1) * unit_map[char]
            current_num = 0
        elif char in large_unit_map:
            temp_sum += current_num
            total_sum += (temp_sum if temp_sum else 1) * large_unit_map[char]
            temp_sum, current_num = 0, 0
    total_sum += temp_sum + current_num
    return total_sum


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc: text += page.get_text()
    return text


def parse_register_info_detailed(text):
    features = {
        '건축물_유형': None, '근저당권_개수': 0, '채권최고액': 0, '근저당권_설정일_최근': None,
        '신탁_등기여부': False, '압류_가압류_개수': 0, '선순위_채권_존재여부': False,
        '전입_가능여부': True, '우선변제권_여부': True, '주소': None, '전세가': None,
        '매매가': None, '전세가율': None, '과거_매매가': None, '과거_전세가': None, '과거_전세가율': None,
    }

    addr_match = re.search(r'\[\s*(?:건물|집합건물)\s*\]\s*(.+?)\n', text)
    if addr_match: features['주소'] = addr_match.group(1).strip()

    building_type_list = ['아파트', '빌라', '오피스텔', '다세대주택', '단독주택', '연립주택', '다가구주택']
    features['건축물_유형'] = '기타'
    for b_type in building_type_list:
        if b_type in text:
            features['건축물_유형'] = b_type
            break

    # --- [ ✨ 핵심 수정 ✨ ] 갑구 분석 로직 전면 개선 ---
    gapgu_section_match = re.search(r'【\s*갑\s*구\s*】([\s\S]+?)(?=【\s*을\s*구\s*】|--\s*이\s*하\s*여\s*백\s*--)', text)
    if gapgu_section_match:
        gapgu_section = gapgu_section_match.group(1)

        trusts = {}
        # 등기목적, 등기원인 등 상세정보를 포함하여 파싱
        gapgu_entries = re.finditer(
            r'(?P<num>\d+(?:-\d+)?)\n(?P<purpose>[\s\S]+?)\n(?P<receipt>.+?)\n(?P<cause>[\s\S]+?)\n(?P<details>[\s\S]+?)(?=\n\d+(?:-\d+)?\s|\Z)',
            gapgu_section)

        for entry in gapgu_entries:
            entry_dict = entry.groupdict()
            num = entry_dict['num'].strip()
            purpose = entry_dict['purpose'].strip()
            cause = entry_dict['cause'].strip()
            details = entry_dict['details'].strip()
            full_entry_text = entry.group(0)  # 해당 등기 항목 전체 텍스트

            # 1. 말소/귀속 등 신탁 해지 사유를 먼저 확인
            if '신탁등기말소' in purpose or '신탁재산의 귀속' in cause:
                target_nums = re.findall(r'(\d+)번', full_entry_text)
                for target_num in target_nums:
                    if target_num in trusts:
                        trusts[target_num]['active'] = False

            # 2. 신탁 설정 등기 확인 (말소/귀속이 아닌 경우)
            elif '소유권이전' in purpose and '신탁' in cause:
                trusts[num] = {'active': True}

        active_trusts_count = sum(1 for t in trusts.values() if t.get('active'))
        if active_trusts_count > 0:
            features['신탁_등기여부'] = True

        trade_prices = re.findall(r"거래가액\s*금([일이삼사오육칠팔구십백천만억조\d,\s]+)(?:원|정)", gapgu_section)
        if trade_prices:
            features['과거_매매가'] = korean_to_int(trade_prices[-1])

        seizures_list = re.findall(r"가압류|압류|경매개시결정", gapgu_section)
        features['압류_가압류_개수'] = len(seizures_list)

    # --- 을구 분석 로직 (기존과 동일) ---
    eulgu_section_match = re.search(r'【\s*을\s*구\s*】([\s\S]+?)(?=--\s*이\s*하\s*여\s*백\s*--|$)', text)
    if eulgu_section_match:
        # (을구 분석 로직은 이전과 동일하므로 생략)
        eulgu_section = eulgu_section_match.group(1)
        entries_text = re.split(r'\n(?=\d+(?:-\d+)?\s)', eulgu_section)
        mortgages, leases = {}, {}
        for entry_text in entries_text[1:]:
            # ... (이하 근저당권, 전세권 분석 로직은 기존과 동일)
            entry_text = entry_text.strip();
            if not entry_text: continue
            num_match = re.match(r'(\d+(?:-\d+)?)', entry_text)
            if not num_match: continue
            num = num_match.group(1)
            content = entry_text[len(num):].strip()
            if '말소' in content or '해지' in content:
                target_nums = re.findall(r'(\d+)번', content)
                if not target_nums and num.isdigit(): target_nums.append(num)
                for target_num in target_nums:
                    if target_num in mortgages: mortgages[target_num]['active'] = False
                    if target_num in leases: leases[target_num]['active'] = False
            elif ('전세권설정' in content or '주택임차권' in content):
                amount_match = re.search(r"(?:전세금|임차보증금)[\s\xa0]*금\s*([일이삼사오육칠팔구십백천만억조\d,\s]+)원", content)
                if amount_match: leases[num] = {'active': True, 'amount': korean_to_int(amount_match.group(1))}
            elif '근저당권설정' in content:
                amount_match = re.search(r"채권최고액\s*금\s*([일이삼사오육칠팔구십백천만억조\d,\s]+)(?:원|정)", content)
                date_match = re.search(r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)", content)
                if amount_match and date_match:
                    amount = korean_to_int(amount_match.group(1))
                    date_str = re.sub(r'\s+', '', date_match.group(1))
                    date = datetime.strptime(date_str, "%Y년%m월%d일")
                    mortgages[num] = {'active': True, 'amount': amount, 'date': date}
            elif '근저당권변경' in content and '-' in num:
                main_num = num.split('-')[0]
                if main_num in mortgages:
                    amount_match = re.search(r"채권최고액\s*금\s*([일이삼사오육칠팔구십백천만억조\d,\s]+)원", content)
                    if amount_match: mortgages[main_num]['amount'] = korean_to_int(amount_match.group(1))

        active_mortgages = [m for m in mortgages.values() if m.get('active')]
        features['근저당권_개수'] = len(active_mortgages)
        if active_mortgages:
            features['채권최고액'] = sum(m['amount'] for m in active_mortgages)
            features['근저당권_설정일_최근'] = max(m['date'] for m in active_mortgages).strftime('%Y-%m-%d')

        active_leases = [l for l in leases.values() if l.get('active')]
        if active_leases:
            features['과거_전세가'] = active_leases[-1]['amount']

    # --- 최종 피처 계산 ---
    if features['근저당권_개수'] > 0 or features['압류_가압류_개수'] > 0 or features['신탁_등기여부']:
        features['선순위_채권_존재여부'] = True

    if features['압류_가압류_개수'] > 0 or '주택임차권' in text:
        features['전입_가능여부'] = False

    if features['선순위_채권_존재여부'] or not features['전입_가능여부']:
        features['우선변제권_여부'] = False

    if features.get('과거_매매가') and features.get('과거_전세가'):
        try:
            ratio = (features['과거_전세가'] / features['과거_매매가']) * 100
            features['과거_전세가율'] = f"{ratio:.2f}%"
        except (TypeError, ZeroDivisionError):
            features['과거_전세가율'] = None

    return features


# --- 2. 모델 및 관련 데이터 로드 ---
try:
    with open('real_estate_model.pkl', 'rb') as f:
        saved_model_data = pickle.load(f)
    loaded_model = saved_model_data['model']
    training_columns = saved_model_data['columns']
    training_dtypes = saved_model_data['dtypes']
    train_scores = saved_model_data.get('train_scores')
    print("✅ 모델과 관련 데이터가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    loaded_model = None
    train_scores = None


# --- 3. 분석 등급 및 요약 생성 함수 ---
def get_risk_grade(risk_score, percentile):
    if risk_score <= 0:
        return "안전"
    if percentile is None:
        return "확인 필요"
    if percentile < 33.3:
        return "관심"
    elif percentile < 66.6:
        return "주의"
    else:
        return "위험"


# def generate_rule_based_summary(data_row, final_grade):
#     risk_factors = []
#     safe_factors = []
#
#     if data_row.get('우선변제권_여부') == False: risk_factors.append("'우선변제권' 확보 불확실")
#     if data_row.get('선순위_채권_존재여부') == True: risk_factors.append("'선순위 채권' 존재")
#     if data_row.get('압류_가압류_개수', 0) > 0: risk_factors.append(f"'압류/가압류' ({int(data_row.get('압류_가압류_개수', 0))}건) 존재")
#     if data_row.get('신탁_등기여부') == True: risk_factors.append("'신탁 등기' 존재")
#     if data_row.get('전입_가능여부') == False: risk_factors.append("'전입 불가' 상태")
#
#     past_price = data_row.get('과거_매매가', 0)
#     past_jeonse = data_row.get('과거_전세가', 0)
#     if past_price > 0 and past_jeonse > 0:
#         jeonse_ratio = (past_jeonse / past_price) * 100
#         if jeonse_ratio > 80:
#             risk_factors.append(f"높은 과거 전세가율 ({jeonse_ratio:.0f}%)")
#
#     if data_row.get('압류_가압류_개수', 0) == 0: safe_factors.append("'압류/가압류' 없음")
#     if not risk_factors: safe_factors.append("등기부등본상 특이사항 없음")
#
#     summary_parts = [f"최종 분석 등급은 '{final_grade}'입니다."]
#     if final_grade != "안전" and risk_factors:
#         summary_parts.append(f"주된 위험 요인: {', '.join(risk_factors)}.")
#     elif final_grade == "안전" and safe_factors:
#         summary_parts.append(f"주된 안전 요인: {', '.join(safe_factors)}.")
#         if risk_factors:
#             summary_parts.append(f"다만, {', '.join(risk_factors)} 부분은 확인이 필요합니다.")
#
#     return " ".join(summary_parts)


# --- 4. Flask 서버 설정 ---
app = Flask(__name__)
CORS(app, resources={r"/api/analyze": {"origins": "*"}})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def health_check():
    return "✅ Knock AI (One-Class SVM) 서버가 정상적으로 실행 중입니다!"


@app.route('/api/analyze', methods=['POST'])
def analyze_pdf_endpoint():
    if not loaded_model:
        return jsonify({"error": "서버에 모델이 로드되지 않았습니다."}), 500
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "PDF 파일이 아니거나 선택되지 않았습니다."}), 400

    try:
        # --- ✨ [수정 1] 체계적인 저장을 위한 경로 및 파일명 생성 ---

        # 1. 오늘 날짜로 폴더 경로 생성 (예: uploads/2025/08/20)
        today = datetime.now()
        date_path = os.path.join(app.config['UPLOAD_FOLDER'], today.strftime('%Y/%m/%d'))
        os.makedirs(date_path, exist_ok=True)  # exist_ok=True는 폴더가 이미 있어도 오류를 내지 않음

        # 2. 고유한 파일명 생성 (예: 150730_등기부등본.pdf)
        original_filename = secure_filename(file.filename)
        timestamp = today.strftime('%H%M%S')
        unique_filename = f"{timestamp}_{original_filename}"

        # 3. 최종 저장 경로 설정
        save_path = os.path.join(date_path, unique_filename)
        file.save(save_path)

        pdf_text = extract_text_from_pdf(save_path)
        if "등기사항전부증명서" not in pdf_text:
            # ✨ [수정 2] 유효하지 않은 파일이면 즉시 삭제
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({"error": "올바른 등기부등본 파일이 아닙니다."}), 400

        features_data = parse_register_info_detailed(pdf_text)
        test_df = pd.DataFrame([features_data])

        processed_df = pd.DataFrame(columns=training_columns, index=test_df.index)
        common_cols = [col for col in test_df.columns if col in processed_df.columns]
        processed_df[common_cols] = test_df[common_cols]
        for col, dtype in training_dtypes.items():
            if col in processed_df.columns:
                if 'int' in str(dtype): processed_df[col] = processed_df[col].fillna(0)
                try:
                    processed_df[col] = processed_df[col].astype(dtype)
                except (TypeError, ValueError):
                    if 'boolean' in str(dtype).lower():
                        processed_df[col] = processed_df[col].astype(str).str.lower().map(
                            {'true': True, 'false': False, 'nan': pd.NA}
                        ).astype('boolean')

        risk_score = loaded_model.decision_function(processed_df)[0]

        risk_percentile = "N/A"
        if train_scores is not None:
            risk_percentile = stats.percentileofscore(train_scores, risk_score)

        final_grade = get_risk_grade(risk_score, risk_percentile if risk_percentile != "N/A" else None)

        # [ ✨ 핵심 수정 ✨ ] 규칙 기반 함수 대신 OpenAI API 호출로 변경

        # 4-1. AI에게 전달할 데이터 정리
        #      processed_df는 AI가 이해하기 어려운 숫자이므로, 원본 features_data를 사용
        analysis_data = {
            "최종 분석 등급": final_grade,
            "위험도 점수": f"{risk_score:.2f}",
            "위험 백분위(%)": f"{risk_percentile:.2f}" if isinstance(risk_percentile, float) else "N/A",
            "주요 등기 정보": {
                "선순위 채권 존재 여부": features_data['선순위_채권_존재여부'],
                "압류/가압류 개수": features_data['압류_가압류_개수'],
                "근저당권 개수": features_data['근저당권_개수'],
                "신탁 등기 여부": features_data['신탁_등기여부'],
                "전입 가능 여부": features_data['전입_가능여부'],
                "우선변제권 확보 여부": features_data['우선변제권_여부'],
                "과거 전세가율": features_data.get('과거_전세가율', 'N/A')
            }
        }

        # 4-2. 프롬프트 생성
        prompt = f"""
                당신은 부동산 등기부등본 분석 결과를 일반인 사용자에게 설명해주는 AI 전문가입니다.
                아래는 분석된 등기부등본의 핵심 데이터입니다.

                데이터:
                {json.dumps(analysis_data, indent=2, ensure_ascii=False)}

                임무:
                1. 위 데이터를 바탕으로, 최종 분석 등급에 대한 이유를 설명하는 '분석 요약'을 생성하세요.
                2. 위험 요인과 안전 요인을 명확히 짚어주되, 각 요인이 왜 중요한지 구체적인 설명을 포함하여 작성하세요.
                3. 모든 내용을 종합하여 각 정보별로 같은 형식을 유지하며 한줄로 요약하세요.
                4. 분석 요약은 다음 형식을 따르세요: 번호. 등급(있다면) - 내용
                5. 줄바꿈은 전혀 없이 작성하세요. 
                """

        # 4-3. OpenAI API 호출
        response = openai.chat.completions.create(
            model="gpt-4o",  # 또는 "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "당신은 부동산 등기부등본 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,  # 일관된 결과물을 위해 온도를 낮게 설정
            max_tokens=400
        )
        analysis_summary = response.choices[0].message.content.strip()

        response_data = {
            "prediction": final_grade,
            "risk_score": f"{risk_score:.4f}",
            "risk_probability": f"{risk_percentile:.2f}%" if isinstance(risk_percentile, float) else risk_percentile,
            "analysis_summary": analysis_summary,
            "all_features": {k: str(v) for k, v in features_data.items()}
        }

        # PDF와 같은 이름으로 .json 확장자로 저장
        json_save_path = os.path.splitext(save_path)[0] + ".json"
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)

        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": "분석 중 오류가 발생했습니다.", "details": str(e)}), 500
    # finally:
    #     if save_path and os.path.exists(save_path):
    #         os.remove(save_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)