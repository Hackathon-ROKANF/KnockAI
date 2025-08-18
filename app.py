import os
import re
import fitz  # PyMuPDF
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename


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
    addr_match = re.search(r'\[\s*집합건물\s*\]\s*(.+?)\n', text)
    if addr_match: features['주소'] = addr_match.group(1).strip()
    building_type_list = ['아파트', '빌라', '오피스텔', '다세대주택', '단독주택', '연립주택', '다가구주택']
    features['건축물_유형'] = '기타'
    for b_type in building_type_list:
        if b_type in text:
            features['건축물_유형'] = b_type
            break
    gapgu_section_match = re.search(r'【\s*갑\s*구\s*】([\s\S]+?)(?=【\s*을\s*구\s*】|--\s*이\s*하\s*여\s*백\s*--)', text)
    if gapgu_section_match:
        gapgu_section = gapgu_section_match.group(1)
        trade_prices = re.findall(r"거래가액\s*금([일이삼사오육칠팔구십백천만억조\d,\s]+)(?:원|정)", gapgu_section)
        if trade_prices: features['과거_매매가'] = korean_to_int(trade_prices[-1])
        seizures_list = re.findall(r"가압류|압류", gapgu_section)
        features['압류_가압류_개수'] = len(seizures_list)
    eulgu_section_match = re.search(r'【\s*을\s*구\s*】([\s\S]+?)(?=--\s*이\s*하\s*여\s*백\s*--|$)', text)
    if eulgu_section_match:
        eulgu_section = eulgu_section_match.group(1)
        entries_text = re.split(r'\n(?=\d+(?:-\d+)?\s)', eulgu_section)
        mortgages, leases = {}, {}
        for entry_text in entries_text[1:]:
            entry_text = entry_text.strip()
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
        if active_leases: features['과거_전세가'] = active_leases[-1]['amount']
    if '신탁원부' in text or '신탁등기' in text: features['신탁_등기여부'] = True
    if features['근저당권_개수'] > 0 or features['압류_가압류_개수'] > 0: features['선순위_채권_존재여부'] = True
    if '경매개시결정' in text or '주택임차권' in text: features['전입_가능여부'] = False
    if features['선순위_채권_존재여부'] or not features['전입_가능여부']: features['우선변제권_여부'] = False
    if features.get('과거_매매가') and features.get('과거_전세가'):
        ratio = (features['과거_전세가'] / features['과거_매매가']) * 100
        features['과거_전세가율'] = f"{ratio:.2f}%"
    return features


# --- 2. 모델 및 메타데이터 로드 ---
try:
    with open('real_estate_model.pkl', 'rb') as f:
        saved_model_data = pickle.load(f)
    loaded_model = saved_model_data['model']
    training_columns = saved_model_data['columns']
    training_dtypes = saved_model_data['dtypes']
    print("✅ 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"❌ 모델 로딩 실패: {e}")
    loaded_model = None

# --- 3. Flask 서버 설정 ---
app = Flask(__name__)
CORS(app, resources={r"/api/analyze": {"origins": "*"}})  # 개발 편의를 위해 모든 출처 허용

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def health_check():
    return "✅ Knock AI 서버가 정상적으로 실행 중입니다!"

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
        # 1. PDF 저장 및 피처 추출
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        pdf_text = extract_text_from_pdf(save_path)
        features_data = parse_register_info_detailed(pdf_text)
        test_df = pd.DataFrame([features_data])

        # 2. PDF 유효성 검사
        if "등기사항전부증명서" not in pdf_text:
            os.remove(save_path)  # 검증 실패 시 임시 파일 즉시 삭제
            return jsonify({"error": "올바른 등기부등본 파일이 아닙니다."}), 400
        # ---------------------------

        # 2. 예측을 위한 데이터 전처리 (학습 구조와 동일하게)
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

        # 3. 예측 수행
        prediction = loaded_model.predict(processed_df)[0]
        prediction_proba = loaded_model.predict_proba(processed_df)[0]
        risk_probability = prediction_proba[1]

        # 4. SHAP 분석 및 근거 생성
        base_model_pipeline = loaded_model.estimators_[0]
        preprocessor = base_model_pipeline.named_steps['preprocessor']
        classifier = base_model_pipeline.named_steps['classifier']
        explainer = shap.TreeExplainer(classifier)
        test_data_transformed = preprocessor.transform(processed_df)
        transformed_feature_names = preprocessor.get_feature_names_out()
        shap_values = explainer(test_data_transformed)

        shap_instance_values = shap_values.values[0, :, 1]
        df_shap = pd.DataFrame(shap_instance_values, index=transformed_feature_names, columns=['shap_value'])

        original_numerical = ['채권최고액', '과거_매매가', '과거_전세가', '근저당권_개수', '압류_가압류_개수']
        original_categorical = ['건축물_유형', '신탁_등기여부', '선순위_채권_존재여부', '전입_가능여부', '우선변제권_여부']

        def get_original_feature_name(t_name):
            parts = t_name.split('__')
            if len(parts) > 1:
                f_part = parts[1]
                for c_name in original_categorical:
                    if f_part.startswith(c_name): return c_name
                if f_part in original_numerical: return f_part
            return t_name

        df_shap['feature_group'] = [get_original_feature_name(name) for name in df_shap.index]
        shap_sum_by_feature = df_shap.groupby('feature_group')['shap_value'].sum()
        risk_factors = shap_sum_by_feature[shap_sum_by_feature > 0].sort_values(ascending=False)
        safe_factors = shap_sum_by_feature[shap_sum_by_feature < 0].sort_values()

        analysis_summary = ""
        if risk_probability >= 0.5:
            analysis_summary += "이 등기부등본은 '위험'으로 예측됩니다.\\n"
            if not risk_factors.empty:
                top_risk = ", ".join([f"'{name}'" for name in risk_factors.head(2).index])
                analysis_summary += f"주된 위험 요인은 {top_risk} 등입니다."
        else:
            analysis_summary += "이 등기부등본은 '안전'으로 예측됩니다.\\n"
            if not safe_factors.empty:
                top_safe = ", ".join([f"'{name}'" for name in safe_factors.head(2).index])
                analysis_summary += f"주된 안전 요인은 {top_safe} 등입니다."

        os.remove(save_path)  # 임시 파일 삭제

        # 5. 최종 결과 JSON으로 반환
        return jsonify({
            "prediction": "위험" if prediction == 1 else "안전",
            "risk_probability": f"{risk_probability * 100:.2f}%",
            "analysis_summary": analysis_summary,
            "all_features": {k: str(v) for k, v in features_data.items()}  # 모든 피처 정보도 함께 반환
        }), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": "분석 중 오류가 발생했습니다.", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)