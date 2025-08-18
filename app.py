import os
import re
import fitz  # PyMuPDF
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename


# --- 1. PDF 분석 로직 (기존 코드) ---

def korean_to_int(kstr):
    """'일억이천만원' 같은 한글 금액 문자열을 정수형 숫자로 변환합니다."""
    kstr = str(kstr).replace(",", "").replace(" ", "").strip()
    if kstr.isdigit():
        return int(kstr)

    num_map = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    unit_map = {'십': 10, '백': 100, '천': 1000}
    large_unit_map = {'만': 10000, '억': 100000000, '조': 1000000000000}

    total_sum = 0
    temp_sum = 0
    current_num = 0

    for char in kstr:
        if char in num_map:
            current_num = num_map[char]
        elif char in unit_map:
            if current_num == 0: current_num = 1
            temp_sum += current_num * unit_map[char]
            current_num = 0
        elif char in large_unit_map:
            if current_num != 0:
                temp_sum += current_num
            if temp_sum == 0: temp_sum = 1
            total_sum += temp_sum * large_unit_map[char]
            temp_sum = 0
            current_num = 0

    total_sum += temp_sum + current_num
    return total_sum


def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 모든 텍스트를 추출하여 하나의 문자열로 반환합니다."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def parse_register_info_detailed(text):
    """등기부등본 텍스트에서 요청된 상세 피처를 추출하고 JSON으로 반환 가능한 형태로 가공합니다."""
    features = {
        '건축물_유형': None, '근저당권_개수': 0, '채권최고액': 0, '근저당권_설정일_최근': None,
        '신탁_등기여부': False, '압류_가압류_개수': 0, '선순위_채권_존재여부': False,
        '전입_가능여부': True, '우선변제권_여부': True, '주소': None, '전세가': None,
        '매매가': None, '전세가율': None, '과거_매매가': None, '과거_전세가': None, '과거_전세가율': None,
    }

    addr_match = re.search(r'\[\s*집합건물\s*\]\s*(.+?)\n', text)
    if addr_match:
        features['주소'] = addr_match.group(1).strip()

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
        if trade_prices:
            features['과거_매매가'] = korean_to_int(trade_prices[-1])
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
        if active_leases:
            features['과거_전세가'] = active_leases[-1]['amount']

    if '신탁원부' in text or '신탁등기' in text: features['신탁_등기여부'] = True
    if features['근저당권_개수'] > 0 or features['압류_가압류_개수'] > 0: features['선순위_채권_존재여부'] = True
    if '경매개시결정' in text or '주택임차권' in text: features['전입_가능여부'] = False
    if features['선순위_채권_존재여부'] or not features['전입_가능여부']: features['우선변제권_여부'] = False
    if features.get('과거_매매가') and features.get('과거_전세가'):
        ratio = (features['과거_전세가'] / features['과거_매매가']) * 100
        features['과거_전세가율'] = f"{ratio:.2f}%"

    return features


# --- 2. Flask 서버 설정 ---

app = Flask(__name__)
# React 앱이 실행될 주소(보통 http://localhost:3000)로부터의 요청을 허용합니다.
CORS(app, resources={r"/analyze": {"origins": "http://localhost:3000"}})

# 임시 PDF 파일을 저장할 폴더 (기존 data 폴더를 활용)
UPLOAD_FOLDER = 'data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # PDF 텍스트 추출 및 분석
            register_text = extract_text_from_pdf(save_path)
            analysis_result = parse_register_info_detailed(register_text)

            # 임시 파일 삭제
            os.remove(save_path)

            # 분석 결과를 JSON으로 반환
            return jsonify(analysis_result), 200

        except Exception as e:
            # 에러 발생 시 서버 로그에 에러 기록
            app.logger.error(f"An error occurred: {e}")
            return jsonify({"error": "PDF 분석 중 오류가 발생했습니다.", "details": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDF is allowed."}), 400


if __name__ == '__main__':
    # host='0.0.0.0'는 외부에서도 접속 가능하게 합니다 (필요에 따라 '127.0.0.1'로 변경).
    app.run(host='0.0.0.0', port=5000, debug=True)