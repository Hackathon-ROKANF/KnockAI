# pdf_processor.py
import re
import fitz  # PyMuPDF
import logging
from datetime import datetime
from utils import korean_to_int

# --- ✨ 커스텀 예외 클래스 정의 ---
class PDFProcessingError(Exception):
    """PDF 처리 중 발생하는 오류를 위한 커스텀 예외"""
    pass

def extract_text_from_pdf(pdf_path):
    """PDF 파일 경로를 받아 텍스트를 추출하고, 실패 시 PDFProcessingError를 발생시킵니다."""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        if not text.strip():
            raise PDFProcessingError("PDF에서 텍스트를 추출할 수 없습니다. 빈 파일이거나 이미지로만 구성되어 있을 수 있습니다.")
        return text
    except Exception as e:
        # fitz 라이브러리 오류 또는 기타 예외를 잡아 커스텀 예외로 변환
        logging.error(f"fitz 라이브러리 오류 발생: {e}")
        raise PDFProcessingError(f"PDF 파일을 열거나 읽는 중 오류가 발생했습니다: {e}")


def parse_register_info_detailed(text):
    """추출된 텍스트에서 등기부등본의 상세 정보를 파싱합니다."""
    # (기존 파싱 로직은 동일하게 유지)
    features = {
        '건축물_유형': None, '근저당권_개수': 0, '채권최고액': 0, '근저당권_설정일_최근': None,
        '신탁_등기여부': False, '압류_가압류_개수': 0, '선순위_채권_존재여부': False,
        '전입_가능여부': True, '우선변제권_여부': True, '주소': None, '전세가': None,
        '매매가': None, '전세가율': None, '과거_매매가': None, '과거_전세가': None, '과거_전세가율': None,
    }

    # 주소 추출
    addr_match = re.search(r'\[\s*(?:건물|집합건물)\s*\]\s*(.+?)\n', text)
    if addr_match:
        features['주소'] = addr_match.group(1).strip()

    # 건축물 유형 파악
    building_type_list = ['아파트', '빌라', '오피스텔', '다세대주택', '단독주택', '연립주택', '다가구주택']
    features['건축물_유형'] = '기타'
    for b_type in building_type_list:
        if b_type in text:
            features['건축물_유형'] = b_type
            break

    # 갑구 분석 (소유권 관련)
    gapgu_section_match = re.search(r'【\s*갑\s*구\s*】([\s\S]+?)(?=【\s*을\s*구\s*】|--\s*이\s*하\s*여\s*백\s*--)', text)
    if gapgu_section_match:
        gapgu_section = gapgu_section_match.group(1)
        trusts, seizures = {}, {}
        gapgu_entries = re.split(r'\n(?=\d+\s)', gapgu_section)

        for entry_text in gapgu_entries:
            entry_text = entry_text.strip()
            if not entry_text: continue
            num_match = re.match(r'(\d+)', entry_text)
            if not num_match: continue
            num = num_match.group(1)
            content = entry_text[len(num):].strip()

            if '말소' in content:
                target_nums = re.findall(r'(\d+)번', content)
                for target_num in target_nums:
                    if target_num in trusts: trusts[target_num]['active'] = False
                    if target_num in seizures: seizures[target_num]['active'] = False
                continue

            if '소유권이전' in content and '신탁' in content:
                trusts[num] = {'active': True}
            if any(keyword in content for keyword in ['압류', '가압류', '경매개시결정']):
                seizures[num] = {'active': True}

        features['신탁_등기여부'] = any(t.get('active') for t in trusts.values())
        features['압류_가압류_개수'] = sum(1 for s in seizures.values() if s.get('active'))
        trade_prices = re.findall(r"거래가액\s*금([일이삼사오육칠팔구십백천만억조\d,\s]+)(?:원|정)", gapgu_section)
        if trade_prices:
            features['과거_매매가'] = korean_to_int(trade_prices[-1])

    # 을구 분석 (소유권 이외의 권리)
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

        active_mortgages = [m for m in mortgages.values() if m.get('active')]
        features['근저당권_개수'] = len(active_mortgages)
        if active_mortgages:
            features['채권최고액'] = sum(m['amount'] for m in active_mortgages)
            features['근저당권_설정일_최근'] = max(m['date'] for m in active_mortgages).strftime('%Y-%m-%d')
        active_leases = [l for l in leases.values() if l.get('active')]
        if active_leases:
            features['과거_전세가'] = active_leases[-1]['amount']

    # 최종 피처 계산
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