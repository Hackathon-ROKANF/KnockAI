# app.py
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 모듈화된 기능들 임포트
import config
from pdf_processor import extract_text_from_pdf, parse_register_info_detailed
from risk_analyzer import load_model, analyze_risk
from summary_generator import generate_summary

# Flask 앱 초기화
app = Flask(__name__)
CORS(app, resources={r"/api/analyze": {"origins": "*"}})

# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 시작 시 모델 로드
model_data = load_model(config.MODEL_PATH)

@app.route('/')
def health_check():
    """서버 상태 확인 엔드포인트"""
    return "✅ Knock AI (One-Class SVM) 서버가 정상적으로 실행 중입니다!"

@app.route('/api/analyze', methods=['POST'])
def analyze_pdf_endpoint():
    """PDF 파일 분석 API 엔드포인트"""
    if not model_data:
        return jsonify({"error": "서버에 모델이 로드되지 않았습니다."}), 500
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return jsonify({"error": "PDF 파일이 아니거나 선택되지 않았습니다."}), 400

    save_path = None
    try:
        # 1. 파일 저장 경로 및 이름 생성
        today = datetime.now()
        date_path = os.path.join(app.config['UPLOAD_FOLDER'], today.strftime('%Y/%m/%d'))
        os.makedirs(date_path, exist_ok=True)
        original_filename = secure_filename(file.filename)
        timestamp_str = today.strftime('%H%M%S')
        unique_filename = f"{timestamp_str}_{original_filename}"
        save_path = os.path.join(date_path, unique_filename)
        file.save(save_path)

        # 2. PDF 텍스트 추출 및 유효성 검사
        pdf_text = extract_text_from_pdf(save_path)
        if "등기사항전부증명서" not in pdf_text:
            if os.path.exists(save_path):
                os.remove(save_path)
            return jsonify({"error": "올바른 등기부등본 파일이 아닙니다."}), 400

        # 3. 등기 정보 파싱 및 위험 분석
        features_data = parse_register_info_detailed(pdf_text)
        risk_score, risk_percentile, final_grade = analyze_risk(model_data, features_data)

        # 4. AI 기반 요약 생성
        analysis_summary = generate_summary(features_data, final_grade, risk_score, risk_percentile)

        # 5. 최종 응답 데이터 구성 (타임스탬프 추가)
        response_data = {
            "timestamp": today.isoformat(),  # 요청 처리 시간
            "prediction": final_grade,
            "risk_score": f"{risk_score:.4f}",
            "risk_probability": f"{risk_percentile:.2f}%" if isinstance(risk_percentile, float) else risk_percentile,
            "analysis_summary": analysis_summary,
            "all_features": {k: str(v) for k, v in features_data.items()}
        }

        # 6. 분석 결과 JSON 파일로 저장
        json_save_path = os.path.splitext(save_path)[0] + ".json"
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)

        return jsonify(response_data), 200

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": "분석 중 오류가 발생했습니다.", "details": str(e)}), 500
    # finally:
    #     # 필요 시 분석 후 PDF 파일 삭제
    #     if save_path and os.path.exists(save_path):
    #         os.remove(save_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)