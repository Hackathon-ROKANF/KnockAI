# app.py
import os
import json
import logging
import jwt
from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from functools import wraps

# 모듈화된 기능들 임포트
import config
from pdf_processor import extract_text_from_pdf, parse_register_info_detailed, PDFProcessingError
from risk_analyzer import load_model, analyze_risk, ModelAnalysisError
from summary_generator import generate_summary
from models import db, AnalysisResult

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-default-secret-key")


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            # Bearer 토큰 형식 "Bearer <token>"
            token = request.headers['Authorization'].split(" ")[1]

        if not token:
            return jsonify({'message': '토큰이 존재하지 않습니다.'}), 401

        try:
            data = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
            current_user_id = data['sub']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': '토큰이 만료되었습니다.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': '유효하지 않은 토큰입니다.'}), 401

        return f(current_user_id, *args, **kwargs)

    return decorated

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def custom_serializer(*args, **kwargs):
    """ensure_ascii=False를 기본값으로 사용하는 커스텀 JSON 직렬 변환기"""
    kwargs['ensure_ascii'] = False
    return json.dumps(*args, **kwargs)

# Flask 앱 초기화
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 한글 깨짐 방지
# --- 🔽 DB 설정 로드 🔽 ---
app.config.from_object(config)

app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'json_serializer': custom_serializer
}

# --- 🔽 DB 초기화 🔽 ---
db.init_app(app)

CORS(app, resources={r"/api/analyze": {"origins": "*"}})

# 업로드 폴더 설정
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 시작 시 모델 로드
model_data = load_model(config.MODEL_PATH)

# --- ✨ 전역 에러 핸들러 추가 ---
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "요청한 리소스를 찾을 수 없습니다."}), 404

@app.errorhandler(500)
def internal_server_error(error):
    # 실제 운영 환경에서는 더 일반적인 메시지를 반환하는 것이 좋습니다.
    return jsonify({"error": "서버 내부에서 오류가 발생했습니다.", "details": str(error)}), 500


@app.route('/')
def health_check():
    """서버 상태 확인 엔드포인트"""
    return "✅ Knock AI (One-Class SVM) 서버가 정상적으로 실행 중입니다!"


@app.route('/api/analyze', methods=['POST'])
@token_required
def analyze_pdf_endpoint(current_user_id):
    """PDF 파일 분석 API 엔드포인트"""
    if not model_data:
        logging.error("모델 데이터가 로드되지 않아 분석 요청을 처리할 수 없습니다.")
        return jsonify({"error": "모델이 서버에 정상적으로 로드되지 않았습니다. 관리자에게 문의하세요."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "파일이 요청에 포함되지 않았습니다."}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "PDF 파일이 아니거나 선택되지 않았습니다."}), 400

    save_path = None
    try:
        # 1. 파일 저장
        today = datetime.now()
        date_path = os.path.join(app.config['UPLOAD_FOLDER'], today.strftime('%Y/%m/%d'))
        os.makedirs(date_path, exist_ok=True)
        original_filename = secure_filename(file.filename)
        timestamp_str = today.strftime('%H%M%S')
        unique_filename = f"{timestamp_str}_{original_filename}"
        save_path = os.path.join(date_path, unique_filename)
        file.save(save_path)

        # 2. PDF 처리 및 유효성 검사 (✨ 예외 처리된 모듈 사용)
        pdf_text = extract_text_from_pdf(save_path)
        if "등기사항전부증명서" not in pdf_text:
            return jsonify({"error": "업로드된 파일이 유효한 등기부등본이 아닙니다."}), 400

        # 3. 등기 정보 파싱 및 위험 분석 (✨ 예외 처리된 모듈 사용)
        features_data = parse_register_info_detailed(pdf_text)
        risk_score, risk_percentile, final_grade = analyze_risk(model_data, features_data)

        # 4. AI 기반 요약 생성 (✨ 예외 처리된 모듈 사용)
        # 이 함수는 내부적으로 API 오류 등을 처리하고, 실패 시에도 기본 정보를 담은 dict를 반환합니다.
        analysis_summary = generate_summary(features_data, final_grade, risk_score, risk_percentile)

        # 5. 최종 응답 데이터 구성
        response_data = {
            "timestamp": today.isoformat(),
            "prediction": final_grade,
            "risk_score": f"{risk_score:.4f}",
            "risk_probability": f"{risk_percentile:.2f}%" if isinstance(risk_percentile, float) else risk_percentile,
            "analysis_summary": analysis_summary,
            "all_features": {k: str(v) for k, v in features_data.items()}
        }

        # DB에 결과 저장
        new_result = AnalysisResult(
            prediction=final_grade,
            risk_score=risk_score,
            risk_probability=f"{risk_percentile:.2f}%" if isinstance(risk_percentile, float) else risk_percentile,
            summary_json=analysis_summary,
            features_json={k: str(v) for k, v in features_data.items()},
            file_path=save_path,
            user_id=current_user_id
        )
        db.session.add(new_result)
        db.session.commit()

        # 6. 분석 결과 저장
        json_save_path = os.path.splitext(save_path)[0] + ".json"
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=4)

        # ✨ 새로운 코드: 직접 JSON 응답 생성
        json_response = json.dumps(response_data, ensure_ascii=False, indent=4)
        return Response(json_response, content_type='application/json; charset=utf-8'), 200

    # ✨ 각 모듈에서 발생시킨 커스텀 예외를 잡아 구체적인 오류 메시지 반환
    except PDFProcessingError as e:
        logging.warning(f"PDF 처리 실패: {e}")
        return jsonify({"error": "PDF 파일을 처리할 수 없습니다.", "details": str(e)}), 400
    except ModelAnalysisError as e:
        logging.error(f"모델 분석 실패: {e}")
        return jsonify({"error": "위험도 분석 중 오류가 발생했습니다.", "details": str(e)}), 500
    except IOError as e:
        logging.error(f"파일 저장/쓰기 오류: {e}")
        return jsonify({"error": "파일을 서버에 저장하는 중 오류가 발생했습니다.", "details": str(e)}), 500
    except Exception as e:
        # 그 외 예측하지 못한 모든 오류 처리
        logging.error(f"분석 엔드포인트에서 예측하지 못한 오류 발생: {e}", exc_info=True)
        return jsonify({"error": "분석 중 예측하지 못한 오류가 발생했습니다.", "details": "관리자에게 문의하세요."}), 500
    finally:
        # 분석 성공/실패 여부와 관계없이 임시 파일 삭제
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                logging.info(f"임시 파일 삭제 완료: {save_path}")
            except OSError as e:
                logging.error(f"임시 파일 삭제 실패: {e}")


if __name__ == '__main__':
    # 운영 환경에서는 debug=False로 설정하는 것을 권장합니다.
    app.run(host='0.0.0.0', port=5000, debug=False)