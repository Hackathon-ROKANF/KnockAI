# models.py (새 파일)

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# SQLAlchemy 객체 생성 (app.py에서 초기화)
db = SQLAlchemy()

class AnalysisResult(db.Model):
    """
    등기부등본 분석 결과를 저장하는 테이블
    """
    __tablename__ = 'analysis_results'

    id = db.Column(db.Integer, primary_key=True)
    prediction = db.Column(db.String(50), nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_probability = db.Column(db.String(50))
    summary_json = db.Column(db.JSON) # 분석 요약 JSON
    features_json = db.Column(db.JSON) # 전체 특징 JSON
    file_path = db.Column(db.String(255)) # 원본 파일 경로
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AnalysisResult id={self.id} prediction='{self.prediction}'>"