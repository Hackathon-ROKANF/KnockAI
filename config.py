# config.py
import os

# OpenAI API 키 설정
# 환경 변수에서 API 키를 가져옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 파일 업로드 폴더 경로 설정
UPLOAD_FOLDER = 'uploads'

# 머신러닝 모델 파일 경로
MODEL_PATH = 'real_estate_model.pkl'

#DB 연동
# SQLAlchemy 데이터베이스 URI
# MariaDB 형식: "mysql+pymysql://유저이름:비밀번호@호스트주소:포트/DB이름?charset=utf8mb4"
DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///local_database.db")  # 기본값은 SQLite 로컬 DB

# SQLAlchemy 설정
SQLALCHEMY_DATABASE_URI = DATABASE_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False