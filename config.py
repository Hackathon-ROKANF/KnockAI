# config.py
import os

# OpenAI API 키 설정
# 환경 변수에서 API 키를 가져옵니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 파일 업로드 폴더 경로 설정
UPLOAD_FOLDER = 'uploads'

# 머신러닝 모델 파일 경로
MODEL_PATH = 'real_estate_model.pkl'