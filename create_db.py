# create_db.py (새 파일)

from app import app, db

# app 컨텍스트 내에서 실행
with app.app_context():
    # 정의된 모든 모델에 대해 테이블 생성
    db.create_all()

print("✅ 데이터베이스 테이블이 성공적으로 생성되었습니다.")