# summary_generator.py
import openai
import json
import config

# OpenAI API 키 설정
openai.api_key = config.OPENAI_API_KEY
if not openai.api_key:
    print("❌ [경고] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

def generate_summary(features_data, final_grade, risk_score, risk_percentile):
    """OpenAI API를 호출하여 분석 결과를 요약합니다."""
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

    prompt = (
        "당신은 부동산 등기부등본 분석 결과를 일반인 사용자에게 설명해주는 AI 전문가입니다. "
        "아래는 분석된 등기부등본의 핵심 데이터입니다.\n"
        f"데이터: {json.dumps(analysis_data, ensure_ascii=False)}\n"
        "임무:\n"
        "1. 위 데이터를 바탕으로, 최종 분석 등급에 대한 이유를 설명하는 '분석 요약'을 생성하세요.\n"
        "2. 위험 요인과 안전 요인을 합쳐서 6개 혹은 7개로 명확히 짚어주되, 각 요인이 왜 중요한지 구체적인 설명을 포함하여 작성하세요.\n"
        "3. 위험 요인은 위험 요인끼리, 안전 요인은 안전 요인끼리 묶어서 설명하세요.\n"
        "4. 모든 내용을 종합하여 각 정보별로 같은 형식을 유지하며 한줄로 요약하세요.\n"
        "5. 분석 요약은 반드시 아래 예시처럼 JSON 형식(key-value 쌍)으로만 출력하세요. 동일한 요인이 여러 개일 경우 반드시 리스트로 묶어서 출력하세요. 예시: {\"관심\": \"최종 분석 등급이 '관심'인 이유는 ...\", \"위험 요인\": [\"...\", \"...\"], \"안전 요인\": [\"...\", \"...\"]}\n"
        "6. 순서는 최종 분석 등급이 가장 앞에 오도록 하고, 위험 요인과 안전 요인은 그 다음에 오도록 하세요.\n"
        "7. 줄바꿈 없이 한 줄로만 출력하세요.\n"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 부동산 등기부등본 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=400
        )
        analysis_summary = response.choices[0].message.content.strip()
        return json.loads(analysis_summary)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"error": "AI 응답을 JSON으로 변환하지 못했습니다.", "raw": str(e)}