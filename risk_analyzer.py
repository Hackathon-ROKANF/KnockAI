# risk_analyzer.py
import pickle
import logging
from scipy import stats
import pandas as pd

class ModelAnalysisError(Exception):
    """모델 로딩 또는 분석 중 발생하는 오류를 위한 커스텀 예외"""
    pass


def load_model(model_path):
    """지정된 경로에서 머신러닝 모델과 관련 데이터를 로드합니다."""
    try:
        with open(model_path, 'rb') as f:
            saved_model_data = pickle.load(f)
        logging.info("✅ 모델과 관련 데이터가 성공적으로 로드되었습니다.")
        return saved_model_data
    except FileNotFoundError:
        logging.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    except (pickle.UnpicklingError, KeyError) as e:
        logging.error(f"❌ 모델 파일을 로드하는 데 실패했습니다 (파일 손상 가능성): {e}")
        return None


def get_risk_grade(risk_score, percentile):
    """위험 점수와 백분위를 기반으로 최종 위험 등급을 반환합니다."""
    # (기존 로직과 동일)
    if risk_score <= 0:
        return "안전"
    if percentile is None or not isinstance(percentile, (int, float)):
        return "확인 필요"
    if percentile < 33.3:
        return "관심"
    elif percentile < 66.6:
        return "주의"
    else:
        return "위험"


def analyze_risk(model_data, features_data):
    """파싱된 데이터를 모델에 입력하여 위험 점수, 백분위, 등급을 계산합니다."""
    try:
        if not model_data:
            raise ModelAnalysisError("모델 데이터가 없습니다. 서버 시작 시 모델 로딩에 실패했을 수 있습니다.")

        loaded_model = model_data['model']
        training_columns = model_data['columns']
        training_dtypes = model_data['dtypes']
        train_scores = model_data.get('train_scores')

        test_df = pd.DataFrame([features_data])
        processed_df = pd.DataFrame(columns=training_columns, index=test_df.index)

        # ... (기존 데이터프레임 처리 로직) ...
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

        risk_score = loaded_model.decision_function(processed_df)[0]

        risk_percentile = "N/A"
        if train_scores is not None:
            risk_percentile = stats.percentileofscore(train_scores, risk_score)

        final_grade = get_risk_grade(risk_score, risk_percentile if risk_percentile != "N/A" else None)

        return risk_score, risk_percentile, final_grade
    except (KeyError, ValueError, TypeError) as e:
        # 데이터 처리 또는 모델 예측 중 발생할 수 있는 오류
        logging.error(f"모델 분석 중 데이터 관련 오류 발생: {e}")
        raise ModelAnalysisError(f"분석에 필요한 데이터 형식이 올바르지 않습니다: {e}")
    except Exception as e:
        # 그 외 예측하지 못한 오류
        logging.error(f"모델 분석 중 예측하지 못한 오류 발생: {e}", exc_info=True)
        raise ModelAnalysisError(f"분석 중 예측하지 못한 오류가 발생했습니다: {e}")