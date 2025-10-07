import requests
import logging

# 호갱예스 API의 기본 URL 주소입니다.
HOGANG_YES_API_URL = "http://3.37.47.2"


def get_real_estate_info(address: str):
    """
    호갱예스 API를 호출하여 특정 주소의 부동산 실거래 정보를 가져옵니다.
    호갱예스의 Swagger 명세서를 참고하여 실제 엔드포인트와 파라미터를 맞춰야 합니다.

    :param address: 조회하고자 하는 주소 (예: "서울특별시 강남구 역삼동")
    :return: API 응답을 JSON 형태로 반환합니다. 에러 발생 시 None을 반환합니다.
    """
    # README에 명시된 Swagger를 참고하여 실제 API 엔드포인트로 수정해야 합니다.
    # 여기서는 예시로 /api/v1/apartments 엔드포인트를 사용합니다.
    api_endpoint = f"{HOGANG_YES_API_URL}/api/v1/apartments"
    params = {
        "address": address
    }

    try:
        response = requests.get(api_endpoint, params=params, timeout=10)

        # HTTP 응답 코드가 200 (성공)일 경우 JSON 데이터를 반환합니다.
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"호갱예스 API 에러: 상태 코드 {response.status_code}, 응답 {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"호갱예스 API 호출 중 예외 발생: {e}")
        return None


# 간단한 테스트를 위한 코드
if __name__ == '__main__':
    sample_address = "서울특별시 강남구 개포동"
    info = get_real_estate_info(sample_address)

    if info:
        print(f"'{sample_address}' 주소의 정보:")
        print(info)
    else:
        print("정보를 가져오는 데 실패했습니다.")