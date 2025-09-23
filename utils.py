# utils.py

def korean_to_int(kstr):
    """한글로 된 숫자를 정수로 변환합니다."""
    if not kstr: return 0
    kstr = str(kstr).replace(",", "").replace(" ", "").strip()
    if kstr.isdigit(): return int(kstr)

    num_map = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    unit_map = {'십': 10, '백': 100, '천': 1000}
    large_unit_map = {'만': 10000, '억': 100000000, '조': 1000000000000}

    total_sum, temp_sum, current_num = 0, 0, 0
    for char in kstr:
        if char in num_map:
            current_num = num_map[char]
        elif char in unit_map:
            temp_sum += (current_num if current_num else 1) * unit_map[char]
            current_num = 0
        elif char in large_unit_map:
            temp_sum += current_num
            total_sum += (temp_sum if temp_sum else 1) * large_unit_map[char]
            temp_sum, current_num = 0, 0

    total_sum += temp_sum + current_num
    return total_sum