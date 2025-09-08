# 날씨 검색 도구

def search_weather(location: str):

    weather_data = {
        '서울' : '맑음, 25도',
        '부산' : '흐림, 13도',
        '뉴욕' : '비, 15도'
    }

    if location in weather_data:
        return f"{location}의 현재 날씨는 {weather_data[location]}입니다."
    else:
        return "해당 도시의 날씨 정보를 찾을 수 없습니다."