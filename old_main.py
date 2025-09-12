import os
import google.generativeai as genai #제미나이 모델 API 사용
import traceback
from dotenv import load_dotenv #환경 변수 사용
from tools.current_time_tool import get_current_time
from tools.weather_tool import search_weather


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=API_KEY)

# model = genai.GenerativeModel('gemini-pro')
# model = genai.GenerativeModel('gemini-1.5-pro-latest')
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    # 'gemini-2.5-flash',
    tools=[get_current_time, search_weather]
    )

def chat_with_gemini():
    print("Gemini와 대화를 시작합니다. '종료'를 입력하면 끝납니다.")

    chat = model.start_chat(history=[])

    while True:
        user_input = input("나: ")
        if user_input.lower() == '종료':
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡ종료ㅡㅡㅡㅡㅡㅡㅡㅡ")
            break

        try:
            response = chat.send_message(user_input)
            # 모델이 도구 사용을 제안 했는지 확인 , tool call 기능 확인
            # API 기술적 한계로 해당 툴 호출 불가 , 소스 내부에 수동으로 생성해야됨 참고
            # if response.tool_calls:
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_name = tool_call.name

                tool_result = None

                if tool_name == 'get_current_time':
                    tool_result = get_current_time()
                elif tool_name == 'search_weather':
                    location = tool_call.args.get('location')   # 딕셔너리로 담고 있기에 args 사용
                    tool_result = search_weather(location)

                response = chat.send_message(
                    [
                        f"도구 호출 결과 : {tool_result}"
                    ],
                    tool_results = [{'name' : tool_name, 'result': tool_result}]
                )

            # 길이가 100 넘을시 5개 삭제
            if len(chat.history) > 100:
                chat.history = chat.history[5:]

            if not response.parts:
                print("gemini : 죄송합니다. 응답 생성할 수 없습니다.")
            else:
                print("Gemini: ", end="")
                for part in response.parts:
                    print(part.text, end="")
                print()
                    
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            traceback.print_exc()
            break
    #대화 기록 확인 (메모리 기능)
    print("\n-----대화기록-----")
    for message in chat.history:
        print(f"역할: {message.role}, 내용 : {message.parts[0].text}")
if __name__ == "__main__":
    chat_with_gemini()