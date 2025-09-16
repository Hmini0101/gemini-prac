from rag_module import RAGChatbot


def main():
    try:
        chatbot = RAGChatbot(
            api_key_var="GEMINI_API_KEY",
            model_name="gemini-1.5-flash",
            data_paths=[
                "manual.txt",
                "https://ko.wikipedia.org/wiki/%ED%8C%8C%EC%9D%B4%EC%8D%AC",
            ],
        )

        print("문서기반 챗봇 시작! , '종료' 입력하면 끝납니다.")

        chat_history = []

        while True:
            user_input = input("나: ")
            if user_input == "종료":

                print("내용 기록")
                for msg in chat_history:
                    role = msg["role"]
                    text = msg["parts"][0]["text"]
                    print(f"역할: {role}, 내용 : {text}")

                break

            chat_history.append({"role": "user", "parts": [{"text": user_input}]})
            response = chatbot.generate_response(user_input, chat_history)
            chat_history.append({"role": "model", "parts": [{"text": response}]})
            print("Gemini: ", response)

            if len(chat_history) > 4:
                chat_history = chat_history[-4:]

    except Exception as e:
        print(f"오류가 발생했습니다. {e}")


if __name__ == "__main__":
    main()
