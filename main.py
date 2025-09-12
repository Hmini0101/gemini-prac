from rag_module import RAGChatbot


def main():
    try:
        chatbot = RAGChatbot(
            api_key_var="GEMINI_API_KEY",
            model_name="gemini-1.5-flash",
            document_path="manual.txt",
        )

        print("문서기반 챗봇 시작! , '종료' 입력하면 끝납니다.")

        while True:
            user_input = input("나: ")
            if user_input == "종료":
                break

            response = chatbot.generate_response(user_input)
            print("Gemini: ", response)

    except Exception as e:
        print(f"오류가 발생했습니다. {e}")


if __name__ == "__main__":
    main()
