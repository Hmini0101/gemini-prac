import os
import google.generativeai as genai
import traceback
from dotenv import load_dotenv

# RAG에 필요한 라이브러리 임포트
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tensorflow_hub as hub
import numpy as np

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


def chat_with_gemini():

    print("문서 기반 챗봇을 시작합니다. '종료'를 입력하면 끝납니다.")

    # --- 1. 문서 로딩 및 분할 ---
    try:
        with open("manual.txt", "r", encoding="utf-8") as f:
            document_text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        document_chunks = text_splitter.split_text(document_text)
        print(f"문서를 {len(document_chunks)}개 덩어리로 나눴습니다.")
    except FileNotFoundError:
        print("오류: 'manual.txt' 파일을 찾을 수 없습니다. 파일을 생성해주세요.")
        return

    # --- 2. 임베딩 모델 로드 및 문서 임베딩 ---
    print("⏳ 임베딩 모델을 로드하는 중입니다...")
    try:
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        document_embeddings = embed(document_chunks)
        print("✅ 임베딩 모델 로드 및 문서 임베딩 완료.")
    except Exception as e:
        print(f"오류: 임베딩 모델 로드 또는 변환 중 오류가 발생했습니다: {e}")
        return

    chat = model.start_chat(history=[])

    while True:
        user_input = input("나: ")
        if user_input.lower() == "종료":
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡ종료ㅡㅡㅡㅡㅡㅡㅡㅡ")
            break

        try:
            # RAG 관련 로직 (다음 단계)
            user_embedding = embed([user_input])
            similarties = np.inner(user_embedding, document_embeddings)

            # 가장 유사한 상위 3개 덩어리 인덱스 찾기
            top_indices = np.argsort(similarties[0])[-3:][::-1]
            relevant_chunks = [document_chunks[i] for i in top_indices]

            # gemini 에게 전달할 프롬프트
            prompt = f"""
다음 정보를 참고해서 사용자의 질문에 답해줘.
만약 참고 자료에 답이 없으면, "참고 자료에 답변이 없습니다." 라고 말해줘.
---
참고 자료:
{relevant_chunks}
---
사용자 질문: {user_input}
"""
            # 정제된 내용을 보냄
            response = chat.send_message(prompt)

            # 응답 출력 로직
            if not response.parts:
                print("Gemini: 죄송합니다. 응답을 생성할 수 없습니다.")
            else:
                print("Gemini: ", end="")
                for part in response.parts:
                    print(part.text, end="")
                print()

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            traceback.print_exc()
            break


if __name__ == "__main__":
    chat_with_gemini()
