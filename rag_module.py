import os
import google.generativeai as genai
import tensorflow_hub as hub
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


class RAGChatbot:
    def __init__(self, api_key_var, model_name, document_path):
        load_dotenv()

        api_key = os.getenv(api_key_var)
        if not api_key:
            raise ValueError(f"{api_key_var} 환경 변수가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # 임베딩 로드
        self.embed_model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )
        self.document_chunks, self.document_embeddings = self._load_and_embed_document(
            document_path
        )

    def _load_and_embed_document(self, docunment_path):
        if not os.path.exists(docunment_path):
            raise FileNotFoundError(f"{docunment_path} 파일을 찾을수 없습니다.")

        with open(docunment_path, "r", encoding="utf-8") as f:
            document = f.read()

        # 긴 문서내용을 작은 덩어리(chunk)로 나누는 도구 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(document)
        # 각 덩어리를 벡터로 변환
        embeddings = self.embed_model(chunks)
        return chunks, embeddings

    def generate_response(self, user_input):

        # 사용자 질문을 벡터로 변환
        user_embedding = self.embed_model([user_input])
        similarities = np.dot(user_embedding, np.transpose(self.document_embeddings))
        most_similar_chunk_index = np.argmax(similarities)

        relevant_chunk = self.document_chunks[most_similar_chunk_index]

        prompt = (
            "다음 참고 자료를 바탕으로 질문에 답변하세요. 만약 참고 자료에 답변이 없다면 "
            "'참고 자료에 답변이 없습니다.'라고 응답하세요. "
            "참고 자료는 다음과 같습니다.\n\n"
            f"참고 자료: {relevant_chunk}\n\n"
            f"질문: {user_input}"
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"API 호줄중 에러 발생 : {e}"
