import os
import google.generativeai as genai
import tensorflow_hub as hub
import numpy as np
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


class RAGChatbot:
    def __init__(self, api_key_var, model_name, data_paths):
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

        self.document_chunks = []
        self.document_embeddings = []

        for data_path in data_paths:
            if os.path.exists(data_path):
                chunks, embeddings = self._load_and_embed_document(data_path)
                self.document_chunks.extend(chunks)
                self.document_embeddings.extend(embeddings)
            elif data_path.startswith("http"):
                chunks, embeddings = self._load_and_embed_webpage(data_path)
                self.document_chunks.extend(chunks)
                self.document_embeddings.extend(embeddings)
            else:
                print(f"경로를 찾을수 없습니다.: {data_path}")

    def _load_and_embed_document(self, document_path):
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"'{document_path}' 파일을 찾을 수 없습니다.")

        with open(document_path, "r", encoding="utf-8") as f:
            document = f.read()
        return self._process_text(document)

    def _load_and_embed_webpage(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text()
            return self._process_text(text_content)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"URL에 접근하는 중 오류가 발생했습니다. : {e}")

    def _process_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        embeddings = self.embed_model(chunks)
        return chunks, embeddings

    def generate_response(self, user_input, chat_history=[]):
        user_embedding = self.embed_model([user_input])
        similarities = np.dot(user_embedding, np.transpose(self.document_embeddings))

        top_k = 3
        top_k_indices = np.argsort(similarities.flatten())[-top_k:][::-1]

        # most_similar_chunk_index = np.argmax(similarities)
        relevant_chunks = [self.document_chunks[i] for i in top_k_indices]

        combined_chunks = "\n\n".join(relevant_chunks)

        history_list = []
        for msg in chat_history:
            role = msg["role"]
            text = msg["parts"][0]["text"]
            history_list.append(f"{role}: {text}")

        history_string = "\n".join(history_list)

        prompt = (
            "다음 참고 자료를 바탕으로 질문에 답변하세요. 만약 참고 자료에 답변이 없다면 "
            "'참고 자료에 답변이 없습니다.'라고 응답하고 너가 인터넷에서 찾아보고 대답해 "
            "참고 자료는 다음과 같습니다.\n\n"
            f"참고 자료: {combined_chunks}\n\n"
            f"질문: {user_input}"
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"API 호출 중 오류 발생: {e}"
