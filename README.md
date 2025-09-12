# 🤖 Gemini 학습
특정 문서(`manual.txt`)를 기반으로 질문에 답변하는 챗봇입니다. 검색 증강 생성(RAG) 아키텍처를 사용하여 정확하고 신뢰성 있는 답변을 제공합니다.

---

## 🚀 주요 기능 (챌린지 진행중)
- **문서 기반 답변**: 사용자의 질문에 가장 관련성 높은 문서 내용을 찾아 답변을 생성합니다.
- **환각 현상 방지**: 참고 문서에 없는 질문에는 "참고 자료에 답변이 없습니다."라고 응답하여 AI의 환각(Hallucination)을 방지합니다.
- **효율적인 검색**: 문서 내용을 벡터화하여 빠르고 정확하게 관련 정보를 검색합니다.

---

## 🛠️ 기술 스택
- **언어**: Python
- **라이브러리**:
    - LangChain (문서 분할)
    - TensorFlow Hub (임베딩)
    - NumPy (벡터 연산)
    - python-dotenv (환경 변수 관리)
- **API**: Google Gemini API

---

## 📂 프로젝트 구조
프로젝트의 기본 파일 구조는 다음과 같습니다.<br>
/프로젝트명<br>
├── .env                  # 환경 변수<br>
├── .gitignore            # Git Ignore 파일<br>
├── .vscode/              # VS Code 설정<br>
├── main.py               # 메인 코드<br>
├── rag_module.py         # 모듈(기능) 코드<br>
├── manual.txt            # 문서내용<br>
└── venv/                 # 가상 환경<br>
