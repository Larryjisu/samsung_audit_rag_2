# samsung_audit_rag_structured_risk

삼성전자 감사보고서(2014~2024)를 대상으로, **HTML 감사보고서를 구조적으로 파싱하고**, **표와 주석을 DB에 적재한 뒤**, **질문에 대해 주석 우선 탐색 방식으로 답변하는 5조의 사이드 RAG 프로젝트**입니다.

---

## 1. 이 프로젝트가 하는 일

이 프로젝트의 목표는 단순히 감사보고서 텍스트를 검색하는 것이 아닙니다.

감사보고서 안에는 다음처럼 서로 성격이 다른 정보가 섞여 있습니다.

- 긴 설명형 텍스트
- 주석
- 복잡한 표
- 표 안의 셀 단위 숫자
- 특정 리스크를 나타내는 문장

그래서 이 프로젝트는 문서를 한 번에 뭉뚱그려 다루지 않고, 다음 두 방향을 함께 고려합니다.

1. **텍스트 청크 검색**: 설명형 질문이나 주석 중심 질문 대응
2. **구조화 표 조회**: 금액, 총액, 잔액, 특정 항목 숫자 질문 대응

즉, 이 프로젝트는

> **“감사보고서를 잘게 구조화해서 저장하고, 질문에 맞는 근거를 찾아 답하는 시스템”**

이라고 이해하면 됩니다.

---

## 2. 핵심 아이디어 한 줄 요약

질문이 들어오면:

- 질문에서 **연도, 주석 제목, 세부 항목, 숫자 단서**를 먼저 잡고
- 관련된 **텍스트 청크**와 **구조화된 표/셀**을 함께 탐색한 뒤
- 가장 적절한 근거를 조합해서 답변합니다.

즉, 이 프로젝트는 단순 키워드 검색기가 아니라,

> **“문장과 표를 함께 읽는 감사보고서 질의응답 시스템”**

입니다.

---

## 3. 전체 처리 흐름

프로젝트 전체 흐름은 아래 순서입니다.

```text
감사보고서 HTML
   ↓
HTML 파싱 (parser.py)
   ↓
구조화 JSON 생성
   ↓
텍스트 청킹 + 표 구조화 (chunker.py)
   ↓
임베딩 생성 (embedder.py)
   ↓
PostgreSQL + pgvector 적재 (ingest.py)
   ↓
질문 입력
   ↓
Local 검색
   ↓
근거 검색
   ↓
답변 생성
```

---

## 4. 폴더 구조

```text
samsung_audit_rag_structured_risk/
├─ app/
│  ├─ config.py              # 환경변수/설정 로드
│  ├─ db.py                  # PostgreSQL 연결
│  ├─ parser.py              # HTML → 구조화 report 객체
│  ├─ chunker.py             # 텍스트 청크 / 구조화 표 생성
│  ├─ embedder.py            # 임베딩 모델 로드 및 벡터 생성
│  ├─ ingest.py              # 전체 적재 파이프라인 실행
│  ├─ search_local.py        # 정밀 검색(숫자형/항목형)
│  ├─ search.py              # local search 재-export
│  ├─ qa_local.py            # local 검색 결과를 발표용 답변으로 포맷
│  ├─ qa.py                  # 개발용 QA 엔트리 (extractive/LLM)
│  ├─ generator.py           # LLM 기반 생성기
│  └─ cli_local.py           # local CLI 보조 엔트리
├─ data/
│  ├─ html/                  # 원본 감사보고서 HTML
│  ├─ parsed/                # 파싱 결과 JSON
│  └─ models/                # 모델 관련 캐시/저장소 용도
├─ sql/
│  ├─ init.sql               # DB 스키마 초기화
│  └─ migrations/
│     └─ 001_add_pipeline_id.sql
├─ eval_structured_risk_model.py   # structured risk 모델 평가
├─ eval_dev_model.py               # 다른 dev 프로젝트 평가용 스크립트
├─ common_eval_utils.py            # 평가 공통 유틸
├─ contingent_question_set.json    # 평가용 질문셋
├─ risk_qna_2014_2024.csv          # 질문/답 저장 파일
├─ pyproject.toml                  # Poetry 의존성 관리
├─ docker-compose.yml              # PostgreSQL + pgvector 실행
├─ .env.example                    # 환경변수 예시
├─ README.md
└─ README_KO.md
```

---

## 5. 처음 보시는 분들이 꼭 알아야 할 파일 6개

아래 6개만 보시면 됩니다.

### 1) `app/parser.py`
가장 먼저 원본 HTML을 읽어 **구조화된 보고서 객체**로 바꿉니다.

이 파일이 하는 일:
- EUC-KR / CP949 / UTF-8 디코딩 시도
- BeautifulSoup 파서 fallback (`lxml -> html5lib -> html.parser`)
- 섹션 / 주석 / 표를 감지
- 보고서 연도 추출
- 구조화 JSON 생성


### 2) `app/chunker.py`
파싱된 결과를 검색 가능한 단위로 나눕니다.

이 파일이 하는 일:
- 설명형 텍스트를 청크로 분리
- note 단위 / subsection 단위 / table 관련 chunk 생성
- 구조화 표(`structured_tables`)와 셀(`structured_table_cells`) 행 생성

**검색 단위**와 **정밀 추출 단위**를 만드는 단계입니다.

### 3) `app/ingest.py`
실제 적재 파이프라인의 중심입니다.

이 파일이 하는 일:
- HTML 파일 순회
- `parse_html_file()` 호출
- `build_chunks()` 호출
- `build_structured_tables()` 호출
- 임베딩 생성
- DB에 `documents`, `chunks`, `structured_tables`, `structured_table_cells` 저장

이 파일이 돌아가면 “질문 가능한 상태”가 됩니다.

### 4) `app/search_local.py`
숫자 질문에 강한 검색기입니다.

예:
- “2020년 지급보증한 내역에서 관련 차입금 총액은 얼마인가?”
- “2018년 충당부채 기말잔액은 얼마인가?”

이 파일이 하는 일:
- 질문에서 연도 추출
- 질문 정규화
- 리스크/주석 힌트 추출
- coarse/fine retrieval 수행
- 구조화 표/셀 정보와 연결
- 정밀한 amount/evidence 추출 지원

### 5) `app/qa_local.py`
`search_local.py`의 결과를 사람이 읽기 쉬운 발표용 답변으로 바꿉니다.

출력 예시 형태:

```text
질문: 2020년 지급보증한 내역에서 관련 차입금 총액은 얼마인가?
답변: 312,414
주석: 우발부채와 약정사항
세부항목: 지급보증한 내역
근거: ...
판단근거: ...
```

### 6) `app/qa.py`
개발용 QA 엔트리입니다.

이 파일은
- 추출형 QA를 시험하거나
- LLM 기반 생성형 답변을 시험하거나
- 개발 중 디버깅용으로 검색/생성 결과를 확인할 때 사용합니다.

`qa_local.py`가 발표용 진입점이라면, `qa.py`는 실험용 진입점에 가깝습니다.

---

## 6. DB에는 무엇이 저장되나?

이 프로젝트는 PostgreSQL + pgvector를 사용합니다.

### 1) `documents`
문서 자체를 저장합니다.

주요 내용:
- 파일명
- 제목
- 보고서 연도
- 원본 HTML
- 파싱된 JSON

 “원문과 구조화 결과의 원본 저장소”입니다.

### 2) `chunks`
검색용 텍스트 청크를 저장합니다.

주요 내용:
- 청크 키
- 소속 연도
- note 제목 / subtopic
- section type
- risk domain
- 본문 content
- embedding vector

**벡터 검색의 주된 대상**입니다.

### 3) `structured_tables`
표 자체의 메타데이터를 저장합니다.

주요 내용:
- table_id
- note_title
- subtopic
- table_title
- unit
- risk_domain

“이 표가 어떤 표인지”를 설명하는 테이블입니다.

### 4) `structured_table_cells`
표의 셀 단위 값을 저장합니다.

주요 내용:
- row_label / col_label
- value_raw / value_numeric
- unit
- row_role / period_role
- row_year
- entity_label

**숫자 질의를 정확히 풀기 위한 핵심 테이블**입니다.

---

## 7. Local 검색 방식이 중요한 이유

이 프로젝트를 처음 볼 때 가장 중요한 포인트는, **감사보고서 질문의 상당수가 특정 숫자와 항목을 정확히 찾는 문제**라는 점입니다.

예를 들어 아래 같은 질문은 단순히 유사한 문장을 찾는 것만으로는 답하기 어렵습니다.

- “2020년 지급보증한 내역에서 관련 차입금 총액은 얼마인가?”
- “2018년 판매보증충당부채 기말잔액은 얼마인가?”
- “2021년 우발부채와 약정사항 주석에서 보증 한도는 얼마인가?”

이런 질문에 답하려면:
1. 질문 속 **연도**를 찾아야 하고
2. 질문이 가리키는 **주석 제목**을 찾아야 하고
3. 표 안의 **행/열 의미**를 해석해야 하며
4. 최종적으로 **셀 값**을 정확히 뽑아야 합니다.

그래서 이 프로젝트는 텍스트 청크 검색만 하지 않고, **구조화된 표와 셀 정보**를 함께 저장하고 활용합니다.

---

## 8. 설치 방법

### 1) Poetry 설치 및 패키지 설치

```bash
poetry install
cp .env.example .env
```

### 2) DB 실행

```bash
docker compose up -d
```

### 3) 원본 HTML 배치

감사보고서 `.htm` 또는 `.html` 파일을 `data/html/`에 넣습니다.

예:

```text
data/html/2014.htm
data/html/2015.htm
...
data/html/2024.htm
```

---

## 9. 환경변수

`.env.example` 기준 주요 설정은 아래와 같습니다.

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=samsung_audit
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

HTML_DIR=./data/html
PARSED_DIR=./data/parsed
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIM=384
NOTE_CHUNK_SIZE=1100
NOTE_CHUNK_OVERLAP=150
TEXT_CHUNK_SIZE=1000
TEXT_CHUNK_OVERLAP=120
TABLE_ROW_GROUP_SIZE=12
TOP_K=5
```

중요한 값:
- `HTML_DIR`: 원본 HTML 위치
- `PARSED_DIR`: 파싱 JSON 저장 위치
- `EMBEDDING_MODEL`: 텍스트 임베딩 모델
- `TOP_K`: 검색 시 기본 상위 개수

---

## 10. 데이터 적재 방법

모든 준비가 끝났으면 아래를 실행합니다.

```bash
poetry run python -m app.ingest
```

이 명령이 하는 일:
1. `data/html/`의 모든 HTML 파일 순회
2. 각 파일 파싱
3. 청크 및 표 구조 생성
4. 임베딩 생성
5. DB 저장
6. `data/parsed/*.json` 저장

실행 후 기대 결과:
- `documents` 채워짐
- `chunks` 채워짐
- `structured_tables` 채워짐
- `structured_table_cells` 채워짐
- `data/parsed/`에 파싱 결과 JSON 생성

---

## 11. 실행 예시

### A. Local QA
숫자/금액 질문에 적합합니다.

```bash
poetry run python -m app.qa_local --llm "2014년 비용 중 당기 원재료 등의 사용액 및 상품 매입액 등의 금액은?"
```

---

## 12. 질문 유형별 추천 엔트리

처음 쓰는 사람은 아래처럼 생각하면 됩니다.

| 질문 유형 | 추천 엔트리 | 이유 |
|---|---|---|
| 특정 연도의 특정 금액 | `app.qa_local` | 표/셀 기반 정밀 추출에 강함 |
| 특정 항목의 잔액/총액/내역 | `app.qa_local` | 숫자형 질의에 강함 |
| 주석 중심의 근거 확인 | `app.qa_local` 또는 `app.qa` | 정밀 검색 또는 개발용 확인 가능 |
| 실험/디버깅/LLM 테스트 | `app.qa` | 개발용 옵션이 많음 |

---

## 13. 코드 읽는 추천 순서

처음 보는 사람에게는 아래 순서를 권합니다.

### 1단계. 전체 흐름 이해
- `README.md`
- `app/ingest.py`

먼저 전체 파이프라인이 어떻게 흘러가는지 봅니다.

### 2단계. 문서가 어떻게 구조화되는지 이해
- `app/parser.py`
- `app/chunker.py`

원본 HTML이 어떤 구조 데이터로 바뀌는지 봅니다.

### 3단계. 질문에 어떻게 답하는지 이해
- `app/search_local.py`
- `app/qa_local.py`
- `app/qa.py`

검색기와 답변기가 어떻게 연결되는지 봅니다.

### 4단계. DB 구조 확인
- `sql/init.sql`
- `sql/migrations/001_add_pipeline_id.sql`

데이터가 어디에 어떤 형태로 저장되는지 확인합니다.

### 5단계. 평가 방식 이해
- `eval_structured_risk_model.py`
- `common_eval_utils.py`
- `contingent_question_set.json`

이 프로젝트를 어떻게 평가하는지 봅니다.

---

## 14. 평가 파일은 무엇을 위한 것인가?

프로젝트에는 평가용 파일도 포함되어 있습니다.

### `contingent_question_set.json`
우발부채/약정사항 관련 질문셋입니다.

### `eval_structured_risk_model.py`
현재 structured risk 모델을 평가하는 스크립트입니다.

### `common_eval_utils.py`
정답 비교, 지표 계산 등 공통 평가 로직을 담고 있습니다.

이 프로젝트는 “검색/질의응답 시스템”일 뿐 아니라, **질문셋으로 성능을 측정할 수 있는 실험 프로젝트**이기도 합니다.

---

## 15. 자주 헷갈리는 포인트

### Q1. `search.py`와 `search_local.py`는 뭐가 다른가?
`search.py`는 사실상 `search_local.py`를 다시 내보내는 얇은 래퍼입니다.

핵심 로직은 `search_local.py`에 있습니다.

### Q2. `qa.py`와 `qa_local.py`는 뭐가 다른가?
- `qa_local.py`: 발표용/고정 포맷 응답
- `qa.py`: 개발용 RAG QA 엔트리, extractive 또는 LLM 사용 가능

### Q3. 왜 표를 따로 구조화하나?
감사보고서 질문 중 상당수는 텍스트가 아니라 **표의 특정 셀 숫자**를 정확히 찾아야 하기 때문입니다.

### Q4. 왜 셀 단위까지 저장하나?
행/열 레이블과 숫자 값을 함께 저장해야 “총액”, “기말잔액”, “관련 차입금” 같은 세부 질문에 정확히 대응할 수 있기 때문입니다.

---

## 16. 이 프로젝트의 강점

- 감사보고서 HTML을 직접 구조화함
- 텍스트 청크와 표 셀을 함께 다룸
- 숫자 질문에 강한 정밀 검색 구조를 가짐
- 평가용 질문셋과 평가 스크립트가 같이 있음

---

## 17. 현재 한계

- 문서 전체를 넓게 종합하는 질문은 추가적인 근거 결합이 더 필요할 수 있음
- exact number 추출에는 강하지만, 매우 추상적인 서술형 질문에는 한계가 있을 수 있음
- LLM 생성 품질은 별도 모델/환경에 영향을 받음

즉, 이 프로젝트는
**“감사보고서 질문을 전부 완벽히 해결하는 시스템”** 이라기보다,
**“리스크/주석/표 중심 질문에 강한 구조화 RAG 시스템”** 으로 보는 것이 정확합니다.

---

## 18. 빠르게 써보고 싶은 사람용 요약

정말 빠르게 시작하려면 아래 순서만 따라가면 됩니다.

```bash
poetry install
cp .env.example .env
docker compose up -d
# data/html/ 에 감사보고서 html 넣기
poetry run python -m app.ingest
poetry run python -m app.qa_local --llm "2014년 현금및현금성자산 중 당기말 예금 등의 금액은"
```

---

## 19. 한 문장으로 다시 정리

이 프로젝트는

> **삼성전자 감사보고서를 HTML 단계에서 구조화하고, 텍스트 검색과 표 셀 조회를 결합해서, 질문에 맞는 근거를 찾아 답변하는 리스크 중심 RAG 시스템**

입니다.
