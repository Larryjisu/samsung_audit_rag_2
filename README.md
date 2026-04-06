# samsung-audit-rag-structured

삼성전자 감사보고서 `.htm/.html` 파일을 대상으로, **통합형 구조(PostgreSQL + pgvector)** 로

1. EUC-KR/CP949 HTML 디코딩
2. 방어적 HTML 파싱 (`lxml -> html5lib -> html.parser` fallback)
3. 구조화 JSON 생성
4. 섹션별 청킹
5. 임베딩 생성
6. PostgreSQL + pgvector 적재

까지 수행하는 최소 프로젝트입니다.

## 왜 통합형인가?

이번 데이터는 **삼성전자 감사보고서 10년치 수준**이라서, 별도 벡터 DB를 두는 분리형보다
**PostgreSQL 하나에 원문 메타데이터 + 임베딩을 같이 저장**하는 방식이 더 단순하고 관리하기 쉽습니다.

- `documents`: 원본 파일과 구조화 JSON 저장
- `chunks`: 검색 단위 text + 메타데이터 + embedding 저장

## 파싱 구조

이 프로젝트는 감사보고서 구조를 아래처럼 봅니다.

- `h2.SECTION-1`
  - 독립된 감사인의 감사보고서
  - (첨부)재무제표
- `(첨부)재무제표` 아래 table 제목 패턴
  - 재무상태표
  - 손익계산서
  - 포괄손익계산서
  - 자본변동표
  - 현금흐름표
- `h3.SECTION-2`
  - 주석
- 주석 아래 번호형 소제목
  - 1. 일반적 사항
  - 2. 중요한 회계처리방침
  - 3. 중요한 회계추정 및 가정
  - ...

## 폴더 구조

```bash
samsung-audit-rag-structured/
├─ app/
│  ├─ config.py
│  ├─ db.py
│  ├─ parser.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ ingest.py
│  └─ search.py
├─ data/
│  ├─ html/
│  └─ parsed/
├─ sql/
│  └─ init.sql
├─ docker-compose.yml
├─ pyproject.toml
└─ .env.example
```

## 1. Poetry 설치

```bash
poetry install
cp .env.example .env
```

## 2. PostgreSQL 실행

```bash
docker compose up -d
```

## 3. 감사보고서 파일 넣기

`data/html/` 폴더에 10년치 `.htm` 파일을 넣습니다.

예:

```bash
data/html/감사보고서_2014.htm
data/html/감사보고서_2015.htm
...
```

## 4. 적재 실행

```bash
poetry run python -m app.ingest
```

실행하면:

- `documents.parsed_json` 에 구조화 JSON 저장
- `data/parsed/*.json` 에 파싱 결과 파일 저장
- `chunks` 테이블에 청크 + 임베딩 저장

## 5. 검색 테스트

```bash
poetry run python -m app.search "현금및현금성자산"
```

## 6. RAG 프롬프트 / QA 테스트
# samsung-audit-rag-structured-risk

삼성전자 감사보고서(2014~2024) 기반 RAG 프로젝트입니다.

현재 구조는 **공통 적재 + 분리된 검색/QA**로 운영됩니다.

- Local 모델: 특정 연도/항목/셀 중심 정밀 추출
- Hybrid 모델: 다중 연도/다중 주석/다중 근거 결합 설명
- Auto 라우터: 질문 유형에 따라 Local/Hybrid 자동 선택

---

## 1) 프로젝트 구조

```text
samsung_audit_rag_structured_risk/
├─ app/
│  ├─ config.py
│  ├─ db.py
│  ├─ parser.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ ingest.py
│  ├─ search_local.py
│  ├─ search_hybrid.py
│  ├─ qa_local.py
│  ├─ qa_hybrid.py
│  ├─ qa_auto.py
│  ├─ cli_local.py
│  └─ cli_hybrid.py
├─ data/
│  ├─ html/
│  └─ parsed/
├─ sql/
│  ├─ init.sql
│  └─ migrations/
│     └─ 001_add_pipeline_id.sql
├─ pyproject.toml
├─ poetry.lock
├─ .env.example
└─ README.md
```

---

## 2) 모델별 역할

### Local

- 핵심 파일: `app/search_local.py`, `app/qa_local.py`
- 강점: 숫자형/항목형 질문(예: 금액, 총액, 잔액, 특정 연도)
- 출력: 발표용 고정 텍스트 포맷

### Hybrid

- 핵심 파일: `app/search_hybrid.py`, `app/qa_hybrid.py`
- 강점: 리스크/흐름/비교/설명형 질문
- 특징: coarse + fine 결합, year/note diversity 보정

### Auto Router

- 파일: `app/qa_auto.py`
- 라우팅 규칙(요약):
  1. 기간 흐름형 → Hybrid
  2. 강한 숫자형 → Local
  3. 리스크형 → Hybrid
  4. 설명/비교형 → Hybrid
  5. fallback → Hybrid

---

## 3) 설치 및 초기 설정

```bash
poetry install
cp .env.example .env
```

DB 실행 (필요 시):

```bash
docker compose up -d
```

---

## 4) 데이터 적재

`data/html/`에 감사보고서 `.htm/.html` 파일을 넣은 뒤:

```bash
poetry run python -m app.ingest
```

적재 결과:

- `documents`: 원문/메타/구조화 JSON
- `chunks`: 검색 청크 + 임베딩
- `structured_tables`, `structured_table_cells`: 표 구조화 데이터

---

## 5) 실행 방법

### Local QA (발표용 숫자형)

```bash
poetry run python -m app.qa_local "2020년 지급보증한 내역에서 관련 차입금 총액은 얼마인가?"
```

### Hybrid QA (발표용 설명/흐름형)

```bash
poetry run python -m app.qa_hybrid "2014~2024년 우발부채/약정 관련 흐름을 설명해줘"
```

### Auto QA (권장 발표 엔트리)

```bash
poetry run python -m app.qa_auto "담보/보증 관련 리스크는 어느 주석에 집중되어 있는가?"
```

### 개발용 QA (`app.qa`)

```bash
# 기본: extractive(LLM 미사용)
poetry run python -m app.qa "질문"

# LLM 사용
poetry run python -m app.qa --llm "질문"

# 디버그 출력
poetry run python -m app.qa --debug "질문"
```

> 참고: `--llm`은 모델 다운로드/로딩이 필요해 초기 실행이 오래 걸릴 수 있습니다.

---

## 6) 자주 쓰는 발표용 질문

- `2020년 지급보증한 내역에서 관련 차입금 총액은 얼마인가?`
- `2018년 현금및현금성자산 중 현금 금액은 얼마인가?`
- `소송 관련 리스크가 언급된 주석을 찾아줘`
- `2014~2024년 우발부채/약정 관련 흐름을 설명해줘`

---

## 7) 트러블슈팅

1. `dquote>`가 뜨는 경우
   - 스마트 따옴표(`“ ”`) 대신 일반 따옴표(`" "`) 사용

2. Python 3.13 경고
   - 프로젝트는 `>=3.11,<3.13` 범위 사용
   - Poetry가 자동으로 3.12를 선택하면 정상

3. `--llm`이 느린 경우
   - 모델 다운로드 중일 수 있음
   - 발표 직전에는 `qa_local`/`qa_hybrid`/`qa_auto` 경량 모드 권장

---

## 8) 공유용 파일 목록

공유 대상 파일은 아래 매니페스트를 참고하세요.

- `REQUIRED_FILES_MANIFEST.txt`
