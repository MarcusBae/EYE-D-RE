-- ivfflat 근사검색 recall 향상: 기본 probes=1 -> 10 (lists 권장값 sqrt(lists))
-- 배경: reset_for_test.sql의 IVFFlat 인덱스가 "low recall" 경고를 냈고, 실측 확인됨.
--   probes=1일 때 /match·/ingest 최근접 검색이 진짜 1순위를 놓침
--   (sim=None 누락 / 더 낮은 sim의 엉뚱한 global_id 반환 = 온라인 over-split 유발).
-- 적용: 신규 DB 구축 후 1회 실행. 비파괴·가역(RESET 가능). 공유 DB라 팀 공유 필요.
-- 주의: IVFFlat<->HNSW 구조 전환은 별도 팀 결정. 본 변경은 그 전 단계의 안전한 튜닝.
ALTER DATABASE eyed SET ivfflat.probes = 10;
