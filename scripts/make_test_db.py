#!/usr/bin/env python3
"""실 eyed 스키마만 복제해 eyed_test 테스트 DB 생성(데이터 미포함). commit=true 검증용.
사용: python scripts/make_test_db.py   (접속정보는 .env 의 EYED_DB_DSN 사용)"""
import os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2
from server.db import get_conn  # .env load_dotenv 수행

CONTAINER = os.environ.get("EYED_PG_CONTAINER", "eyed-postgres")

def main():
    base = os.environ.get("EYED_DB_DSN")
    if not base: sys.exit("[중단] EYED_DB_DSN(.env) 필요")
    pw = (re.search(r"password=(\S+)", base) or [None, ""])[1] if re.search(r"password=(\S+)", base) else ""
    test_dsn = re.sub(r"dbname=\S+", "dbname=eyed_test", base)

    conn = get_conn(); conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='eyed_test' AND pid<>pg_backend_pid()")
        cur.execute("DROP DATABASE IF EXISTS eyed_test"); cur.execute("CREATE DATABASE eyed_test")
    conn.close(); print("DROP+CREATE eyed_test 완료")

    subprocess.run(["docker","exec","-e",f"PGPASSWORD={pw}",CONTAINER,"bash","-c",
        "pg_dump -U eyed -d eyed --schema-only --no-owner --no-privileges | psql -U eyed -d eyed_test -v ON_ERROR_STOP=1 -q"],
        check=True); print("스키마 복제 OK")

    c2 = psycopg2.connect(test_dsn); c2.autocommit = True
    with c2.cursor() as cur: cur.execute("ALTER DATABASE eyed_test SET ivfflat.probes = 10")
    c2.close(); print("probes=10 OK")
    print("\n준비 완료. 테스트DB로 서버 기동:\n  EYED_DB_DSN='...dbname=eyed_test...' uvicorn server.main:app --port 8000")

if __name__ == "__main__":
    main()
