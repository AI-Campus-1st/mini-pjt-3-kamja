"""제출 전 구조 검사: 계층 분리 · 같은 체인 · 개인정보 · API 키 (Day 6 오전)
실행: 프로젝트 루트에서  HF_HUB_OFFLINE=1 python -m eval.check_layers [--live]
- 원자료 eval/results/check_layers_detail.csv 를 먼저 저장하고, 요약 report/evidence/check_layers.md 를 씀
- --live: 질문 1개를 chain.answer로 실제 실행해 인덱스 건수가 그대로인지 확인 (OpenAI 호출 1회, 약 $0.01)
- 키·학번은 앞 4자만 보이고 나머지는 가림"""
import argparse, ast, os, re, sqlite3, subprocess, sys, unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules", "chroma"}
TEXT_EXT = {".py", ".ipynb", ".md", ".csv", ".json", ".jsonl", ".txt", ".toml", ".log", ".yaml", ".yml", ".cfg"}
rows = []


def add(group, item, verdict, file="", line="", detail=""):
    rows.append({"구분": group, "항목": item, "판정": verdict, "파일": file, "줄": line, "내용": detail})


def nfc(p):
    return unicodedata.normalize("NFC", str(p))


def rel(p):
    return nfc(Path(p).relative_to(ROOT))


def walk(base=ROOT):
    for d, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            yield Path(d) / f


def mask(s):
    return s[:4] + "*" * 6


def py_files(folder):
    return sorted((ROOT / folder).glob("*.py"))


# ---------- A. 계층 분리 ----------
CRAWL = {"collector", "indexer", "googleapiclient", "requests", "bs4", "selenium", "urllib.request"}
WRITE = [(re.compile(r"(_col\(\)|\bcol|collection)\.(add|upsert|delete|update|modify)\("), "인덱스 쓰기"),
         (re.compile(r"(create_collection|get_or_create_collection|delete_collection)\("), "컬렉션 만들기·지우기"),
         (re.compile(r"embed_documents\("), "문서 임베딩"),
         (re.compile(r"\.(to_csv|to_json|to_parquet|write_text|write_bytes)\("), "파일 쓰기"),
         (re.compile(r"open\([^)]*['\"](w|a|wb|ab)['\"]"), "파일 쓰기")]
EMBED = [(re.compile(r"SentenceTransformer\("), "임베딩 모델 생성"), (re.compile(r"\.encode\("), "임베딩 계산")]


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(a.name, n.lineno, []) for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.append((n.module, n.lineno, [a.name for a in n.names]))
    return out


def check_static():
    for folder in ["app", "rag"]:
        for f in py_files(folder):
            for mod, ln, _ in imports(f):
                if any(mod == c or mod.startswith(c + ".") for c in CRAWL):
                    add("A 계층", f"{folder}/ 수집·인덱서 import 없음", "실패", rel(f), ln, mod)
            for i, t in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                code = t.split("#")[0]
                for pat, name in WRITE:
                    if pat.search(code):
                        add("A 계층", f"{folder}/ 쓰기 없음", "실패", rel(f), i, f"{name}: {t.strip()[:80]}")
                if "sqlite3.connect(" in code and "mode=ro" not in code:
                    add("A 계층", f"{folder}/ DB는 읽기 전용", "실패", rel(f), i, t.strip()[:80])
                for pat, name in EMBED:
                    if pat.search(code):
                        ok = folder == "rag" and f.name == "retriever.py"
                        add("A 계층", "임베딩 위치", "허용(질문 임베딩)" if ok else "실패", rel(f), i, f"{name}: {t.strip()[:80]}")
    if not any(r["판정"] == "실패" and r["구분"] == "A 계층" for r in rows):
        add("A 계층", "app/·rag/ 정적 검사", "통과", "app/*.py, rag/*.py", "", "수집·인덱서 import, 인덱스·파일 쓰기, 임베딩 생성 없음")


def check_graph():
    """app/*.py에서 출발해 프로젝트 안 import를 끝까지 따라감 → collector/indexer에 닿으면 실패"""
    seen, todo, reached = set(), [f"app.{f.stem}" for f in py_files("app")], set()
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        p = ROOT / (m.replace(".", "/") + ".py")
        if not p.exists():
            continue
        for mod, _, _ in imports(p):
            top = mod.split(".")[0]
            if top in {"collector", "indexer"}:
                reached.add(f"{m} → {mod}")
            if top in {"app", "rag", "store"}:
                todo.append(mod)
    add("A 계층", "app/ import 경로에 collector·indexer 없음", "실패" if reached else "통과",
        "", "", "; ".join(sorted(reached)) or f"따라간 모듈 {len(seen)}개: " + ", ".join(sorted(seen)))


BLOCK_CODE = r'''
import importlib, importlib.abc, sys
sys.path.insert(0, {root!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("indexer", "collector"):
            raise ImportError("BLOCKED " + name)
sys.meta_path.insert(0, Block())
for m in {mods!r}:
    try:
        importlib.import_module(m); print("OK", m)
    except Exception as e:
        print("BLOCKED" if "BLOCKED" in str(e) else "ERR", m, type(e).__name__, str(e)[:120])
'''


def check_no_indexer():
    """indexer·collector를 막은 채로 챗봇 모듈을 불러옴 (가이드: 인덱서를 제외해도 챗봇 구동)"""
    mods = [f"app.{f.stem}" for f in py_files("app") if f.stem != "main"] + ["rag.chain"]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-c", BLOCK_CODE.format(root=str(ROOT), mods=mods)],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    for t in r.stdout.splitlines():
        kind, mod, *rest = t.split(" ", 2)
        verdict = {"OK": "통과", "BLOCKED": "실패"}.get(kind, "확인 필요")
        add("A 계층", "indexer·collector 없이 불러오기", verdict, mod, "", " ".join(rest) or "불러오기 성공")
    if r.returncode or not r.stdout:
        add("A 계층", "indexer·collector 없이 불러오기", "확인 필요", "", "", r.stderr.strip()[-200:])


def check_store():
    for p in [ROOT / "store" / "chroma", ROOT / "store" / "docs.db"]:
        add("A 계층", "벡터 DB·원문 DB 디스크 저장", "통과" if p.exists() else "실패", rel(p) if p.exists() else nfc(p), "",
            "있음" if p.exists() else "없음")


def snapshot():
    snap = {}
    try:
        from rag.retriever import _col
        snap["chroma"] = _col().count()
    except Exception as e:
        snap["chroma"] = f"오류 {type(e).__name__}"
    db = ROOT / "store" / "docs.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for (t,) in con.execute("select name from sqlite_master where type='table'"):
            snap[f"docs.db:{t}"] = con.execute(f'select count(*) from "{t}"').fetchone()[0]
        con.close()
    return snap


def check_live():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(ROOT))
    before = snapshot()
    try:
        from rag.chain import answer
        answer("AI로 만든 모델이 나오는 화장품 광고를 하면 소비자 반응이 어떨까?")
        ran = "실행 성공"
    except Exception as e:
        ran = f"실행 오류 {type(e).__name__}: {str(e)[:100]}"
    after = snapshot()
    diff = {k: (before.get(k), after.get(k)) for k in set(before) | set(after) if before.get(k) != after.get(k)}
    add("A 계층", "답변 1회 전후 인덱스 건수 같음", "실패" if diff else "통과", "rag/chain.py answer()", "",
        f"{ran}; " + (f"바뀜 {diff}" if diff else f"그대로 {before}"))


# ---------- B. 같은 체인 ----------
def check_chain():
    users = {"app": [], "eval": []}
    for folder in users:
        for f in py_files(folder):
            for mod, ln, names in imports(f):
                if mod == "rag.chain":
                    users[folder].append(f"{f.name}: {', '.join(names)}")
                    add("B 체인", f"{folder}/ → rag/chain.py", "정보", rel(f), ln, ", ".join(names))
                elif mod == "rag.retriever" and folder == "eval" and "search" in names:
                    add("B 체인", "체인 밖 검색만 재는 스크립트", "정보(검색 지표용)", rel(f), ln, "rag.retriever.search")
    src = (ROOT / "rag" / "chain.py").read_text(encoding="utf-8")
    same = "RunnableLambda(prepare)" in src and "def stream_body" in src
    ok = users["app"] and users["eval"] and same
    add("B 체인", "챗봇과 평가 하네스가 같은 chain.py", "통과" if ok else "실패", "rag/chain.py", "",
        f"app {len(users['app'])}개, eval {len(users['eval'])}개가 import. answer() = prepare → _body, "
        f"화면 = prepare → stream_body(마지막에 answer()와 같은 본문으로 덮어씀)" if ok else f"app {users['app']}, eval {users['eval']}, prepare 공유 {same}")


# ---------- C. 개인정보 ----------
AUTHOR = re.compile(r"author(?!ity)|channel_?id|profile", re.I)   # authority(문서 종류 메타)는 제외
SID = re.compile(r"(?<!\d)(19|20)\d{6}(?!\d)")   # 학번처럼 보이는 8자리 숫자


def check_privacy():
    bad = 0
    for p in walk():
        if p.suffix in {".csv", ".jsonl", ".json"} and any(x in p.parts for x in ("data", "store", "eval")):
            try:
                cols = (pd.read_csv(p, nrows=0, encoding="utf-8-sig").columns if p.suffix == ".csv"
                        else pd.read_json(p, lines=p.suffix == ".jsonl", nrows=1 if p.suffix == ".jsonl" else None).columns)
            except Exception:
                continue
            hit = [c for c in map(str, cols) if AUTHOR.search(c)]
            if hit:
                bad += 1
                add("C 개인정보", "작성자 정보 열 없음", "실패", rel(p), "", ", ".join(hit))
        if SID.search(nfc(p.name)):
            bad += 1
            add("C 개인정보", "파일 이름에 학번·이름 없음", "확인 필요", rel(p), "", mask(nfc(p.name)) + " (제출 저장소에서 빼거나 이름 바꾸기)")
    db = ROOT / "store" / "docs.db"
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for (t,) in con.execute("select name from sqlite_master where type='table'"):
            hit = [c[1] for c in con.execute(f'pragma table_info("{t}")') if AUTHOR.search(c[1])]
            if hit:
                bad += 1
                add("C 개인정보", "작성자 정보 열 없음", "실패", f"store/docs.db:{t}", "", ", ".join(hit))
        con.close()
    try:
        sys.path.insert(0, str(ROOT))
        from rag.retriever import _col
        keys = set().union(*[set(m) for m in _col().get(include=["metadatas"])["metadatas"] if m])
        hit = sorted(k for k in keys if AUTHOR.search(k))
        add("C 개인정보", "벡터 DB 메타에 작성자 정보 없음", "실패" if hit else "통과", "store/chroma", "",
            ", ".join(hit) or f"메타 키 {len(keys)}개 확인")
        bad += bool(hit)
    except Exception as e:
        add("C 개인정보", "벡터 DB 메타에 작성자 정보 없음", "확인 필요", "store/chroma", "", f"{type(e).__name__}: {str(e)[:80]}")
    if not bad:
        add("C 개인정보", "data·store·eval 파일 검사", "통과", "", "", "작성자 이름·채널 ID·프로필 열, 학번 파일 이름 없음")


# ---------- D. API 키 ----------
KEYS = [(re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "OpenAI 키 모양"),
        (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "Google API 키 모양"),
        (re.compile(r"(OPENAI|YOUTUBE)_API_KEY\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{12,})"), "키 값 직접 입력")]


def check_keys():
    gi = ROOT / ".gitignore"
    listed = gi.exists() and any(l.strip() in {".env", "/.env", "*.env", ".env*"} for l in gi.read_text().splitlines())
    add("D API 키", ".env가 .gitignore에 있음", "통과" if listed else "실패", ".gitignore", "", "있음" if listed else "없음")
    try:
        r = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True, cwd=ROOT)
        if r.returncode:   # 아직 git 저장소가 아님 → 통과로 세지 않음
            add("D API 키", ".env가 git에 올라가지 않음", "확인 필요", ".env", "", "git 저장소 아님: git init 뒤 다시 실행")
        else:
            tracked = r.stdout.strip()
            add("D API 키", ".env가 git에 올라가지 않음", "실패" if tracked else "통과", ".env", "", tracked or "추적 안 됨")
    except FileNotFoundError:
        add("D API 키", ".env가 git에 올라가지 않음", "확인 필요", "", "", "git 없음")
    n = 0
    for p in walk():
        if p.suffix not in TEXT_EXT or p.name == ".env" or p.stat().st_size > 50_000_000:
            continue
        for i, t in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for pat, name in KEYS:
                m = pat.search(t)
                if m:
                    n += 1
                    add("D API 키", "코드·결과 파일에 키 없음", "실패", rel(p), i, f"{name}: {mask(m.group(m.lastindex or 0))}")
    if not n:
        add("D API 키", "코드·결과 파일에 키 없음", "통과", "", "", "OpenAI·Google 키 모양, 키 값 직접 입력 없음 (.env 제외)")


def md_table(df):
    """tabulate 없이 마크다운 표 만들기"""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in df.astype(str).values]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="answer() 1회 실행 전후 인덱스 건수 비교 (OpenAI 호출)")
    a = ap.parse_args()
    check_static(); check_graph(); check_no_indexer(); check_store()
    if a.live:
        check_live()
    check_chain(); check_privacy(); check_keys()
    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "check_layers_detail.csv", index=False, encoding="utf-8-sig")       # 원자료 먼저
    summ = df.groupby(["구분", "판정"]).size().unstack(fill_value=0)
    lines = ["# 구조 검사 결과 (`python -m eval.check_layers" + (" --live" if a.live else "") + "`)", "",
             "- 원자료: `eval/results/check_layers_detail.csv`", "", "## 판정 집계", "", md_table(summ.reset_index()), "",
             "## 통과가 아닌 항목", ""]
    rest = df[df["판정"].isin(["실패", "확인 필요"])]
    lines += [md_table(rest) if len(rest) else "없음", "", "## 전체", "", md_table(df)]
    (ROOT / "report" / "evidence").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "evidence" / "check_layers.md").write_text("\n".join(lines), encoding="utf-8")
    print(summ.to_string(), "\n")
    print(rest.to_string(index=False) if len(rest) else "통과가 아닌 항목 없음")


if __name__ == "__main__":
    main()