# make_public v2 (2026-11-10): 노트북 셀 코드·사례 요약 CSV의 댓글 인용까지 가림
"""공개 저장소용으로 유튜브 댓글 원문을 가림. 원본은 private/에 같은 경로로 보관 (private/는 .gitignore)
- 검수 CSV: text·parent_text 열을 비움 표시로 / 답변 JSON·md: 실제 댓글과 같은 인용만 가림 / API 시험 노트북: 출력만 지움
- 다시 실행해도 private/의 원본에서 새로 만듦. 마지막에 커밋될 파일 전체에서 남은 댓글 조각을 셈 (0이어야 함)
실행 (프로젝트 루트, .gitignore에 private/ 추가 후): python make_public.py
"""
import csv, io, json, re, shutil, sqlite3, subprocess
from pathlib import Path

MASK = "(댓글 원문 비공개)"
norm = lambda s: re.sub(r"\s+", " ", s or "").strip()
con = sqlite3.connect("file:store/docs.db?mode=ro", uri=True)
coms = [norm(re.sub(r"^(@사용자\s*)+", "", t or "")) for (t,) in con.execute("select text from youtube_comments")]
coms = [c for c in coms if len(c) >= 12]
BIG = "\x00".join(coms)
PRE = {}
for c in coms:
    if len(c) >= 20: PRE.setdefault(c[:20], []).append(c)
QUOTE = re.compile(r'(["“‘\'])([^"”’\'\n]{12,300})(["”’\'])')

def is_comment(q):
    q = norm(q).rstrip("…. ")
    return len(q) >= 12 and q in BIG

def mask_str(s):
    s = QUOTE.sub(lambda m: m.group(1) + MASK + m.group(3) if is_comment(m.group(2)) else m.group(0), s)
    flat = norm(s)
    for i in range(max(len(flat) - 19, 0)):                     # 따옴표 없이 들어간 댓글
        for c in PRE.get(flat[i:i + 20], []):
            for L in range(min(len(c), 120), 19, -1):
                if c[:L] in s: s = s.replace(c[:L], MASK); break
    return s

def walk(x, key=None):
    if isinstance(x, dict): return {k: walk(v, k) for k, v in x.items()}
    if isinstance(x, list): return [walk(v, key) for v in x]
    if isinstance(x, str): return MASK if key == "quote" and is_comment(x) else mask_str(x)
    return x

def do_csv(s):
    rows = list(csv.reader(io.StringIO(s)))
    head = rows[0]
    hide = {i for i, h in enumerate(head) if h.lstrip("\ufeff") in ("text", "parent_text")}
    new = [head] + [[(MASK if r[i].strip() else "") if i in hide else mask_str(r[i]) for i in range(len(r))] for r in rows[1:]]
    if new == rows: return s                                   # 내용이 같으면 손대지 않음
    out = io.StringIO(); csv.writer(out, lineterminator="\n").writerows(new)
    return out.getvalue()

def do_json(s):
    obj = json.loads(s); new = walk(obj)
    return s if new == obj else json.dumps(new, ensure_ascii=False, indent=2)

def do_nb(s):
    nb = json.loads(s); before = json.dumps(nb, ensure_ascii=False)
    for c in nb["cells"]:
        if c["cell_type"] == "code": c["outputs"], c["execution_count"] = [], None
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        c["source"] = mask_str(src)                              # 셀 코드·메모에 직접 적힌 댓글도 가림
    return s if json.dumps(nb, ensure_ascii=False) == before else json.dumps(nb, ensure_ascii=False, indent=1) + "\n"

def do_csv_soft(s):                                             # 열을 통째로 지우지 않고 댓글 인용만 가림
    rows = list(csv.reader(io.StringIO(s)))
    new = [rows[0]] + [[mask_str(x) for x in r] for r in rows[1:]]
    if new == rows: return s
    out = io.StringIO(); csv.writer(out, lineterminator="\n").writerows(new)
    return out.getvalue()

TARGETS = [("eval/grading", "label_check*.csv", do_csv), ("eval/results", "**/*_miss.csv", do_csv),
           ("eval/grading", "*_grading.csv", do_csv),
           ("eval/results", "**/*.json", do_json),
           ("report/evidence", "*.md", mask_str), ("report/evidence", "case_summary.csv", do_csv_soft),
           ("notebooks", "youtube_api_test.ipynb", do_nb)]
changed = 0
for folder, pat, fn in TARGETS:
    for p in sorted(Path(folder).glob(pat)):
        orig = Path("private") / p
        src = orig.read_text(encoding="utf-8-sig" if p.suffix == ".csv" else "utf-8") if orig.exists() else \
              p.read_text(encoding="utf-8-sig" if p.suffix == ".csv" else "utf-8")
        new = fn(src)
        if new == src: continue
        if p.read_text(encoding="utf-8-sig" if p.suffix == ".csv" else "utf-8") == new: continue   # 이미 가린 상태
        if not orig.exists(): orig.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, orig)
        p.write_text(new, encoding="utf-8-sig" if p.suffix == ".csv" else "utf-8"); changed += 1
        print(f"가림: {p}  (원본 → {orig})")
print(f"\n가린 파일 {changed}개")

# 남은 댓글 조각 확인: 커밋될 파일 전체
files = [f for f in subprocess.run(["git", "-c", "core.quotepath=false", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                   capture_output=True, text=True).stdout.split("\0") if f]
left = []
for f in files:
    p = Path(f)
    if p.suffix not in {".md", ".txt", ".csv", ".json", ".jsonl", ".ipynb", ".log"}: continue
    s = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix in {".json", ".ipynb"}:
        try: s = json.dumps(json.loads(s), ensure_ascii=False)
        except Exception: pass
    s = norm(s.replace("\\n", " "))
    n = sum(1 for k in PRE if k in s)
    if n: left.append((f, n))
print(f"\n--- 커밋될 파일 {len(files)}개 중 댓글 조각(앞 20자)이 남은 파일: {len(left)}개 ---")
for f, n in left: print(f"  {f}: {n}개")
