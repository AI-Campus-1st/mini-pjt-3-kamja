"""필터 (Day 4 오후 5번, 가이드 4-7 '기간 및 출처 종류 필터', 기획서 7-2 기간·문서 종류·업종)
- 사용자가 고를 때만 조건이 생김. 아무것도 안 고르면 None → 검색·판정이 지금과 똑같음 (run1 기준선 그대로)
- where(): 검색(청크) 조건. 기간 = year 메타(indexer.add_year_meta), 문서 종류 = authority 메타, 업종 = 사례 청크만 좁힘(뉴스·정부 문서는 그대로)
- allowed_cases(): 예상·비교의 사례 판정도 같은 조건으로 좁힘 (근거 자료와 판정 대상이 어긋나지 않게)"""
from rag.reaction import cases

KINDS = ["소비자 반응", "언론 보도", "정부 기준"]
YEARS = (2023, 2026)   # 화면 슬라이더 끝값. 끝까지 두면 그쪽은 제한 없음 (2023 이전 기사도 포함)


def industries():
    return sorted({c["industry"] for c in cases().values() if c["qualified"] and c["industry"]})


def normalize(years=None, kinds=None, inds=None):
    """화면 값 → {"lo", "hi", "kinds", "inds"} 또는 None(필터 없음)"""
    lo, hi = (years or YEARS)
    f = {"lo": lo if lo > YEARS[0] else None, "hi": hi if hi < YEARS[1] else None,
         "kinds": [k for k in (kinds or []) if k in KINDS], "inds": list(inds or [])}
    return f if any(f.values()) else None


def where(f):
    if not f:
        return None
    parts = ([{"year": {"$gte": f["lo"]}}] if f["lo"] else []) + ([{"year": {"$lte": f["hi"]}}] if f["hi"] else [])
    if f["kinds"]:
        parts.append({"authority": {"$in": f["kinds"]}})
    if f["inds"]:
        parts.append({"$or": [{"industry": {"$in": f["inds"]}}, {"doc_type": {"$in": ["gov", "news_provided"]}}]})
    return None if not parts else parts[0] if len(parts) == 1 else {"$and": parts}


def _year(c):
    p = str(c.get("period") or "")[:4]
    return int(p) if p.isdigit() else None


def allowed_cases(f):
    """필터 조건에 맞는 사례 id 집합 (필터 없으면 None = 제한 없음). 소비자 반응을 빼면 사례 근거가 없으므로 빈 집합"""
    if not f:
        return None
    if f["kinds"] and "소비자 반응" not in f["kinds"]:
        return set()
    ok = set()
    for cid, c in cases().items():
        y = _year(c)
        if f["inds"] and c["industry"] not in f["inds"]:
            continue
        if (f["lo"] and (y is None or y < f["lo"])) or (f["hi"] and (y is None or y > f["hi"])):
            continue
        ok.add(cid)
    return ok


def label(f):
    if not f:
        return ""
    lo, hi = f["lo"], f["hi"]
    yrs = f"{lo}~{hi}년" if lo and hi else f"{lo}년 이후" if lo else f"{hi}년까지" if hi else ""
    return "필터: " + " · ".join(x for x in [yrs, ", ".join(f["kinds"]), ", ".join(f["inds"])] if x)
