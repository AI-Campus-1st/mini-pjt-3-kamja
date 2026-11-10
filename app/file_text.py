"""
기획서 확인 탭: 올린 파일 → 글자 (메모리에서만 읽고 저장·임베딩하지 않음)
- PDF: pypdf (이미 requirements에 있음) / Word(.docx)·한글(.hwpx): 압축 파일 안 XML을 직접 읽음 (새 라이브러리 없음)
- 구형 한글(.hwp)은 지원하지 않음 → 한글에서 PDF로 저장해 올리도록 안내 (라이브러리 미검증, 리포트 개선 후보)
- 표는 칸마다 한 줄로 나옴. PDF는 2단 편집·표에서 글 순서가 섞일 수 있음 (정량 측정 밖, 정성 확인만)
"""
import io, re, unicodedata, zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader

SPACE_MIN = 0.1           # 이보다 띄어쓰기가 적으면 PDF 추출이 붙어 버린 것으로 봄
MAX_CHARS = 20000          # LLM에 한 번에 보내는 기획서 길이 상한 (넘으면 앞부분만, 화면에 알림)
TYPES = ["pdf", "docx", "hwpx", "hwp"]


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _paragraphs(xml_bytes, p_tag, t_tag):
    """XML에서 문단마다 한 줄. 표 안 문단(문단 속 문단)은 따로 한 줄로 내고 바깥 문단과 섞지 않음"""
    out = []

    def walk(node, buf):
        tag = _local(node.tag)
        if tag == p_tag:
            slot, mine = len(out), []
            out.append(None)                         # 바깥 문단 자리를 먼저 잡아 표 칸보다 앞에 오게
            for ch in node:
                walk(ch, mine)
            out[slot] = "".join(mine).strip()
            return
        if tag == t_tag:
            buf.append("".join(node.itertext()))
            return
        for ch in node:
            walk(ch, buf)

    walk(ET.fromstring(xml_bytes), [])
    return [line for line in out if line]


def _docx(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return "\n".join(_paragraphs(z.read("word/document.xml"), "p", "t"))


def _hwpx(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        secs = sorted((n for n in z.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n)),
                      key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[-1]).group()))
        if not secs:
            raise ValueError("HWPX 안에서 본문(section)을 찾지 못함")
        return "\n".join(line for n in secs for line in _paragraphs(z.read(n), "p", "t"))


def _space_ratio(t):
    """띄어쓰기 확인: 공백 수 / 한글 글자 수 (보통 한국어 글은 0.2 이상)"""
    return t.count(" ") / max(1, len(re.findall(r"[가-힣]", t)))


def _pdf(data):
    """기본 추출에서 띄어쓰기가 사라지면(글자 위치로만 간격을 둔 PDF) 글자 위치를 보는 layout 방식으로 다시 뽑아 더 나은 쪽"""
    pages = PdfReader(io.BytesIO(data)).pages
    text = "\n".join((pg.extract_text() or "").strip() for pg in pages)
    if _space_ratio(text) < SPACE_MIN:
        alt = "\n".join((pg.extract_text(extraction_mode="layout") or "").strip() for pg in pages)
        if _space_ratio(re.sub(r"[ \t]{2,}", " ", alt)) > _space_ratio(text):
            text = alt
    return text


def read_upload(name, data):
    """(글, 안내 문구). 글이 없으면 안내 문구가 이유. 오류는 그대로 냄 (화면이 기록)"""
    ext = unicodedata.normalize("NFC", name).rsplit(".", 1)[-1].lower()
    if ext == "hwp":
        return "", "구형 한글(.hwp) 파일은 아직 읽지 못합니다. 한글에서 'PDF로 저장' 또는 'HWPX로 저장'한 뒤 올려 주세요."
    text = {"pdf": _pdf, "docx": _docx, "hwpx": _hwpx}[ext](data)
    text = re.sub(r"[ \t\u00a0]{2,}", " ", text)                         # PDF의 긴 공백 줄이기
    text = unicodedata.normalize("NFC", re.sub(r"\n{3,}", "\n\n", text)).strip()   # 맥 한글 NFC
    if not text:
        return "", "파일에서 글자를 찾지 못했습니다. 스캔한 이미지 PDF라면 글자를 읽을 수 없어요."
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS], f"기획서가 길어 앞 {MAX_CHARS:,}자만 넣었습니다. 필요한 부분만 남겨 주세요."
    if ext == "pdf" and _space_ratio(text) < SPACE_MIN:
        return text, "이 PDF는 띄어쓰기가 사라진 채로 읽혔어요. 결과가 부정확할 수 있으니 Word·HWPX로 올리거나 내용을 직접 붙여 넣어 주세요."
    note = "PDF는 표·2단 편집에서 글 순서가 섞일 수 있어요. 아래 칸에서 확인하고 고쳐 주세요." if ext == "pdf" else ""
    return text, note