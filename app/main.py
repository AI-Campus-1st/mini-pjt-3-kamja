"""Streamlit 시작점: 페이지 구성 (챗봇 / 기획서 확인(심화) / 평가 결과, 가이드 4-8 별도 탭)
실행 (프로젝트 루트): HF_HUB_OFFLINE=1 streamlit run app/main.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # rag/ 를 불러오기 위해

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")              # 키는 .env에만

import streamlit as st
from app.chat import render_chat
from app.dashboard import render_dashboard
from app.plan_tab import render_plan

st.set_page_config(page_title="AI 광고 반응 예상 챗봇", page_icon="💬", layout="wide")
st.navigation([st.Page(render_chat, title="챗봇", icon="💬", default=True),
               st.Page(render_plan, title="기획서 확인", icon="📝"),
               st.Page(render_dashboard, title="성능 평가", icon="📊")], position="top").run()