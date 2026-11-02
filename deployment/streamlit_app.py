"""Community Cloud entrypoint; the local app retains its offline-only behavior."""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.frontend.cloud import prepare_cloud_assets  # noqa: E402

st.set_page_config(page_title="Promotion Lens", layout="wide")
page = st.navigation([
    st.Page(str(ROOT / "app.py"), title="Image analysis", default=True),
    st.Page(str(ROOT / "pages/1_ML_Lab.py"), title="ML Lab", url_path="ml-lab"),
    st.Page(str(ROOT / "pages/2_Promotion_Audit.py"), title="Promotion Audit",
            url_path="promotion-audit"),
    st.Page(str(ROOT / "pages/3_Audio_Video.py"), title="Audio Video", url_path="audio-video"),
])


@st.cache_resource(show_spinner=False)
def prepare_assets(speech):
    prepare_cloud_assets(ROOT, speech=speech)


try:
    with st.spinner("Preparing the demo. The first visit can take a few minutes."):
        prepare_assets(page.url_path == "audio-video")
except RuntimeError:
    st.warning("Some demo features could not be prepared. Text-only inspection is still "
               "available. Refresh to retry; the app owner can check the server logs.")

page.run()
