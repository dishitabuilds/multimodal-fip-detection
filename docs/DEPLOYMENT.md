# Free demo deployment

Use **Streamlit Community Cloud** for the first college-project demo. It hosts
Streamlit apps for free and deploys from GitHub. The configured app is a research
workbench; deployment does not establish classifier accuracy.

## Deployment settings

- Repository: `dishitabuilds/multimodal-fip-detection`
- Branch: `codex/streamlit-demo` (the tested deployment snapshot).
- Entrypoint: `deployment/streamlit_app.py`
- Python: **3.12** in Advanced settings.
- No collection API key is needed for the demo.

Community Cloud reads `deployment/requirements.txt` beside the entrypoint before
the root requirements file. This installs the frontend, classical lab and media
dependencies, with CPU-only PyTorch wheels. Keep the collection-only root
requirements unchanged.

The cloud entrypoint prepares public EasyOCR English/Hindi weights on first use,
and the public multilingual Whisper base weights when Audio Video is first
visited. Preparation can take several minutes after a fresh instance starts.
Model setup uses a separate process and sends no uploads or captions upstream.
Image, text and media analysis still use locally prepared weights. Running the
existing `app.py` directly retains its offline-only startup.

## Before creating the app

1. Review, test, commit and push the current app source, `src/`, `pages/`,
   `configs/`, `scripts/`, `.streamlit/config.toml`, `test_samples/`, `pyproject.toml`
   and `deployment/`. The local checkout contains substantial uncommitted work;
   the existing GitHub branch does not automatically include it.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with the
   GitHub account that can access this repository.
3. Select **Create app**, enter the settings above, and deploy.
4. Verify a sample image's OCR, a caption-only Promotion Audit, and a short audio
   clip. Confirm Hindi OCR and speech separately before demonstrating them.

The root `.gitignore` keeps collected data and model files out of Git. Do not
upload the virtual environment, raw datasets, user uploads or credentials.
The first cloud instance downloads around **285 MiB of OCR weights**, plus
approximately **141 MiB of speech weights** if the media page is used.

## Classifiers and demo scope

The tiny fused pilot checkpoint, its encoder/tokenizer assets, the seven ML Lab
classifiers and MobileNet weights are local ignored artifacts. They are **not**
included in this deployment setup. A fresh deployment supports OCR, rule-based
Promotion Audit, speech transcription and sampled-frame OCR. ML Lab reports that
no trained models are available, and research inference remains unavailable.

To include pilot scoring in a later release, explicitly package only the trusted
pilot checkpoint and its matching assets at the paths in `README.md` and
`docs/ML_LAB.md`. Preserve the internal-pilot metadata and opt-in notices. Joblib
artifacts must use the same scikit-learn version as training. No validated
research classifier is currently supplied.

## Free-host limits

Streamlit documents shared resources of up to approximately 2.7 GB RAM and two
CPU cores, with limits subject to change. End-to-end OCR plus speech memory has
not been measured on this host. Treat this as a small demo until cloud testing
establishes capacity; cold starts and concurrent media jobs may be slow or exceed
the available memory. Download evidence reports instead of relying on server
storage to preserve them.

For a larger ML demo, Hugging Face Docker Spaces provides CPU Basic hardware
with 16 GB RAM and two CPU cores. Its current policy requires a paid plan to
create Docker compute Spaces, so it is not the free-first choice.

Sources checked on 2 November 2026:

- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [Dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Hugging Face Spaces policy and hardware](https://huggingface.co/docs/hub/spaces-overview)
