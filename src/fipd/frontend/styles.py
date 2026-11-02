"""Static presentation for the Promotion Lens research workbench.

CSS is a raw stylesheet; the calling UI owns its <style> element. No runtime
dependencies, remote fonts or Streamlit APIs are needed by this module.
"""

CSS = """
:root {
    --lens-background: #f5f7fa;
    --lens-panel: #ffffff;
    --lens-ink: #12233f;
    --lens-muted: #5b6b80;
    --lens-faint: #79879a;
    --lens-line: #dce3eb;
    --lens-accent: #147d78;
    --lens-accent-soft: #edf7f5;
}

html, body, [data-testid="stApp"] {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    color: var(--lens-ink);
    background: var(--lens-background);
}

input, textarea, button, select { font-family: inherit; }

[data-testid="stMarkdownContainer"] :is(
    .brand-name, .brand-kicker, .hero-eyebrow, .hero-description,
    .workflow-step, .section-eyebrow, .panel-title, .panel-note,
    .status-pill, .result-heading, .result-description, .score-label,
    .score-value, .cue-title, .cue-description, .source-note, .report-footnote
) {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--lens-background);
}

[data-testid="stHeader"] { background: var(--lens-background); }

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1360px;
    padding: 4rem 2.5rem 3.5rem;
}

[data-testid="stSidebar"] {
    background: #eff3f7;
    border-right: 1px solid var(--lens-line);
}

[data-testid="stSidebarUserContent"] { padding: 1.75rem 1.25rem 2rem; }

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--lens-muted);
    font-size: 0.875rem;
    line-height: 1.55;
}

[data-testid="stSidebar"] button p,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-primary"] p {
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.35;
}

[data-testid="stSidebar"] button {
    min-height: 2.7rem;
    height: auto;
    padding-top: 0.65rem;
    padding-bottom: 0.65rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.7rem;
}

.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.35rem;
    height: 2.35rem;
    flex: 0 0 2.35rem;
    border: 1px solid var(--lens-line);
    border-radius: 9px;
    background: var(--lens-panel);
    color: var(--lens-accent);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.brand-name, [data-testid="stMarkdownContainer"] .brand-name {
    color: var(--lens-ink);
    font-size: 1rem;
    font-weight: 650;
    line-height: 1.3;
    letter-spacing: -0.02em;
}

.brand-kicker, [data-testid="stMarkdownContainer"] .brand-kicker {
    margin-top: 0.15rem;
    color: var(--lens-muted);
    font-size: 0.7rem;
    line-height: 1.4;
}

.hero-eyebrow, .section-eyebrow,
[data-testid="stMarkdownContainer"] .hero-eyebrow,
[data-testid="stMarkdownContainer"] p.section-eyebrow {
    color: var(--lens-muted);
    font-size: 0.69rem;
    font-weight: 650;
    letter-spacing: 0.12em;
    line-height: 1.5;
    text-transform: uppercase;
}

.hero-eyebrow { margin: 0 0 0.65rem; }

.hero-title, [data-testid="stMarkdownContainer"] h1.hero-title {
    margin: 0;
    padding: 0;
    max-width: 52rem;
    color: var(--lens-ink);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.25rem;
    font-weight: 400;
    letter-spacing: -0.04em;
    line-height: 1.17;
    text-wrap: balance;
}

.hero-description, [data-testid="stMarkdownContainer"] p.hero-description {
    max-width: 44rem;
    margin: 0.85rem 0 1.6rem;
    color: var(--lens-muted);
    font-size: 0.96rem;
    line-height: 1.7;
}

.workflow-steps {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 1.7rem;
    margin: 0 0 1.9rem;
    padding: 0.85rem 0;
    border-top: 1px solid var(--lens-line);
    border-bottom: 1px solid var(--lens-line);
}

.workflow-step, [data-testid="stMarkdownContainer"] .workflow-step {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    color: var(--lens-muted);
    font-size: 0.78rem;
    line-height: 1.6;
}

.workflow-step strong { color: var(--lens-ink); font-weight: 600; }

.st-key-input_panel, .st-key-results_panel {
    min-width: 0;
    padding: 1.5rem;
    background: var(--lens-panel);
    border: 1px solid var(--lens-line);
    border-radius: 12px;
}

.section-eyebrow { margin: 0 0 0.45rem; }

.panel-title, [data-testid="stMarkdownContainer"] h2.panel-title {
    margin: 0;
    padding: 0;
    color: var(--lens-ink);
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.025em;
    line-height: 1.4;
}

.panel-note, [data-testid="stMarkdownContainer"] p.panel-note {
    margin: 0.5rem 0 1.3rem;
    color: var(--lens-muted);
    font-size: 0.83rem;
    line-height: 1.65;
}

.status-pill, [data-testid="stMarkdownContainer"] .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    width: fit-content;
    max-width: 100%;
    margin: 0.25rem 0 0.6rem;
    padding: 0.25rem 0.6rem;
    border: 1px solid var(--lens-line);
    border-radius: 6px;
    color: var(--lens-muted);
    background: #f7f9fb;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1.5;
    white-space: normal;
}

.status-pill.ready { color: #11615d; background: #edf7f5; border-color: #c9e4df; }
.status-pill.pending { color: #59697b; background: #f3f5f8; }
.status-pill.experimental { color: #775a29; background: #fbf5e9; border-color: #eadfc8; }

.evidence-empty {
    display: flex;
    min-height: 21rem;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.25rem 1.5rem;
    text-align: center;
}

.evidence-icon {
    display: inline-flex;
    width: 3rem;
    height: 3rem;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
    border: 1px solid var(--lens-line);
    border-radius: 10px;
    color: var(--lens-faint);
    background: #f9fafc;
}

.evidence-icon svg { width: 1.4rem; height: 1.4rem; }

.result-heading, [data-testid="stMarkdownContainer"] .result-heading {
    margin: 0 0 0.55rem;
    padding: 0;
    color: var(--lens-ink);
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: -0.025em;
    line-height: 1.35;
}

.result-description, [data-testid="stMarkdownContainer"] .result-description {
    max-width: 35rem;
    margin: 0;
    color: var(--lens-muted);
    font-size: 0.88rem;
    line-height: 1.7;
}

.evidence-empty .result-description { max-width: 25rem; }

.score-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.1rem;
    margin: 1.4rem 0;
    padding: 1.1rem 0;
    border-top: 1px solid var(--lens-line);
    border-bottom: 1px solid var(--lens-line);
}

.score-item { min-width: 0; }
.score-label, [data-testid="stMarkdownContainer"] .score-label { color: var(--lens-muted); font-size: 0.73rem; line-height: 1.5; }
.score-value, [data-testid="stMarkdownContainer"] .score-value { margin-top: 0.35rem; color: var(--lens-ink); font-size: 1.55rem; font-weight: 600; }

.cue-list { margin: 1rem 0 0; }
.cue-item { margin: 0; padding: 0.85rem 0; border-bottom: 1px solid var(--lens-line); }
.cue-item:last-child { border-bottom: 0; }
.cue-title, [data-testid="stMarkdownContainer"] p.cue-title { margin: 0; color: var(--lens-ink); font-size: 0.83rem; font-weight: 600; line-height: 1.55; }
.cue-description, [data-testid="stMarkdownContainer"] p.cue-description { margin: 0.25rem 0 0; color: var(--lens-muted); font-size: 0.8rem; line-height: 1.65; }

.source-note, [data-testid="stMarkdownContainer"] p.source-note {
    margin: 0.65rem 0 1rem;
    padding-left: 0.75rem;
    border-left: 2px solid var(--lens-line);
    color: var(--lens-muted);
    font-size: 0.76rem;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.report-footnote, [data-testid="stMarkdownContainer"] p.report-footnote {
    margin: 1.25rem 0 0;
    padding-top: 0.9rem;
    border-top: 1px solid var(--lens-line);
    color: var(--lens-muted);
    font-size: 0.74rem;
    line-height: 1.7;
}

[data-testid="stFileUploader"] section {
    border: 1px dashed #bac7d5;
    border-radius: 10px;
    background: #fafbfd;
}

[data-testid="stTextArea"] textarea { line-height: 1.6; }

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button { min-height: 2.65rem; height: auto; box-shadow: none; }

[data-testid="stExpander"] {
    border-color: var(--lens-line);
    border-radius: 10px;
    background: var(--lens-panel);
}

a { color: var(--lens-accent); text-underline-offset: 0.18em; }
hr { border-color: var(--lens-line); }

@media (min-width: 761px) {
    [data-testid="stSidebar"] { width: 264px; min-width: 248px; max-width: 280px; }
}

@media (max-width: 760px) {
    .block-container, [data-testid="stMainBlockContainer"] { padding: 4rem 1rem 2.5rem; }
    .hero-title, [data-testid="stMarkdownContainer"] h1.hero-title { font-size: 1.875rem; line-height: 1.2; }
    .hero-description, [data-testid="stMarkdownContainer"] p.hero-description { font-size: 0.9rem; margin-bottom: 1.25rem; }
    .workflow-steps { gap: 0.55rem 1rem; margin-bottom: 1.3rem; }
    .workflow-step { font-size: 0.74rem; }
    .st-key-input_panel, .st-key-results_panel { padding: 1.1rem; }
    .evidence-empty { min-height: 16rem; padding: 1.75rem 0.75rem; }
    .score-strip { gap: 0.9rem; }
    .score-value, [data-testid="stMarkdownContainer"] .score-value { font-size: 1.4rem; }
    .result-heading, [data-testid="stMarkdownContainer"] .result-heading { font-size: 1.2rem; }
}

#a-closer-look { scroll-margin-top: 7rem; }

@media (max-width: 480px) {
    .score-strip { grid-template-columns: 1fr; }
    .score-item + .score-item { padding-top: 0.8rem; border-top: 1px solid var(--lens-line); }
}
"""


def header_html() -> str:
    """Optional static identity block; safe to render without model state."""
    return (
        '<div class="brand"><span class="brand-mark" aria-hidden="true">PL</span>'
        '<div><div class="brand-name">Promotion Lens</div>'
        '<div class="brand-kicker">A multimodal research workbench</div></div></div>'
    )


def empty_state_html() -> str:
    """Optional neutral state before any evidence or model output exists."""
    return (
        '<div class="evidence-empty"><div class="evidence-icon" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4">'
        '<rect x="4" y="3" width="16" height="18" rx="2"/>'
        '<path d="M8 8h8M8 12h8M8 16h5"/></svg></div>'
        '<div class="result-heading">A closer look starts here</div>'
        '<p class="result-description">Choose an example or upload an image to inspect '
        'its text and visual evidence.</p></div>'
    )
