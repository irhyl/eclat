import streamlit as st
import numpy as np
import json

st.set_page_config(page_title='éclat demo', layout='centered')
st.title('éclat — Underwriting Demo (Dummy Values)')

st.markdown('This demo shows a minimal mapping from simple features to a PD estimate and Expected Loss (EL). Values are synthetic and for demo only.')

# Inputs
age = st.slider('Age', 18, 75, 35)
annual_income = st.number_input('Annual income (USD)', min_value=1000, value=40000, step=1000)
credit_score = st.slider('Credit score', 300, 850, 650)
existing_balance = st.number_input('Existing balance (USD)', min_value=0, value=2000, step=100)

# Simple logistic scoring (dummy weights)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

w0 = -3.0
w_age = -0.01
w_income = -0.00002
w_score = -0.005
w_balance = 0.0001

score_raw = w0 + w_age * age + w_income * annual_income + w_score * credit_score + w_balance * existing_balance
pd_point = float(sigmoid(score_raw))

# Conservative percentile (bootstrap-style approximation using noise)
rng = np.random.default_rng(42)
samples = sigmoid(score_raw + rng.normal(0, 0.2, size=1000))
pd_95 = float(np.percentile(samples, 95))

# Expected loss demo
import streamlit as st
import numpy as np
import json
from textwrap import dedent


st.set_page_config(page_title='éclat — Demo', layout='wide')

# --- Style ---
st.markdown(
    "<style>"
    "body {background-color: #fbfbfb;}"
    ".big-title {font-size:28px; font-weight:700;}"
    ".muted {color:#6c757d;}"
    ".card {background: white; padding: 16px; border-radius:10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);}"
    "</style>",
    unsafe_allow_html=True,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_pd(el_inputs):
    # deterministic demo weights (chic minimal surrogate)
    w = {
        'intercept': -3.0,
        'age': -0.01,
        'income': -0.00002,
        'score': -0.005,
        'balance': 0.0001,
    }
    raw = (
        w['intercept']
        + w['age'] * el_inputs['age']
        + w['income'] * el_inputs['annual_income']
        + w['score'] * el_inputs['credit_score']
        + w['balance'] * el_inputs['existing_balance']
    )
    pd_point = float(sigmoid(raw))
    # conservative 95th percentile via simple bootstrap noise
    rng = np.random.default_rng(123)
    samples = sigmoid(raw + rng.normal(0, 0.2, size=2000))
    pd_95 = float(np.percentile(samples, 95))
    return pd_point, pd_95, w


def render_sanction_preview(artifact):
    tmpl = dedent(
        """
        Sanction Letter (Preview)

        Dear Applicant,

        Based on the information provided (ID: {model_version}), we are pleased to inform you of the provisional sanction.

        Estimated PD (point): {pd_point:.2%}
        Estimated PD (conservative): {pd_95:.2%}

        Expected Loss (conservative estimate): ${el_conservative:,.2f}

        This preview is indicative and subject to verification and final underwriting.

        Regards,
        éclat — Sanction Preview
        """
    )
    return tmpl.format(**artifact)


# --- Layout ---
with st.container():
    left, right = st.columns([3, 1])
    with left:
        st.markdown('<div class="big-title">éclat — Underwriting & Conversational Demo</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Minimalist demo to showcase PD mapping, conservative provisioning and an auditable artifact.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"> <strong>Demo</strong><br/>Status: <span style="color:green">Ready</span></div>', unsafe_allow_html=True)

st.markdown('---')

tab_overview, tab_demo, tab_team, tab_artifacts = st.tabs(["Overview", "Interactive Demo", "Team & Contacts", "Artifacts & Notebooks"])

with tab_overview:
    st.header('Executive summary')
    st.write(
        'éclat is an orchestration-first conversational system that preserves customer dignity while mapping model outputs to actionable underwriting decisions and conservative provisioning artifacts.'
    )
    st.write('Use the `Interactive Demo` tab to try dummy inputs and download a sample artifact for audit purposes.')

with tab_demo:
    st.header('Interactive Underwriting Demo')
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader('Inputs')
        age = st.slider('Age', 18, 75, 33)
        annual_income = st.number_input('Annual income (USD)', min_value=500, value=42000, step=500)
        credit_score = st.slider('Credit score', 300, 850, 680)
        existing_balance = st.number_input('Existing balance (USD)', min_value=0, value=1500, step=100)
        show_explain = st.checkbox('Show deterministic feature contributions', value=True)
        run = st.button('Compute PD & EL')

    with c2:
        st.subheader('Results')
        if not run:
            st.info('Adjust inputs on the left and click "Compute PD & EL"')
        else:
            el_inputs = {
                'age': age,
                'annual_income': annual_income,
                'credit_score': credit_score,
                'existing_balance': existing_balance,
            }
            pd_point, pd_95, weights = compute_pd(el_inputs)
            LGD = 0.45
            EAD = existing_balance + 0.1 * annual_income
            el_point = pd_point * LGD * EAD
            el_conservative = pd_95 * LGD * EAD

            st.metric('PD (point)', f"{pd_point:.4f}", delta=None)
            st.metric('PD (95th pctile)', f"{pd_95:.4f}")
            st.write('Expected Loss (point):', f"${el_point:,.2f}")
            st.write('Expected Loss (conservative):', f"${el_conservative:,.2f}")

            # progress bar for confidence (higher pd -> lower confidence here, demo only)
            conf = max(0.05, 1.0 - pd_point)
            st.progress(int(conf * 100))

            if show_explain:
                # deterministic contributions
                contrib = {
                    'age': weights['age'] * age,
                    'income': weights['income'] * annual_income,
                    'score': weights['score'] * credit_score,
                    'balance': weights['balance'] * existing_balance,
                }
                st.subheader('Feature contributions (deterministic surrogate)')
                st.bar_chart({k: v for k, v in contrib.items()})

            artifact = {
                'inputs': el_inputs,
                'pd_point': pd_point,
                'pd_95': pd_95,
                'el_point': el_point,
                'el_conservative': el_conservative,
                'model_version': 'demo-v0.1',
            }

            st.download_button('Download artifact (JSON)', json.dumps(artifact, indent=2).encode('utf-8'), file_name='artifact_demo.json')

            st.markdown('---')
            st.subheader('Sanction letter preview')
            preview = render_sanction_preview(artifact)
            st.code(preview)

with tab_team:
    st.header('Team & Contacts')
    st.markdown('- **Aditi Ramakrishnan** — Conversational AI & Orchestration')
    st.markdown('- **Manisha Kumari Joshi** — Specialized Agent / Backend Developer')
    st.markdown('- **Srusti M Suryavanshi** — LLM MLOps & Data Persistence')

with tab_artifacts:
    st.header('Artifacts & Notebooks')
    st.write('The notebooks are exported to `html/` for convenient sharing. Open the files locally or serve them from a static host.')
    st.markdown('- `html/00-system-overview.html`')
    st.markdown('- `html/01-environment.html`')
    st.markdown('- `html/02-data-preparation.html`')
    st.markdown('- `html/03-core-state-models.html`')
    st.markdown('- `html/04-sales-agent.html`')
    st.markdown('- `html/05-underwriting-agent.html`')
    st.markdown('- `html/06-psychological-theories.html`')
    st.markdown('- `html/07-sociological-theories.html`')
    st.markdown('- `html/08-financial-theories.html`')
    st.markdown('\n— Use `python -m http.server` in the `html/` folder to preview locally.')
