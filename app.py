
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CreditLens | Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F3F6FA;
    color: #172B4D;
}

[data-testid="stSidebar"] {
    background: #14243D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.main-title {
    background: #14243D;
    padding: 28px 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.main-title h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 5px;
}

.main-title p {
    color: #C6D5E8;
    margin: 0;
}

.section-title {
    color: #14243D;
    font-size: 22px;
    font-weight: 700;
    margin: 20px 0 12px 0;
}

.info-card {
    background: white;
    border: 1px solid #E1E8F0;
    border-radius: 13px;
    padding: 20px;
    min-height: 110px;
    box-shadow: 0 3px 12px rgba(20,36,61,0.04);
}

.card-label {
    color: #687B91;
    font-size: 13px;
    font-weight: 600;
}

.card-value {
    color: #14243D;
    font-size: 26px;
    font-weight: 800;
    margin-top: 8px;
}

.result-good {
    background: #E7F7F1;
    border: 1px solid #9DDDC5;
    padding: 24px;
    border-radius: 14px;
    color: #087A57;
}

.result-bad {
    background: #FFF0ED;
    border: 1px solid #F2B9AE;
    padding: 24px;
    border-radius: 14px;
    color: #B43B32;
}

.stButton > button[kind="primary"] {
    background: #087F8C;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button[kind="primary"]:hover {
    background: #05636E;
    color: white;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #DFE7EF;
    padding: 22px;
    border-radius: 14px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #E1E8F0;
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #78899C;
    padding: 25px;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATASET AND MODEL
# --------------------------------------------------

COLUMNS = [
    'checking_account', 'duration', 'credit_history',
    'purpose', 'credit_amount', 'savings',
    'employment', 'installment_rate', 'personal_status',
    'other_debtors', 'residence', 'property',
    'age', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'dependents',
    'telephone', 'foreign_worker'
]

@st.cache_resource
def load_resources():

    model = joblib.load("credit_scoring_model.pkl")
    preprocessor = joblib.load("credit_preprocessor.pkl")

    df = pd.read_csv(
        "german.data",
        sep=r'\s+',
        header=None,
        names=COLUMNS + ['credit_risk']
    )

    df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})

    return model, preprocessor, df


try:
    model, preprocessor, df = load_resources()
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.info(
        "Make sure german.data, credit_scoring_model.pkl "
        "and credit_preprocessor.pkl are in the same folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# CATEGORY DESCRIPTIONS
# --------------------------------------------------

LABELS = {
    'checking_account': {
        'A11': 'Below 0 DM',
        'A12': '0 to below 200 DM',
        'A13': '200 DM or more',
        'A14': 'No checking account'
    },

    'credit_history': {
        'A30': 'No credits / all paid',
        'A31': 'All credits paid at this bank',
        'A32': 'Existing credits paid duly',
        'A33': 'Delay in paying past credits',
        'A34': 'Critical account / other credits'
    },

    'purpose': {
        'A40': 'New car',
        'A41': 'Used car',
        'A42': 'Furniture / Equipment',
        'A43': 'Radio / Television',
        'A44': 'Domestic appliances',
        'A45': 'Repairs',
        'A46': 'Education',
        'A47': 'Vacation',
        'A48': 'Retraining',
        'A49': 'Business',
        'A410': 'Other'
    },

    'savings': {
        'A61': 'Below 100 DM',
        'A62': '100 to below 500 DM',
        'A63': '500 to below 1000 DM',
        'A64': '1000 DM or more',
        'A65': 'Unknown / no savings'
    },

    'employment': {
        'A71': 'Unemployed',
        'A72': 'Below 1 year',
        'A73': '1 to below 4 years',
        'A74': '4 to below 7 years',
        'A75': '7 years or more'
    },

    'personal_status': {
        'A91': 'Male: divorced / separated',
        'A92': 'Female: divorced / married',
        'A93': 'Male: single',
        'A94': 'Male: married / widowed',
        'A95': 'Female: single'
    },

    'other_debtors': {
        'A101': 'None',
        'A102': 'Co-applicant',
        'A103': 'Guarantor'
    },

    'property': {
        'A121': 'Real estate',
        'A122': 'Building savings / life insurance',
        'A123': 'Car or other property',
        'A124': 'Unknown / no property'
    },

    'other_installment_plans': {
        'A141': 'Bank',
        'A142': 'Stores',
        'A143': 'None'
    },

    'housing': {
        'A151': 'Rent',
        'A152': 'Own house',
        'A153': 'Free accommodation'
    },

    'job': {
        'A171': 'Unemployed / unskilled non-resident',
        'A172': 'Unskilled resident',
        'A173': 'Skilled employee / official',
        'A174': 'Management / self-employed'
    },

    'telephone': {
        'A191': 'No telephone',
        'A192': 'Telephone registered'
    },

    'foreign_worker': {
        'A201': 'Yes',
        'A202': 'No'
    }
}


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

@st.cache_data
def calculate_metrics():

    X = df[COLUMNS]
    y = df['credit_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    processed = preprocessor.transform(X_test)

    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions)

    return accuracy, auc, matrix, y_test, predictions


accuracy, auc, matrix, y_test, predictions = calculate_metrics()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("""
<div style="text-align:center; padding:20px 0;">
    <h1 style="font-size:32px;">💳</h1>
    <h2 style="margin:0;">CreditLens</h2>
    <p style="font-size:12px;color:#B9CADC;">
        CREDIT RISK INTELLIGENCE
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Credit Prediction",
        "Model Analytics",
        "Methodology"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("MACHINE LEARNING PROJECT")
st.sidebar.caption("CodeAlpha Internship")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

def page_header(title, subtitle):

    st.markdown(f"""
    <div class="main-title">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value):

    st.markdown(f"""
    <div class="info-card">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    page_header(
        "Credit Risk Intelligence",
        "Machine learning powered creditworthiness analysis"
    )

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "CreditLens is an educational machine learning application "
        "that analyses customer financial information and predicts "
        "whether a customer belongs to the good-credit or bad-credit "
        "category in the German Credit dataset."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Records", "1,000")

    with c2:
        metric_card("Input Features", "20")

    with c3:
        metric_card("Test Accuracy", f"{accuracy*100:.1f}%")

    with c4:
        metric_card("ROC-AUC", f"{auc:.3f}")

    st.markdown(
        '<div class="section-title">Credit Risk Distribution</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:

        counts = df['credit_risk'].value_counts().sort_index()

        chart_data = pd.DataFrame({
            "Credit Category": ["Bad Credit", "Good Credit"],
            "Customers": [counts.get(0, 0), counts.get(1, 0)]
        })

        st.bar_chart(
            chart_data.set_index("Credit Category")
        )

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3 style="color:#14243D;">Project Highlights</h3>
            <p>✔ Customer financial data analysis</p>
            <p>✔ Data preprocessing and encoding</p>
            <p>✔ Machine learning classification</p>
            <p>✔ Prediction probability analysis</p>
            <p>✔ Model performance evaluation</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Selected Algorithm</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Random Forest is the selected model from the training "
        "comparison. Its current test accuracy is displayed above."
    )


# --------------------------------------------------
# PAGE 2: CREDIT PREDICTION
# --------------------------------------------------

elif page == "Credit Prediction":

    page_header(
        "Applicant Assessment",
        "Enter customer information to generate a credit risk prediction"
    )

    st.markdown(
        '<div class="section-title">Applicant Financial Profile</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Complete the fields below. All categorical options use "
        "readable descriptions while preserving the original dataset codes."
    )

    numeric_fields = {
        'duration': ('Loan Duration (Months)', 4, 72, 12),
        'credit_amount': ('Credit Amount (DM)', 250, 20000, 2500),
        'installment_rate': ('Installment Rate (%)', 1, 4, 2),
        'residence': ('Years at Current Residence', 1, 4, 2),
        'age': ('Applicant Age', 19, 75, 35),
        'existing_credits': ('Existing Credits', 1, 4, 1),
        'dependents': ('Number of Dependents', 1, 2, 1)
    }

    defaults = {}

    with st.form("applicant_form"):

        st.markdown("#### 01 · Financial Information")

        c1, c2 = st.columns(2)

        for i, field in enumerate([
            'checking_account',
            'savings',
            'employment',
            'credit_amount'
        ]):

            target = c1 if i % 2 == 0 else c2

            with target:

                if field in LABELS:

                    options = list(LABELS[field].keys())

                    selected = st.selectbox(
                        field.replace('_', ' ').title(),
                        options,
                        format_func=lambda x, f=field:
                            LABELS[f].get(x, x)
                    )

                    defaults[field] = selected

                else:

                    label, low, high, default = numeric_fields[field]

                    defaults[field] = st.number_input(
                        label,
                        min_value=low,
                        max_value=high,
                        value=default
                    )

        st.markdown("#### 02 · Credit History")

        c1, c2 = st.columns(2)

        for i, field in enumerate([
            'credit_history',
            'purpose',
            'personal_status',
            'other_debtors'
        ]):

            target = c1 if i % 2 == 0 else c2

            with target:

                options = list(LABELS[field].keys())

                defaults[field] = st.selectbox(
                    field.replace('_', ' ').title(),
                    options,
                    format_func=lambda x, f=field:
                        LABELS[f].get(x, x)
                )

        st.markdown("#### 03 · Personal and Employment Details")

        c1, c2 = st.columns(2)

        for i, field in enumerate([
            'duration',
            'installment_rate',
            'residence',
            'age',
            'property',
            'other_installment_plans',
            'housing',
            'existing_credits',
            'job',
            'dependents',
            'telephone',
            'foreign_worker'
        ]):

            target = c1 if i % 2 == 0 else c2

            with target:

                if field in LABELS:

                    options = list(LABELS[field].keys())

                    defaults[field] = st.selectbox(
                        field.replace('_', ' ').title(),
                        options,
                        format_func=lambda x, f=field:
                            LABELS[f].get(x, x)
                    )

                else:

                    label, low, high, default = numeric_fields[field]

                    defaults[field] = st.number_input(
                        label,
                        min_value=low,
                        max_value=high,
                        value=default
                    )

        st.markdown("---")

        submitted = st.form_submit_button(
            "Generate Credit Risk Assessment",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [defaults],
            columns=COLUMNS
        )

        processed = preprocessor.transform(input_df)

        prediction = model.predict(processed)[0]

        probabilities = model.predict_proba(processed)[0]

        bad_probability = probabilities[0] * 100
        good_probability = probabilities[1] * 100

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Assessment Results</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.markdown(f"""
            <div class="result-good">
                <h2>✓ GOOD CREDIT RISK</h2>
                <p>The model classifies this applicant in the
                good-credit category.</p>
                <h3>Good Credit Probability: {good_probability:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-bad">
                <h2>! BAD CREDIT RISK</h2>
                <p>The model classifies this applicant in the
                bad-credit category.</p>
                <h3>Bad Credit Probability: {bad_probability:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        c1.metric(
            "Good Credit Probability",
            f"{good_probability:.2f}%"
        )

        c2.metric(
            "Bad Credit Probability",
            f"{bad_probability:.2f}%"
        )

        st.markdown("#### Probability Distribution")

        st.progress(float(good_probability / 100))

        st.caption(
            "The probability values are model outputs, "
            "not calibrated estimates of actual lending risk."
        )

        st.warning(
            "Educational demonstration only. This prediction must "
            "not be used as a real loan approval or rejection decision."
        )


# --------------------------------------------------
# PAGE 3: MODEL ANALYTICS
# --------------------------------------------------

elif page == "Model Analytics":

    page_header(
        "Model Performance Analytics",
        "Evaluation of the trained machine learning model"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Test Accuracy", f"{accuracy*100:.2f}%")
    c2.metric("ROC-AUC Score", f"{auc:.3f}")
    c3.metric("Test Records", "200")

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=['Bad Credit', 'Good Credit'],
        yticklabels=['Bad Credit', 'Good Credit'],
        ax=ax
    )

    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")
    ax.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '<div class="section-title">Classification Report</div>',
        unsafe_allow_html=True
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=['Bad Credit', 'Good Credit'],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The following comparison uses the original recorded "
        "training results from Google Colab."
    )

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy (%)": [70.5, 67.5, 74.5]
    })

    st.bar_chart(
        comparison.set_index("Model")
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# PAGE 4: METHODOLOGY
# --------------------------------------------------

elif page == "Methodology":

    page_header(
        "Project Methodology",
        "Understanding the machine learning workflow"
    )

    st.markdown(
        '<div class="section-title">Dataset Information</div>',
        unsafe_allow_html=True
    )

    st.write(
        "This project uses the German Credit dataset from the "
        "UCI Machine Learning Repository."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", "1,000")
    c2.metric("Input Features", "20")
    c3.metric("Target Classes", "2")

    st.markdown(
        '<div class="section-title">Algorithms Used</div>',
        unsafe_allow_html=True
    )

    algorithms = [
        ("Logistic Regression", "70.5%"),
        ("Decision Tree", "67.5%"),
        ("Random Forest", "74.5%")
    ]

    for name, score in algorithms:

        st.markdown(f"""
        <div class="info-card" style="margin-bottom:12px;">
            <div class="card-label">{name}</div>
            <div class="card-value">{score}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    steps = [
        "Dataset collection",
        "Data inspection and cleaning",
        "Categorical encoding",
        "Numerical feature scaling",
        "Train-test splitting (80:20)",
        "Model training",
        "Model evaluation",
        "Credit risk prediction"
    ]

    for i, step in enumerate(steps, 1):

        st.write(f"**{i}.** {step}")

    st.markdown(
        '<div class="section-title">Technology Stack</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Python · Pandas · NumPy · Scikit-learn · "
        "Matplotlib · Seaborn · Streamlit"
    )

    st.info(
        "The model is an educational prototype. It has not been "
        "validated for real-world lending or financial decisions."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    <b>CreditLens</b> · Credit Risk Prediction System<br>
    Built with Python, Scikit-learn and Streamlit<br>
    CodeAlpha Machine Learning Internship · 2026
</div>
""", unsafe_allow_html=True)
