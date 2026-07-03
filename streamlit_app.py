import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import datetime
import time
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from src.features.engineer import FeatureEngineer
from src.config import MODELS_DIR, SENDER_EMAIL, SENDER_PASSWORD

# Set Page Config
st.set_page_config(page_title="AI Churn Predictor", page_icon="🔮", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4B4B4B;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔮 AI Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer retention in real-time</div>', unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    train_df = pd.read_csv("data/train.csv")
    fe = FeatureEngineer()
    fe.fit_transform(train_df)
    
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    lgb_model = joblib.load(MODELS_DIR / "lgb_model.pkl")
    return fe, xgb_model, lgb_model

try:
    fe, xgb_model, lgb_model = load_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"Models not found! Please run `python main.py` first to train the models. Error: {e}")
    pipeline_ready = False

# Function to send email
def send_email_report(boss_email, payload):
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = boss_email
        msg['Subject'] = f"Automated AI Report: Customer Churn Risk ({payload['risk_status']})"
        
        body = f"""
Hello,

The AI Customer Churn Predictor has analyzed a customer profile and generated the following report:

Customer Risk Status: {payload['risk_status']}
Churn Probability: {payload['churn_probability'] * 100:.1f}%

Customer Profile Details:
- Age: {payload['profile_details']['Age']}
- Tenure (Months): {payload['profile_details']['Tenure']}
- Support Calls: {payload['profile_details']['Support_Calls']}
- Payment Delay (Days): {payload['profile_details']['Payment_Delay']}
- Total Spend ($): {payload['profile_details']['Total_Spend']}

Recommendation: { "Offer discount or priority support" if payload['churn_probability'] > 0.5 else "No action needed" }

Best,
AI Predictor Agent
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, boss_email, text)
        server.quit()
        print(f"Success: Email sent to {boss_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise e

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 👤 Customer Profile")
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure (Months)", 1, 60, 12)
    usage = st.slider("Usage Frequency", 1, 30, 15)
    calls = st.slider("Support Calls", 0, 10, 1)
    delay = st.slider("Payment Delay (Days)", 0, 30, 2)
    sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    spend = st.number_input("Total Spend ($)", min_value=100.0, max_value=5000.0, value=500.0)
    interaction = st.slider("Days Since Last Interaction", 1, 30, 5)
    
    st.markdown("---")
    st.markdown("### 📧 Automated Email Report")
    send_report = st.checkbox("Email Report to Boss")
    boss_email = st.text_input("Boss's Email Address", value="boss@company.com")
    schedule_date = st.date_input("Schedule Date", value=datetime.date.today())
    schedule_time = st.time_input("Schedule Time", value=datetime.datetime.now().time())

    predict_btn = st.button("Predict Churn Risk 🚀")

with col2:
    if predict_btn and pipeline_ready:
        st.markdown("### 📊 AI Prediction Analysis")
        
        # Build DataFrame
        input_data = {
            'CustomerID': [99999],
            'Age': [age],
            'Gender': [gender],
            'Tenure': [tenure],
            'Usage Frequency': [usage],
            'Support Calls': [calls],
            'Payment Delay': [delay],
            'Subscription Type': [sub_type],
            'Contract Length': [contract],
            'Total Spend': [spend],
            'Last Interaction': [interaction]
        }
        input_df = pd.DataFrame(input_data)
        
        with st.spinner("Analyzing customer profile using Ensemble AI..."):
            # Transform
            X_test = fe.transform(input_df)
            
            # Predict
            xgb_prob = xgb_model.predict_proba(X_test)[0][1]
            lgb_prob = lgb_model.predict_proba(X_test)[0][1]
            
            ensemble_prob = (xgb_prob + lgb_prob) / 2.0
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ensemble_prob * 100,
                title = {'text': "Churn Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgray"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "gold"},
                        {'range': [66, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75}
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            if ensemble_prob > 0.5:
                risk_status = "High Risk of Churn"
                st.error(f"🚨 **{risk_status}!** The AI predicts a {ensemble_prob*100:.1f}% chance this customer will leave. Consider offering a discount or priority support.")
            else:
                risk_status = "Safe"
                st.success(f"✅ **Customer is {risk_status}!** The AI predicts a low ({ensemble_prob*100:.1f}%) chance of churn.")
                
            # Email Report Logic
            if send_report:
                payload = {
                    "churn_probability": float(ensemble_prob),
                    "risk_status": risk_status,
                    "profile_details": {
                        "Age": age,
                        "Tenure": tenure,
                        "Support_Calls": calls,
                        "Payment_Delay": delay,
                        "Total_Spend": spend
                    }
                }
                
                # Check for credentials in .env
                if not SENDER_EMAIL or not SENDER_PASSWORD:
                    st.warning("⚠️ **Missing Credentials:** You checked 'Email Report', but SENDER_EMAIL or SENDER_PASSWORD is not set in your `.env` file. The email was not sent.")
                else:
                    target_dt = datetime.datetime.combine(schedule_date, schedule_time)
                    now = datetime.datetime.now()
                    delay_seconds = (target_dt - now).total_seconds()
                    
                    if delay_seconds <= 0:
                        st.info("📧 Sending email immediately...")
                        try:
                            send_email_report(boss_email, payload)
                            st.success(f"✅ **Success!** Automated Email Report sent to **{boss_email}**.")
                        except Exception as e:
                            st.error(f"❌ **Error sending email:** {e}")
                    else:
                        st.info(f"⏳ **Scheduled!** Email queued in background worker to send to **{boss_email}** in {delay_seconds:.0f} seconds (at {target_dt}).")
                        t = threading.Timer(delay_seconds, send_email_report, args=[boss_email, payload])
                        t.daemon = True
                        t.start()
                    
    elif not pipeline_ready:
         st.info("Awaiting pipeline training...")
    else:
        st.info("👈 Enter customer details in the sidebar and click Predict to see the AI analysis here.")
