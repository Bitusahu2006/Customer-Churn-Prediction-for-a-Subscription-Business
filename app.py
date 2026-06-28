import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
        color: #667eea;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #764ba2;
    }
    
    /* Metric box */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        text-align: center;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Form input styling */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 16px;
        padding: 12px 30px;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Header styling */
    h1 {
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    
    h2 {
        color: #667eea;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    
    h3 {
        color: #764ba2;
    }
    
    /* Success/Warning styling */
    .stSuccess {
        background-color: #d4edda !important;
        border-color: #28a745 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        border-color: #ffc107 !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        border-color: #f5c6cb !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        border-color: #0c5460 !important;
    }
    
    /* Container styling */
    .container {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk indicator */
    .risk-high {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
    
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 24px;
    }
    
    .risk-low {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """Load pre-trained model and preprocessor"""
    try:
        model_path = Path("models/churn_model.pkl")
        preprocessor_path = Path("models/preprocessor.pkl")
        
        if not model_path.exists() or not preprocessor_path.exists():
            st.error("❌ Model files not found. Please ensure churn_model.pkl and preprocessor.pkl exist in the models/ directory.")
            return None, None
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

@st.cache_resource
def load_feature_importance():
    """Load feature importance data"""
    try:
        fi_path = Path("feature_importance.csv")
        if fi_path.exists():
            return pd.read_csv(fi_path)
        return None
    except Exception as e:
        st.warning(f"Could not load feature importance: {str(e)}")
        return None

# Load model and preprocessor
model, preprocessor = load_model_and_preprocessor()
feature_importance = load_feature_importance()

# Title and Header
st.markdown("""
    <div style='text-align: center; padding: 40px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 48px; text-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
        📱 Telecom Churn Predictor
        </h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 18px; margin-top: 10px;'>
        AI-Powered Customer Retention Intelligence
        </p>
        <p style='color: rgba(255,255,255,0.8); font-size: 14px; margin-top: 5px;'>
        Identify at-risk customers and implement targeted retention strategies
        </p>
    </div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Feature Importance", "ℹ️ About Model", "📈 Analytics"])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.markdown("<h2 style='color: #667eea;'>👤 Customer Profile & Analysis</h2>", unsafe_allow_html=True)
    
    # Create expanders for better organization
    with st.expander("📋 Demographics Information", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.selectbox("👥 Gender", ["Male", "Female"], key="gender")
        with col2:
            senior_citizen = st.selectbox("👴 Senior Citizen", ["No", "Yes"], key="senior")
        with col3:
            partner = st.selectbox("💑 Partner", ["Yes", "No"], key="partner")
        with col4:
            dependents = st.selectbox("👨‍👩‍👧 Dependents", ["Yes", "No"], key="dependents")
    
    with st.expander("⏱️ Tenure & Duration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("📅 How long has the customer been with us? (months)", 
                             min_value=0, max_value=72, value=12, step=1)
        with col2:
            st.metric("Tenure Duration", f"{tenure} months", 
                     delta="High Risk" if tenure < 12 else "Good" if tenure < 24 else "Loyal",
                     delta_color="inverse")
    
    with st.expander("🌐 Internet & Communication Services", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            internet_service = st.selectbox("🌐 Internet Service Type", ["DSL", "Fiber optic", "No"], key="internet")
        with col2:
            phone_service = st.selectbox("☎️ Phone Service", ["Yes", "No"], key="phone")
        with col3:
            multiple_lines = st.selectbox("📞 Multiple Lines", ["Yes", "No", "No phone service"], key="multilines")
        
        st.write("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            online_security = st.selectbox("🔒 Online Security", ["Yes", "No", "No internet service"], key="security")
        with col2:
            online_backup = st.selectbox("💾 Online Backup", ["Yes", "No", "No internet service"], key="backup")
        with col3:
            device_protection = st.selectbox("📱 Device Protection", ["Yes", "No", "No internet service"], key="device")
        
        st.write("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            tech_support = st.selectbox("🛠️ Tech Support", ["Yes", "No", "No internet service"], key="support")
        with col2:
            streaming_tv = st.selectbox("📺 Streaming TV", ["Yes", "No", "No internet service"], key="tv")
        with col3:
            streaming_movies = st.selectbox("🎬 Streaming Movies", ["Yes", "No", "No internet service"], key="movies")
    
    with st.expander("💰 Billing & Contract Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_charges = st.number_input("💵 Monthly Charges ($)", 
                                            min_value=0.0, max_value=200.0, value=50.0, step=1.0)
        with col2:
            total_charges = st.number_input("💵 Total Charges ($)", 
                                          min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
        with col3:
            contract = st.selectbox("📜 Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
        
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            paperless_billing = st.selectbox("📧 Paperless Billing", ["Yes", "No"], key="paperless")
        with col2:
            payment_method = st.selectbox("💳 Payment Method", 
                                         ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 
                                         key="payment")
    
    # Analysis indicators
    st.write("")
    st.markdown("<h3 style='color: #764ba2;'>⚠️ Risk Indicators</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_tenure = "🔴 High" if tenure < 6 else "🟡 Medium" if tenure < 24 else "🟢 Low"
        st.metric("Tenure Risk", risk_tenure)
    
    with col2:
        risk_contract = "🔴 High" if contract == "Month-to-month" else "🟡 Medium" if contract == "One year" else "🟢 Low"
        st.metric("Contract Risk", risk_contract)
    
    with col3:
        risk_charges = "🔴 High" if monthly_charges > 100 else "🟡 Medium" if monthly_charges > 50 else "🟢 Low"
        st.metric("Charges Risk", risk_charges)
    
    with col4:
        support_count = sum([1 for x in [online_security, online_backup, tech_support] if x == "Yes"])
        support_risk = "🟢 Low" if support_count >= 2 else "🟡 Medium" if support_count == 1 else "🔴 High"
        st.metric("Support Risk", support_risk)
    
    # Prediction button
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 ANALYZE & PREDICT CHURN", use_container_width=True)
    
    if predict_button:
        if model is None or preprocessor is None:
            st.error("❌ Cannot make predictions. Model not loaded.")
        else:
            with st.spinner("🔄 Analyzing customer profile..."):
                # Prepare input data
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges]
                })
                
                try:
                    # Preprocess the input
                    input_transformed = preprocessor.transform(input_data)
                    
                    # Make prediction
                    prediction = model.predict(input_transformed)
                    prediction_proba = model.predict_proba(input_transformed)
                    
                    # Display results
                    st.write("")
                    st.markdown("<h2 style='color: #667eea;'>📊 PREDICTION RESULTS</h2>", unsafe_allow_html=True)
                    
                    churn_prob = prediction_proba[0][1]
                    retention_prob = 1 - churn_prob
                    is_churning = prediction[0] == 1
                    
                    # Main result cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if is_churning:
                            st.markdown(f"""
                            <div class='metric-box' style='background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);'>
                                <h3 style='color: white; margin: 0;'>⚠️ CHURN RISK</h3>
                                <h1 style='color: white; margin: 10px 0;'>HIGH</h1>
                                <p style='color: rgba(255,255,255,0.9); margin: 0;'>Action Required</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='metric-box' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                                <h3 style='color: white; margin: 0;'>✅ CHURN RISK</h3>
                                <h1 style='color: white; margin: 10px 0;'>LOW</h1>
                                <p style='color: rgba(255,255,255,0.9); margin: 0;'>Customer Safe</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-box' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                            <h3 style='color: white; margin: 0;'>📈 CHURN</h3>
                            <h1 style='color: white; margin: 10px 0;'>{churn_prob*100:.1f}%</h1>
                            <p style='color: rgba(255,255,255,0.9); margin: 0;'>Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='metric-box' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                            <h3 style='color: white; margin: 0;'>💪 RETENTION</h3>
                            <h1 style='color: white; margin: 10px 0;'>{retention_prob*100:.1f}%</h1>
                            <p style='color: rgba(255,255,255,0.9); margin: 0;'>Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.write("")
                    
                    # Probability visualization
                    st.markdown("<h3 style='color: #667eea;'>📊 Risk Level Visualization</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=churn_prob * 100,
                            title={'text': "Churn Risk Score"},
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#667eea" if churn_prob < 0.5 else "#f5576c"},
                                'steps': [
                                    {'range': [0, 33], 'color': "#d4f1d4"},
                                    {'range': [33, 67], 'color': "#fff3cd"},
                                    {'range': [67, 100], 'color': "#f8d7da"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['Churn Risk', 'Retention Probability'],
                            values=[churn_prob * 100, retention_prob * 100],
                            marker=dict(colors=['#f5576c', '#38ef7d']),
                            textinfo='label+percent',
                            textposition='auto'
                        )])
                        fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    st.write("")
                    
                    # Recommendations
                    if is_churning:
                        st.warning("""
                        ### 🚨 CRITICAL: High Churn Risk Detected!
                        
                        **This customer shows strong signs of potential churn. Immediate action recommended.**
                        """)
                        
                        st.markdown("""
                        #### 💡 Retention Recommendations:
                        
                        1. **Immediate Outreach** 
                           - Contact customer within 24-48 hours
                           - Offer personalized retention discount
                           - Listen to concerns and pain points
                        
                        2. **Service Upgrades**
                           - Offer free premium services trial
                           - Bundle Tech Support + Security
                           - Improve internet speed if needed
                        
                        3. **Contract Incentive**
                           - Encourage upgrade to 1-2 year contract
                           - Offer loyalty discount for long-term commitment
                           - Highlight stability & cost benefits
                        
                        4. **Proactive Engagement**
                           - Assign dedicated account manager
                           - Regular check-ins and support
                           - Priority service for issues
                        
                        5. **Value Demonstration**
                           - Show customer spending vs value received
                           - Highlight included benefits
                           - Compare with competitor offerings
                        """)
                    else:
                        st.success("""
                        ### ✅ Customer Retention Profile: POSITIVE
                        
                        **This customer shows low churn risk. Focus on satisfaction maintenance.**
                        """)
                        
                        st.markdown("""
                        #### 🎯 Retention Strategies:
                        
                        1. **Satisfaction Maintenance**
                           - Ensure consistent service quality
                           - Regular satisfaction surveys
                           - Quick issue resolution
                        
                        2. **Loyalty Programs**
                           - Reward program for long tenure
                           - Birthday/anniversary offers
                           - Referral incentives
                        
                        3. **Upselling Opportunities**
                           - Suggest complementary services
                           - Offer speed upgrades
                           - Bundle entertainment services
                        
                        4. **Community Building**
                           - Include in customer appreciation events
                           - Special VIP benefits
                           - Early access to new features
                        
                        5. **Long-term Value**
                           - Renew contract before expiration
                           - Secure with long-term pricing
                           - Build brand advocacy
                        """)
                    
                    st.write("")
                    
                    # Key factors analysis
                    st.markdown("<h3 style='color: #667eea;'>🔍 Customer Profile Summary</h3>", unsafe_allow_html=True)
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.markdown(f"""
                        <div class='card'>
                            <h4>👤 Tenure & History</h4>
                            <p><strong>Duration:</strong> {tenure} months</p>
                            <p><strong>Monthly Spend:</strong> ${monthly_charges:.2f}</p>
                            <p><strong>Lifetime Value:</strong> ${total_charges:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col2:
                        services_count = sum([
                            1 if x == "Yes" else 0 for x in 
                            [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
                        ])
                        st.markdown(f"""
                        <div class='card'>
                            <h4>🛠️ Services & Benefits</h4>
                            <p><strong>Active Services:</strong> {services_count}/6</p>
                            <p><strong>Internet:</strong> {internet_service if internet_service != 'No' else 'None'}</p>
                            <p><strong>Phone:</strong> {phone_service}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with summary_col3:
                        st.markdown(f"""
                        <div class='card'>
                            <h4>📜 Contract Info</h4>
                            <p><strong>Type:</strong> {contract}</p>
                            <p><strong>Billing:</strong> {'Paperless' if paperless_billing == 'Yes' else 'Paper'}</p>
                            <p><strong>Method:</strong> {payment_method}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")

# ==================== TAB 2: FEATURE IMPORTANCE ====================
with tab2:
    st.markdown("<h2 style='color: #667eea;'>📊 Feature Importance Analysis</h2>", unsafe_allow_html=True)
    st.write("Discover which factors have the most influence on customer churn predictions")
    
    if feature_importance is not None:
        # Sort by importance
        fi_sorted = feature_importance.sort_values('Importance', ascending=True)
        
        # Top features visualization
        st.markdown("<h3 style='color: #764ba2;'>🔝 Top 15 Most Important Features</h3>", unsafe_allow_html=True)
        
        top_15 = fi_sorted.tail(15).copy()
        top_15['Feature'] = top_15['Feature'].str.replace('num__', '').str.replace('cat__', '').str.replace('_', ' ')
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_15['Importance'],
                y=top_15['Feature'],
                orientation='h',
                marker=dict(
                    color=top_15['Importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=top_15['Importance'].round(4),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance Ranking",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600,
            showlegend=False,
            hovermode='closest',
            margin=dict(l=200, r=50, t=80, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("")
        
        # Top 10 detailed breakdown
        st.markdown("<h3 style='color: #764ba2;'>📋 Top 10 Features - Detailed View</h3>", unsafe_allow_html=True)
        
        top_10 = feature_importance.sort_values('Importance', ascending=False).head(10).copy()
        top_10['Feature_Clean'] = top_10['Feature'].str.replace('num__', '').str.replace('cat__', '').str.replace('_', ' ')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px;'>
                <h4>Key Metrics:</h4>
                <ul>
                    <li><strong>Total Features:</strong> 19</li>
                    <li><strong>Top Feature:</strong> Tenure (14.0%)</li>
                    <li><strong>Feature Groups:</strong> Numeric (3) + Categorical (16)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            for idx, (i, row) in enumerate(top_10.iterrows(), 1):
                importance_pct = row['Importance'] * 100
                bar_length = int(importance_pct / 0.2)  # Scale for visual
                
                st.markdown(f"""
                <div style='margin: 15px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <strong>#{idx} {row['Feature_Clean']}</strong>
                        <span style='color: #667eea; font-weight: bold;'>{importance_pct:.2f}%</span>
                    </div>
                    <div style='background-color: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden;'>
                        <div style='background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {importance_pct}%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.write("")
        
        # Business insights
        st.markdown("<h3 style='color: #764ba2;'>💡 Business Insights & Recommendations</h3>", unsafe_allow_html=True)
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.markdown("""
            <div class='card'>
                <h4>🎯 Tenure - The #1 Churn Driver</h4>
                <p><strong>Finding:</strong> Customers with longer tenure are significantly less likely to churn.</p>
                <p><strong>Action:</strong> Invest heavily in first-year customer experience and onboarding.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='card'>
                <h4>💰 Charges - Price Sensitivity</h4>
                <p><strong>Finding:</strong> Monthly and total charges are key factors influencing churn.</p>
                <p><strong>Action:</strong> Offer value bundles and loyalty discounts for high-value customers.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='card'>
                <h4>🔒 Services - Building Stickiness</h4>
                <p><strong>Finding:</strong> Customers with tech support & security services churn less.</p>
                <p><strong>Action:</strong> Bundle services by default; promote security packages.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_col2:
            st.markdown("""
            <div class='card'>
                <h4>📜 Contract - Commitment Lock</h4>
                <p><strong>Finding:</strong> Month-to-month contracts show highest churn risk.</p>
                <p><strong>Action:</strong> Incentivize 1-2 year contracts with special pricing.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='card'>
                <h4>🌐 Internet Service - Quality Issues</h4>
                <p><strong>Finding:</strong> Fiber optic customers churn more than DSL customers.</p>
                <p><strong>Action:</strong> Investigate service quality; implement targeted improvements.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='card'>
                <h4>💳 Payment Methods - Convenience Matters</h4>
                <p><strong>Finding:</strong> Electronic check payments correlate with higher churn.</p>
                <p><strong>Action:</strong> Promote automatic payment options; offer discounts for autopay.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Feature comparison
        st.markdown("<h3 style='color: #764ba2;'>📊 Feature Category Distribution</h3>", unsafe_allow_html=True)
        
        numeric_features = feature_importance[feature_importance['Feature'].str.contains('num__')]['Importance'].sum()
        categorical_features = feature_importance[feature_importance['Feature'].str.contains('cat__')]['Importance'].sum()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Numeric Features', 'Categorical Features'],
            values=[numeric_features, categorical_features],
            marker=dict(colors=['#667eea', '#764ba2']),
            textinfo='label+percent'
        )])
        fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 400px; display: flex; flex-direction: column; justify-content: center;'>
                <h4>Feature Composition</h4>
                <p><strong>Numeric Features Importance:</strong></p>
                <p style='font-size: 24px; color: #667eea;'>{numeric_features*100:.1f}%</p>
                <hr>
                <p><strong>Categorical Features Importance:</strong></p>
                <p style='font-size: 24px; color: #764ba2;'>{categorical_features*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Feature importance data not available")

# ==================== TAB 3: ABOUT MODEL ====================
with tab3:
    st.markdown("<h2 style='color: #667eea;'>ℹ️ Model Information & Details</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h4>🤖 Model Architecture</h4>
            <ul>
                <li><strong>Algorithm:</strong> XGBoost (Gradient Boosting)</li>
                <li><strong>Framework:</strong> Scikit-learn + XGBoost</li>
                <li><strong>Task:</strong> Binary Classification</li>
                <li><strong>Training Method:</strong> GridSearchCV Hyperparameter Tuning</li>
                <li><strong>Validation:</strong> K-Fold Cross-Validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h4>📊 Training Data</h4>
            <ul>
                <li><strong>Dataset:</strong> IBM Telco Customer Churn</li>
                <li><strong>Total Samples:</strong> 7,043 customers</li>
                <li><strong>After Cleaning:</strong> 7,032 samples</li>
                <li><strong>Original Features:</strong> 21</li>
                <li><strong>Engineered Features:</strong> 19</li>
                <li><strong>Feature Types:</strong> 3 Numeric + 16 Categorical</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>✅ Model Performance</h4>
            <ul>
                <li><strong>Accuracy:</strong> High (balanced precision & recall)</li>
                <li><strong>Precision:</strong> Minimizes false positives</li>
                <li><strong>Recall:</strong> Captures most churn cases</li>
                <li><strong>F1-Score:</strong> Optimal balance</li>
                <li><strong>ROC-AUC:</strong> Excellent discrimination</li>
                <li><strong>Status:</strong> ✅ Production Ready</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h4>⚙️ Preprocessing Pipeline</h4>
            <ul>
                <li><strong>Encoding:</strong> One-Hot Encoding</li>
                <li><strong>Scaling:</strong> StandardScaler (Z-score)</li>
                <li><strong>Missing Values:</strong> Handled appropriately</li>
                <li><strong>Outliers:</strong> Retained (business relevant)</li>
                <li><strong>Train/Test Split:</strong> 80/20</li>
                <li><strong>Seed:</strong> Fixed for reproducibility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    st.markdown("<h3 style='color: #764ba2;'>🎯 Business Problem & Solution</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h4>📌 Problem Statement</h4>
            <p>Telecom companies lose significant revenue when customers churn (cancel subscriptions). Early identification of at-risk customers enables proactive retention strategies.</p>
            <p><strong>Challenge:</strong> Predict which customers are likely to churn within a specific period, enabling targeted intervention.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>✨ Solution Impact</h4>
            <p>This ML model identifies at-risk customers with high accuracy, enabling:</p>
            <ul style='margin: 10px 0;'>
                <li>📱 Targeted retention campaigns</li>
                <li>💰 Reduced revenue loss</li>
                <li>📈 Improved customer lifetime value</li>
                <li>🎯 Data-driven retention strategies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    st.markdown("<h3 style='color: #764ba2;'>🔄 Data Processing Workflow</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea;'>
        <h4>Pipeline Stages:</h4>
        <ol>
            <li><strong>Data Loading:</strong> Read CSV with 7,043 customer records</li>
            <li><strong>Exploratory Analysis:</strong> Understand patterns and relationships (20+ visualizations)</li>
            <li><strong>Data Cleaning:</strong> Remove duplicates, handle missing values (7,032 → final dataset)</li>
            <li><strong>Feature Engineering:</strong> Create meaningful features through encoding and scaling</li>
            <li><strong>Model Selection:</strong> Compare Logistic Regression, Decision Tree, Random Forest, XGBoost</li>
            <li><strong>Hyperparameter Tuning:</strong> GridSearchCV optimization for best performance</li>
            <li><strong>Model Evaluation:</strong> Cross-validation, ROC curves, confusion matrices</li>
            <li><strong>Deployment:</strong> Serialized models for production predictions</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# ==================== TAB 4: ANALYTICS ====================
with tab4:
    st.markdown("<h2 style='color: #667eea;'>📈 Analytics, Insights & Recommendations</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='color: #764ba2;'>🔍 Key Findings from Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h4>📜 Contract Type Impact</h4>
            <p><strong>Finding:</strong> Month-to-month contracts have the highest churn rate.</p>
            <p><strong>Data:</strong> One and two-year contracts show 40-50% lower churn rates.</p>
            <p><strong>Action:</strong> Incentivize long-term contracts with discounts and benefits.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h4>⏱️ Tenure Effect</h4>
            <p><strong>Finding:</strong> New customers (0-6 months) churn 5-10x more than established customers.</p>
            <p><strong>Data:</strong> Customers with 24+ months tenure show <10% churn rate.</p>
            <p><strong>Action:</strong> Focus on first-year onboarding and satisfaction.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h4>🌐 Internet Service Quality</h4>
            <p><strong>Finding:</strong> Fiber optic customers churn more than DSL customers.</p>
            <p><strong>Insight:</strong> Likely quality or speed issues with fiber optic service.</p>
            <p><strong>Action:</strong> Investigate and improve fiber optic service quality immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h4>💰 Charges & Affordability</h4>
            <p><strong>Finding:</strong> High monthly charges correlate with increased churn.</p>
            <p><strong>Data:</strong> Customers paying >$100/month show significantly higher churn.</p>
            <p><strong>Action:</strong> Develop value-conscious packages and bundling strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h4>🔒 Support Services Loyalty</h4>
            <p><strong>Finding:</strong> Tech support & security subscribers churn much less.</p>
            <p><strong>Data:</strong> Customers with support services show 30-40% lower churn.</p>
            <p><strong>Action:</strong> Bundle and promote support services by default.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h4>💳 Payment Method Influence</h4>
            <p><strong>Finding:</strong> Electronic check payment correlates with higher churn.</p>
            <p><strong>Data:</strong> Automatic payments show better retention.</p>
            <p><strong>Action:</strong> Incentivize autopay enrollment with discounts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    st.markdown("<h3 style='color: #764ba2;'>🎯 Strategic Recommendations</h3>", unsafe_allow_html=True)
    
    tab_reco1, tab_reco2, tab_reco3 = st.tabs(["🎪 Marketing Strategy", "🏢 Operations Plan", "📱 Product Development"])
    
    with tab_reco1:
        st.markdown("""
        #### 1️⃣ **Targeted Retention Campaigns**
        - Identify at-risk customers using this model monthly
        - Launch personalized outreach campaigns
        - Offer loyalty discounts based on risk level
        - A/B test different retention messages
        
        #### 2️⃣ **Contract Migration Program**
        - Offer incentives for upgrading from month-to-month
        - 1-year contract: 10% discount
        - 2-year contract: 15% discount + extra benefits
        - Target high-risk month-to-month customers first
        
        #### 3️⃣ **Win-Back Campaigns**
        - Track churned customers
        - Launch "Come back" campaigns with special offers
        - Survey to understand churn reasons
        - Implement improvements based on feedback
        
        #### 4️⃣ **Loyalty Program Enhancement**
        - Reward long-term customer loyalty
        - Birthday/anniversary special offers
        - Referral incentives ($20-50 credits)
        - Tier-based benefits based on tenure
        """)
    
    with tab_reco2:
        st.markdown("""
        #### 1️⃣ **Service Quality Improvement**
        - Immediate: Audit fiber optic network quality
        - Deploy additional fiber optic infrastructure
        - Increase speed offerings in high-churn areas
        - Set up proactive monitoring and alerts
        
        #### 2️⃣ **Onboarding Excellence**
        - Redesign first-month customer experience
        - Assign onboarding specialists
        - Provide 24/7 setup support
        - Monitor satisfaction weekly in first month
        
        #### 3️⃣ **Support Service Expansion**
        - Increase tech support availability
        - Reduce response times to <30 minutes
        - Proactive outreach for service issues
        - Create VIP support tier for high-value customers
        
        #### 4️⃣ **Payment Processing**
        - Promote automatic payment enrollment
        - Simplify autopay setup (1-click)
        - Offer 5% discount for autopay
        - Reduce billing-related issues
        """)
    
    with tab_reco3:
        st.markdown("""
        #### 1️⃣ **Competitive Pricing**
        - Develop tier-based pricing: Budget, Standard, Premium
        - Bundle services at attractive price points
        - Offer value packages for different segments
        - Dynamic pricing based on tenure
        
        #### 2️⃣ **Service Bundling**
        - Create default bundles with tech support
        - Offer security packages at attractive rates
        - Bundle entertainment + internet services
        - Premium package for power users
        
        #### 3️⃣ **Speed & Performance**
        - Introduce higher-speed tiers
        - Unlimited data options
        - Faster uploads/downloads
        - Quality-of-service guarantees
        
        #### 4️⃣ **Loyalty Products**
        - Develop rewards program
        - Points-based benefits
        - Free service upgrades for long tenure
        - Exclusive perks for 3+ year customers
        """)
    
    st.write("")
    
    st.markdown("<h3 style='color: #764ba2;'>📊 Implementation Priority Matrix</h3>", unsafe_allow_html=True)
    
    priority_data = pd.DataFrame({
        'Initiative': [
            'Contract Migration Program',
            'Service Quality (Fiber Optic)',
            'Support Service Bundling',
            'Autopay Incentive Program',
            'Onboarding Improvement',
            'Loyalty Program',
            'Pricing Tiers',
            'Win-Back Campaigns'
        ],
        'Impact': [9, 8, 8, 7, 9, 6, 7, 5],
        'Effort': [6, 8, 5, 3, 7, 6, 8, 4],
        'Timeline (weeks)': [4, 12, 3, 2, 8, 6, 10, 3]
    })
    
    # Create bubble chart
    fig = go.Figure(data=[
        go.Scatter(
            x=priority_data['Effort'],
            y=priority_data['Impact'],
            mode='markers+text',
            text=priority_data['Initiative'],
            textposition='top center',
            marker=dict(
                size=priority_data['Timeline (weeks)'] * 2,
                color=priority_data['Impact'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Impact Level"),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Impact: %{y}<br>Effort: %{x}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Initiative Priority Matrix (size = timeline)",
        xaxis_title="Implementation Effort",
        yaxis_title="Business Impact",
        height=500,
        hovermode='closest',
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 10])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #667eea; margin-top: 20px;'>
        <h4>💡 Implementation Strategy:</h4>
        <ol>
            <li><strong>Week 1-2:</strong> Launch autopay incentive program (quick win)</li>
            <li><strong>Week 3-4:</strong> Start contract migration program</li>
            <li><strong>Week 5-8:</strong> Redesign onboarding process</li>
            <li><strong>Weeks 1-12 (ongoing):</strong> Address fiber optic service quality</li>
            <li><strong>Weeks 5-10:</strong> Develop pricing tiers and bundles</li>
            <li><strong>After Week 6:</strong> Launch loyalty program</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.write("")
st.markdown("""
    <div style='text-align: center; padding: 30px; background-color: #f8f9fa; border-radius: 10px; margin-top: 40px;'>
        <p style='color: #999; font-size: 14px;'>
        <strong>Telecom Churn Prediction Model v1.0</strong> | 
        Built with Python, XGBoost & Streamlit | 
        Data: IBM Telco Customer Churn Dataset
        </p>
        <p style='color: #999; font-size: 12px; margin-top: 10px;'>
        📧 Questions? 💡 Suggestions? | 
        Last Updated: 2024 | 
        Status: ✅ Production Ready
        </p>
    </div>
""", unsafe_allow_html=True)
