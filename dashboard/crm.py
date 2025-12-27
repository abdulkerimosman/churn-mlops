import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
import json
import sys
from datetime import datetime

# Add parent directory to path to import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import database

# Configuration
API_URL = os.getenv("API_URL")
if not API_URL:
    try:
        API_URL = st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = "http://localhost:8000"
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Initialize & Seed DB (Cached to prevent re-running on every interaction)
@st.cache_resource(show_spinner=False)
def init_app_db():
    database.init_db()
    database.seed_database(RAW_DATA_PATH)

init_app_db()

@st.cache_data
def get_customer_data():
    return database.load_customers()

@st.cache_data
def get_high_risk_data():
    return database.get_high_risk_customers()

st.set_page_config(
    page_title="ChurnGuard CRM",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([9, 1])
with col1:
    st.title("🛡️ ChurnGuard CRM")
    st.markdown("### Customer Retention & Risk Analysis System")
    try:
        health = requests.get(f"{API_URL}/health", timeout=1)
        if health.status_code == 200:
            st.caption(f"🟢 System Online | API: `{API_URL}`")
        else:
            st.caption(f"🔴 System Offline | API: `{API_URL}`")
    except:
        st.caption(f"🔴 System Offline | API: `{API_URL}`")

with col2:
    st.image("https://img.icons8.com/fluency/96/user-group-man-woman.png", width=80)

@st.dialog("Customer Action Plan", width="large")
def show_customer_details(row):
    # Header
    st.subheader(f"{row.get('customer_name', 'Unknown Customer')} (ID: {row['customerID']})")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        # Contact & Profile Combined
        st.markdown("#### 👤 Profile & Contact")
        st.markdown(f"""
        **Email:** {row.get('contact_email', 'N/A')}  
        **Phone:** {row.get('contact_phone', 'N/A')}
        """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Contract:**\n{row['Contract']}")
            st.markdown(f"**Tenure:**\n{row['tenure']} months")
        with c2:
            st.markdown(f"**Monthly:**\n${row['MonthlyCharges']}")
            st.markdown(f"**Internet:**\n{row['InternetService']}")

        st.divider()
        
        # Recommended Actions
        st.markdown("#### 💡 Recommended Actions")
        actions = []
        if row['Contract'] == 'Month-to-month':
            actions.append("• Offer 1-Year Contract (20% off)")
        if row['MonthlyCharges'] > 80:
            actions.append("• Review Pricing Plan / Offer Discount")
        if row['TechSupport'] == 'No':
            actions.append("• Offer Free Tech Support Add-on")
        if row['InternetService'] == 'Fiber optic':
            actions.append("• Check Service Quality")
            
        if actions:
            for action in actions:
                st.markdown(action)
        else:
            st.info("No specific actions recommended.")

    with col_right:
        # Status Update
        st.markdown("#### 📝 Status Update")
        
        with st.form(key=f"form_{row['customerID']}"):
            new_status = st.selectbox(
                "Status",
                ["Pending", "Contacted", "Resolved"],
                index=["Pending", "Contacted", "Resolved"].index(row['follow_up_status'])
            )
            
            notes = st.text_area(
                "Notes", 
                value=row.get('notes', ''),
                height=150
            )
            
            if st.form_submit_button("Save Updates", type="primary", use_container_width=True):
                database.update_customer_status(row['customerID'], new_status, notes)
                get_customer_data.clear()
                get_high_risk_data.clear()
                st.success("Updated!")
                st.rerun()

# Main Content
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")

if 'page' not in st.session_state:
    st.session_state.page = "🏢 Existing Customer Base"

with st.sidebar:
    st.markdown("### Navigation")
    if st.button("🏢 Existing Customer Base", use_container_width=True, type="primary" if st.session_state.page == "🏢 Existing Customer Base" else "secondary"):
        st.session_state.page = "🏢 Existing Customer Base"
        st.rerun()
    
    if st.button("🧪 New Data Analysis", use_container_width=True, type="primary" if st.session_state.page == "🧪 New Data Analysis" else "secondary"):
        st.session_state.page = "🧪 New Data Analysis"
        st.rerun()

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        get_customer_data.clear()
        get_high_risk_data.clear()
        st.rerun()

page = st.session_state.page

# --- PAGE 1: EXISTING CUSTOMER BASE ---
if page == "🏢 Existing Customer Base":
    st.markdown("### Current Customer Database")
    
    # Load data from DB
    db_df = get_customer_data()
    
    if db_df.empty:
        st.warning("Database is empty. Please check data source.")
    else:
        # Top Stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", len(db_df))
        
        # Calculate stats if predictions exist
        if 'churn_probability' in db_df.columns and db_df['churn_probability'].notna().any():
            avg_churn = db_df['churn_probability'].mean()
            high_risk = len(db_df[db_df['risk_level'] == 'High'])
            churn_rate = (db_df['churn_prediction'] == 1).mean()
            
            m2.metric("Avg Churn Probability", f"{avg_churn:.1%}")
            m3.metric("High Risk Customers", high_risk)
            m4.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
            
            # Show last analysis date if available
            if 'prediction_date' in db_df.columns:
                last_date = db_df['prediction_date'].max()
                if pd.notna(last_date):
                    st.caption(f"Last Analysis: {last_date}")
        else:
            m2.metric("Avg Churn Probability", "N/A")
            m3.metric("High Risk Customers", "N/A")
            m4.metric("Predicted Churn Rate", "N/A")

        # Action Bar
        col_act1, col_act2 = st.columns([1, 3])
        with col_act1:
            if st.button("🔄 Run Churn Analysis on Database", type="primary"):
                with st.spinner("Analyzing all customers in database..."):
                    try:
                        # Prepare payload
                        customers = []
                        for _, row in db_df.iterrows():
                            # Handle TotalCharges
                            total_charges = row['TotalCharges']
                            if pd.isna(total_charges) or total_charges == '':
                                total_charges = 0.0
                            else:
                                try:
                                    total_charges = float(total_charges)
                                except:
                                    total_charges = 0.0

                            customer = {
                                "customer_id": str(row['customerID']),
                                "gender": str(row['gender']),
                                "SeniorCitizen": int(row['SeniorCitizen']),
                                "Partner": str(row['Partner']),
                                "Dependents": str(row['Dependents']),
                                "tenure": int(row['tenure']),
                                "PhoneService": str(row['PhoneService']),
                                "MultipleLines": str(row['MultipleLines']),
                                "InternetService": str(row['InternetService']),
                                "OnlineSecurity": str(row['OnlineSecurity']),
                                "OnlineBackup": str(row['OnlineBackup']),
                                "DeviceProtection": str(row['DeviceProtection']),
                                "TechSupport": str(row['TechSupport']),
                                "StreamingTV": str(row['StreamingTV']),
                                "StreamingMovies": str(row['StreamingMovies']),
                                "Contract": str(row['Contract']),
                                "PaperlessBilling": str(row['PaperlessBilling']),
                                "PaymentMethod": str(row['PaymentMethod']),
                                "MonthlyCharges": float(row['MonthlyCharges']),
                                "TotalCharges": float(total_charges)
                            }
                            customers.append(customer)
                        
                        # Batch predict (chunking if necessary, but 200 is fine)
                        response = requests.post(f"{API_URL}/predict/batch", json=customers)
                        
                        if response.status_code == 200:
                            results = response.json()
                            predictions = results['predictions']
                            
                            # Update DB
                            # Map predictions back to DF
                            pred_map = {p['customer_id']: p for p in predictions}
                            
                            db_df['churn_probability'] = db_df['customerID'].map(lambda x: pred_map.get(x, {}).get('churn_probability'))
                            db_df['churn_prediction'] = db_df['customerID'].map(lambda x: pred_map.get(x, {}).get('churn_prediction'))
                            db_df['confidence'] = db_df['customerID'].map(lambda x: pred_map.get(x, {}).get('confidence'))
                            
                            db_df['risk_level'] = db_df['churn_probability'].apply(
                                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low' if pd.notna(x) else 'Unknown'
                            )
                            
                            db_df['prediction_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")

                            database.save_predictions(db_df)
                            get_customer_data.clear()
                            get_high_risk_data.clear()
                            st.success("Analysis complete! Database updated.")
                            st.rerun()
                        else:
                            st.error(f"Analysis failed: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error running analysis: {str(e)}")

        st.divider()
        
        # Intervention Dashboard (High Risk)
        st.subheader("🚨 High Risk Intervention Dashboard")
        
        high_risk_df = get_high_risk_data()
        
        if high_risk_df.empty:
            st.info("No high risk customers found. Run analysis to identify risks.")
        else:
            # Filters
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                status_filter = st.multiselect(
                    "Filter by Status", 
                    ["Pending", "Contacted", "Resolved"],
                    default=["Pending"]
                )
            
            if status_filter:
                filtered_df = high_risk_df[high_risk_df['follow_up_status'].isin(status_filter)]
            else:
                filtered_df = high_risk_df
                
            st.write(f"Showing {len(filtered_df)} customers requiring attention.")
            
            # Grid Layout
            cols = st.columns(3)
            for i, (_, row) in enumerate(filtered_df.iterrows()):
                col = cols[i % 3]
                with col:
                    # Card Container
                    with st.container(border=True):
                        # Avatar & Header
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="flex-shrink: 0;">
                                <img src="https://img.icons8.com/fluency-systems-filled/96/user.png" style="border-radius: 50%; background-color: #e0e0e0; padding: 10px; width: 60px; height: 60px;">
                            </div>
                            <div style="margin-left: 15px;">
                                <div style="font-size: 18px; font-weight: bold; color: #31333F;">{row.get('customer_name', 'Unknown Customer')}</div>
                                <div style="font-size: 14px; color: #808495;">ID: {row['customerID']}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Stats
                        st.metric("Churn Risk", f"{row['churn_probability']:.1%}", delta_color="inverse")
                        st.write(f"**Status:** `{row['follow_up_status']}`")
                        
                        # Actions
                        if st.button("Take Action", key=f"btn_view_{row['customerID']}", width="stretch"):
                            show_customer_details(row)

# --- PAGE 2: NEW DATA ANALYSIS ---
if page == "🧪 New Data Analysis":
    subtab1, subtab2 = st.tabs(["👤 Single Customer", "📂 Batch Upload"])
    
    # Single Customer Analysis
    with subtab1:
        st.markdown("#### Real-time Churn Prediction")
        
        with st.form("customer_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior = st.checkbox("Senior Citizen")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            with col2:
                st.subheader("Basic Services")
                tenure = st.slider("Tenure (Months)", 0, 72, 24)
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            
            with col3:
                st.subheader("Additional Services")
                multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            
            with col4:
                st.subheader("Financials")
                payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
                total = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
                paperless = st.checkbox("Paperless Billing", value=True)

            submitted = st.form_submit_button("Analyze Customer Risk", width="stretch", type="primary")

        if submitted:
            # Construct Payload
            payload = {
                "customer_id": "CRM-USER-001",
                "gender": gender,
                "SeniorCitizen": 1 if senior else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": "Yes" if paperless else "No",
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total
            }
            
            try:
                with st.spinner("Analyzing customer profile..."):
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    
                if response.status_code == 200:
                    result = response.json()
                    prob = result["churn_probability"]
                    is_churn = result["churn_prediction"] == 1
                    
                    # Result Display
                    st.divider()
                    r_col1, r_col2 = st.columns([1, 2])
                    
                    with r_col1:
                        st.metric(
                            "Churn Probability", 
                            f"{prob:.1%}", 
                            delta=f"{'High Risk' if is_churn else 'Safe'}",
                            delta_color="inverse"
                        )
                    
                    with r_col2:
                        if is_churn:
                            st.error("⚠️ **High Risk Alert**: This customer is likely to churn.")
                            st.markdown("**Recommended Actions:**")
                            st.markdown("- Offer 15% discount on next month")
                            st.markdown("- Schedule check-in call with Success Manager")
                        else:
                            st.success("✅ **Low Risk**: Customer appears stable.")
                            st.markdown("**Recommended Actions:**")
                            st.markdown("- Upsell 'Device Protection' package")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

    # Batch Upload
    with subtab2:
        st.markdown("#### Batch File Analysis")
        st.info("Upload a CSV file to analyze new customers not in the database.")
        
        with st.expander("📋 CSV Format Instructions"):
            st.markdown("""
            **Required Columns:**
            - `customerID`: Unique identifier
            - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
            - `tenure` (months), `PhoneService`, `MultipleLines`
            - `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
            - `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
            - `MonthlyCharges`, `TotalCharges`
            
            **Note:** Ensure headers match exactly.
            """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
                
            st.write(f"Loaded {len(df)} customers.")
            st.dataframe(df.head())
            
            if st.button("Analyze Uploaded Batch"):
                with st.spinner("Analyzing customers..."):
                    try:
                        # Prepare payload
                        customers = []
                        for _, row in df.iterrows():
                            # Handle TotalCharges conversion if needed
                            total_charges = row['TotalCharges']
                            if isinstance(total_charges, str):
                                try:
                                    total_charges = float(total_charges.strip())
                                except ValueError:
                                    total_charges = 0.0
                            elif pd.isna(total_charges):
                                total_charges = 0.0
                            
                            customer = {
                                "customer_id": str(row['customerID']),
                                "gender": str(row['gender']),
                                "SeniorCitizen": int(row['SeniorCitizen']),
                                "Partner": str(row['Partner']),
                                "Dependents": str(row['Dependents']),
                                "tenure": int(row['tenure']),
                                "PhoneService": str(row['PhoneService']),
                                "MultipleLines": str(row['MultipleLines']),
                                "InternetService": str(row['InternetService']),
                                "OnlineSecurity": str(row['OnlineSecurity']),
                                "OnlineBackup": str(row['OnlineBackup']),
                                "DeviceProtection": str(row['DeviceProtection']),
                                "TechSupport": str(row['TechSupport']),
                                "StreamingTV": str(row['StreamingTV']),
                                "StreamingMovies": str(row['StreamingMovies']),
                                "Contract": str(row['Contract']),
                                "PaperlessBilling": str(row['PaperlessBilling']),
                                "PaymentMethod": str(row['PaymentMethod']),
                                "MonthlyCharges": float(row['MonthlyCharges']),
                                "TotalCharges": float(total_charges)
                            }
                            customers.append(customer)
                        
                        # Send batch request
                        response = requests.post(f"{API_URL}/predict/batch", json=customers)
                        
                        if response.status_code == 200:
                            results = response.json()
                            predictions = results['predictions']
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(predictions)
                            
                            # Display summary metrics
                            col1, col2, col3 = st.columns(3)
                            avg_churn = results['average_churn_probability']
                            high_risk_count = sum(1 for p in predictions if p['churn_probability'] > 0.7)
                            
                            col1.metric("Average Churn Probability", f"{avg_churn:.1%}")
                            col2.metric("High Risk Customers", high_risk_count)
                            col3.metric("Total Processed", results['total_customers'])
                            
                            # Display detailed results
                            st.subheader("Detailed Analysis")
                            
                            # Prepare display dataframe
                            display_df = results_df.copy()
                            
                            # Add context from input data if available
                            if 'customerID' in df.columns:
                                cols_to_merge = ['customerID', 'Contract', 'MonthlyCharges']
                                cols_to_merge = [c for c in cols_to_merge if c in df.columns]
                                if cols_to_merge:
                                    display_df = display_df.merge(
                                        df[cols_to_merge], 
                                        left_on='customer_id', 
                                        right_on='customerID', 
                                        how='left'
                                    )
                                    if 'customerID' in display_df.columns:
                                        display_df = display_df.drop(columns=['customerID'])

                            # Add Risk Level
                            display_df['Risk Level'] = display_df['churn_probability'].apply(
                                lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                            )
                            
                            # Map prediction to text
                            display_df['churn_prediction'] = display_df['churn_prediction'].map({0: 'Retain', 1: 'Churn'})
                            
                            # Reorder columns
                            cols = ['customer_id', 'Risk Level', 'churn_probability', 'churn_prediction', 'confidence']
                            extra_cols = [c for c in display_df.columns if c not in cols]
                            final_cols = cols + extra_cols
                            display_df = display_df[final_cols]

                            # Configure columns
                            column_config = {
                                "customer_id": st.column_config.TextColumn("Customer ID"),
                                "Risk Level": st.column_config.TextColumn("Risk Level"),
                                "churn_probability": st.column_config.ProgressColumn(
                                    "Churn Probability",
                                    format="%.2f",
                                    min_value=0,
                                    max_value=1,
                                ),
                                "churn_prediction": st.column_config.TextColumn("Prediction"),
                                "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                                "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
                            }
                            
                            # Color code risk
                            def color_risk(val):
                                colors = {'High': '#ff4b4b', 'Medium': '#ffa421', 'Low': '#21c354'}
                                return f'color: {colors.get(val, "black")}'
                                
                            st.dataframe(
                                display_df.style.map(color_risk, subset=['Risk Level']),
                                column_config=column_config,
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Download button
                            csv = display_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results",
                                csv,
                                "churn_predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                        else:
                            st.error(f"Error: {response.text}")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")