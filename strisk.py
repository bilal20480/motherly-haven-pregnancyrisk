import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# Load the trained model and label encoder
model = joblib.load('maternal_health_rf_model.pkl')
le = joblib.load('label_encoder.pkl')

# Configure Gemini API
genai.configure(api_key="AIzaSyCnIkKmtRYHwJOMGGa244-XdWXYtIR_RwE")  # Replace with your actual API key
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Set page config
st.set_page_config(page_title="Maternal Health Risk Assessment", page_icon="🤰")

# Title only - nothing else on main page before prediction
st.title("🤰 Maternal Health Risk Assessment")

# Sidebar with input form and predict button
with st.sidebar:
    st.header("Patient Health Parameters")
    
    with st.form("health_form"):
        age = st.number_input("Age (years)", min_value=10, max_value=70, value=25)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=160, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=100, value=80)
        bs = st.number_input("Blood Sugar (mmol/L)", min_value=6.0, max_value=19.0, value=7.0, step=0.1)
        body_temp = st.number_input("Body Temperature (°F)", min_value=98.0, max_value=103.0, value=98.0, step=0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=90, value=70)
        
        submitted = st.form_submit_button("Assess Risk")

# Only show content after prediction is submitted
if submitted:
    # Create input dataframe
    input_df = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    })
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    risk_level = le.inverse_transform(prediction)[0]
    
    # Display prediction results
    st.subheader("Assessment Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Risk Level", risk_level)
    
    with col2:
        # Get the probability for the predicted risk level
        proba = prediction_proba[0][prediction[0]]
        st.metric("Confidence", f"{proba:.1%}")
    
    # Display detailed probabilities
    with st.expander("Detailed Probabilities"):
        proba_df = pd.DataFrame({
            'Risk Level': le.classes_,
            'Probability': prediction_proba[0]
        })
        st.dataframe(proba_df.style.format({'Probability': '{:.1%}'}))
    
    # Generate recommendations using Gemini
    st.subheader("Personalized Recommendations")
    
    prompt = f"""
    A pregnant woman with the following health parameters:
    - Age: {age} years
    - Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg
    - Blood Sugar: {bs} mmol/L
    - Body Temperature: {body_temp}°F
    - Heart Rate: {heart_rate} bpm
    
    Has been assessed as having a {risk_level} risk level for maternal health complications.
    
   
Please give **exactly 10 clear and practical steps** she should take to **reduce her pregnancy risk**.

The list should include:
- Lifestyle changes
- Dietary recommendations
- Medical advice or checkups
- Things to avoid
- Any emergency precautions

Make sure:
- The tone is supportive, non-alarming
- The advice is easy to understand for a first-time mom
- Each point is short and actionable (not more than 2–3 lines)

Only return the numbered list. No intro or explanation.
    """
    
    with st.spinner("Generating recommendations..."):
        try:
            response = gemini_model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error("Failed to generate recommendations. Please try again later.")
            st.error(str(e))
    
    # Add disclaimer
    st.warning("""
    **Disclaimer**: This assessment is for informational purposes only and should not replace professional medical advice. 
    Always consult with a qualified healthcare provider for medical concerns.
    """)