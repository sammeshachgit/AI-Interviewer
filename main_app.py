import streamlit as st

st.set_page_config(page_title="AI Interviewer", layout="centered")

st.title("🤖 AI Interviewer")
st.markdown("Welcome! Choose which type of interview you want to attend:")

# Option Selection
option = st.selectbox(
    "🧠 Choose Interview Type:",
    [
        "Select...",
        "🧑‍💻 Domain-Based Interview",
        "📄 Resume-Based Interview",
        "💼 HR-Based Interview",
        "🏆 Complete Interview Suite"
    ]
)

# Route to respective apps
if option == "🧑‍💻 Domain-Based Interview":
    st.markdown("### 🔁 Redirecting to Domain-Based Interview...")
    st.switch_page("pages/Interview.py")

elif option == "📄 Resume-Based Interview":
    st.markdown("### 🔁 Redirecting to Resume-Based Interview...")
    st.switch_page("pages/resume_interviewer.py")

elif option == "💼 HR-Based Interview":
    st.markdown("### 🔁 Redirecting to Resume-Based Interview...")
    st.switch_page("pages/hr_interview.py")

elif option == "🏆 Complete Interview Suite":
    st.markdown("### 🔁 Redirecting to Complete Interview Suite...")
    st.switch_page("pages/complete_interview.py")