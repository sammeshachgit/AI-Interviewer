import streamlit as st
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
from groq import Groq
from streamlit_mic_recorder import speech_to_text
from io import BytesIO
import pyttsx3
import tempfile

# ------------------ Config ------------------
# st.set_page_config(page_title="🧑‍💻 Domain-Based Interview", layout="wide")
st.title("🧑‍💻 Domain-Based Interview")

client = Groq(api_key="")

# ----------- Text to Audio ----------- 

def text_to_audio(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        filename = fp.name
    engine.save_to_file(text, filename)
    engine.runAndWait()
    
    with open(filename, 'rb') as f:
        audio_bytes = f.read()
    
    return BytesIO(audio_bytes)


# ------------------ Load Questions (Fixed CSV Handling) ------------------
@st.cache_data
def load_questions():
    try:
        return pd.read_csv("interview_questions.csv", quotechar='"', escapechar='\\')
    except pd.errors.ParserError:
        st.error("⚠️ CSV format error. Ensure questions with commas are wrapped in quotes.")
        st.stop()

df = load_questions()
domains = df['Domain'].unique()

# ------------------ Domain Selection ------------------
selected_domain = st.selectbox("Choose a domain", options=domains)
num_questions = st.slider("How many questions do you want to answer?", 1, 10, 5)

# Initialize session state
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
    st.session_state.domain_qs = []
    st.session_state.q_idx = 0
    st.session_state.answers = []
    st.session_state.feedback = []
    st.session_state.speech_text = ""

# Start interview button
if not st.session_state.interview_started:
    if st.button("🚀 Start Interview"):
        domain_questions = df[df['Domain'] == selected_domain]['Questions'].dropna().tolist()
        st.session_state.domain_qs = random.sample(domain_questions, min(num_questions, len(domain_questions)))
        st.session_state.interview_started = True
        st.session_state.q_idx = 0
        st.session_state.answers = []
        st.rerun()

# ------------------ Q&A Flow ------------------
if st.session_state.interview_started:
    questions = st.session_state.domain_qs
    
    if st.session_state.q_idx < len(questions):
        q = questions[st.session_state.q_idx]
        st.subheader(f"🧠 Question {st.session_state.q_idx + 1} of {len(questions)}")
        st.info(q)
        st.audio(text_to_audio(q), format="audio/mp3", start_time=0)

        # Mic input
        st.markdown("🎤 **Record Your Answer**")
        speech_text = speech_to_text(
            language='en',
            use_container_width=True,
            just_once=True,
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key=f"mic_q{st.session_state.q_idx}"
        )

        if speech_text:
            st.session_state.speech_text = speech_text
            st.success("✔️ Speech converted to text!")

        # Text input
        st.markdown("⌨️ **Type or Edit Your Answer**")
        typed_text = st.text_area(
            "Answer here:",
            value=st.session_state.speech_text,
            key=f"text_q{st.session_state.q_idx}",
            height=150
        )

        final_answer = speech_text if speech_text else typed_text

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏭️ Skip Question"):
                st.session_state.answers.append("(Skipped)")
                st.session_state.q_idx += 1
                st.session_state.speech_text = ""
                st.rerun()
        with col2:
            if st.button("✅ Submit Answer"):
                st.session_state.answers.append(final_answer)
                st.session_state.q_idx += 1
                st.session_state.speech_text = ""
                st.rerun()

    # ------------------ Feedback Generation ------------------
    elif len(st.session_state.answers) == len(questions):
        st.success("🎉 Interview Completed! Generating Feedback...")

        def generate_feedback(questions, answers):
            feedback = []
            scores = []

            for i, (q, a) in enumerate(zip(questions, answers)):
                prompt = f"""
                Interview Question: {q}
                Candidate Answer: {a}

                Please provide:
                1. A score from 1-10 (just the number)
                2. Detailed feedback on what was good and what could be improved
                3. A suggested better answer

                Format your response as:
                Score: [number]
                Feedback: [text]
                Suggested Answer: [text]
                """
                try:
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5
                    )
                    result = response.choices[0].message.content

                    score = int(re.search(r"Score:\s*(\d+)", result).group(1))
                    feedback_text = re.search(r"Feedback:\s*(.+?)(?=\nSuggested Answer:|$)", result, re.DOTALL).group(1).strip()
                    suggested_answer = re.search(r"Suggested Answer:\s*(.+)", result, re.DOTALL).group(1).strip() if "Suggested Answer:" in result else ""

                    feedback.append({
                        "question": q,
                        "answer": a,
                        "score": score,
                        "feedback": feedback_text,
                        "suggested_answer": suggested_answer
                    })
                    scores.append(score)
                except Exception as e:
                    st.error(f"⚠️ Error analyzing answer {i + 1}: {e}")
                    feedback.append({
                        "question": q,
                        "answer": a,
                        "score": 0,
                        "feedback": "Could not analyze this answer due to an error.",
                        "suggested_answer": ""
                    })
                    scores.append(0)

            return feedback, scores

        def show_feedback_visualization(scores):
            # Filter out any NaN values and replace them with 0
            clean_scores = [0 if pd.isna(score) else score for score in scores]
            labels = [f"Q{i + 1}" for i in range(len(clean_scores))]
            
            # Create a figure with a single subplot
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Only create pie chart if we have valid scores
            if any(score > 0 for score in clean_scores):
                ax.pie(clean_scores, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
                ax.set_title("📊 Performance Distribution Across Questions")
                st.pyplot(fig)
            else:
                st.warning("No valid scores available for visualization")

        with st.spinner("🤖 Analyzing your answers..."):
            feedback_data, score_data = generate_feedback(questions, st.session_state.answers)

        st.subheader("📝 Feedback Summary")
        total_score = sum(s for s in score_data if not pd.isna(s))
        max_score = len([s for s in score_data if not pd.isna(s)]) * 10
        
        if max_score > 0:
            final_score = total_score / max_score * 10
        else:
            final_score = 0
            
        st.markdown(f"### 🏁 Total Score: **{final_score:.2f} / 10**")

        for fb in feedback_data:
            with st.expander(f"Question {fb['question'][:30]}... (Score: {fb['score']}/10)"):
                st.markdown(f"**❓ Question:** {fb['question']}")
                st.markdown(f"**✅ Your Answer:** {fb['answer']}")
                st.markdown(f"**💬 Feedback:** {fb['feedback']}")
                if fb['suggested_answer']:
                    st.markdown(f"**💡 Suggested Answer:** {fb['suggested_answer']}")

        show_feedback_visualization(score_data)

        if st.button("🔄 Start New Interview"):
            st.session_state.interview_started = False
            st.session_state.domain_qs = []
            st.rerun()