import streamlit as st
import pandas as pd
import spacy
import re
import matplotlib.pyplot as plt
from groq import Groq
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from io import BytesIO

def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


# ------------------ Config ------------------
# st.set_page_config(page_title="💼 HR Interview", layout="centered")
st.title("💼 HR-Based Interview")

nlp = spacy.load("en_core_web_sm")
client = Groq(api_key="")  # Replace with your actual key

# ------------------ Load Questions ------------------
@st.cache_data
def load_hr_questions(file):
    df = pd.read_csv(file)
    return df["Question"].dropna().tolist()

all_questions = load_hr_questions("hr_questions.csv")

# ------------------ Question Selection ------------------
num_questions = st.slider("How many HR questions would you like to answer?", 1, len(all_questions), 5)
if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = all_questions[:num_questions]

# ------------------ Initialize Session State ------------------
if "hr_q_idx" not in st.session_state:
    st.session_state.hr_q_idx = 0
    st.session_state.hr_answers = []
    st.session_state.hr_feedback = []
    st.session_state.speech_text = ""

questions = st.session_state.selected_questions

# ------------------ Interview Flow ------------------
if st.session_state.hr_q_idx < num_questions:
    q = questions[st.session_state.hr_q_idx]
    st.subheader(f"🧠 Question {st.session_state.hr_q_idx + 1}")
    st.info(q)
    st.audio(text_to_audio(q), format="audio/mp3")

    # Mic input
    st.markdown("🎤 **Record Your Answer**")
    speech_text = speech_to_text(
        language='en',
        use_container_width=True,
        just_once=True,
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key=f"mic_q{st.session_state.hr_q_idx}"
    )

    if speech_text:
        st.session_state.speech_text = speech_text
        st.success("✔️ Speech converted to text!")

    # Text input
    st.markdown("⌨️ **Type or Edit Your Answer**")
    typed_text = st.text_area(
        "Answer here:",
        value=st.session_state.speech_text,
        key=f"text_q{st.session_state.hr_q_idx}",
        height=150
    )

    final_answer = speech_text if speech_text else typed_text

    if st.button("✅ Submit Answer"):
        st.session_state.hr_answers.append(final_answer)
        st.session_state.hr_q_idx += 1
        st.session_state.speech_text = ""
        st.rerun()

# ------------------ Feedback Generation ------------------
elif len(st.session_state.hr_answers) == num_questions:

    st.success("🎉 Interview Completed! Generating Feedback...")

    # ----------- Feedback System -----------
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

                score_match = re.search(r"Score:\s*(\d+)", result)
                score = int(score_match.group(1)) if score_match else 0
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

    # ----------- Visualization -----------
    def show_feedback_visualization(scores):
        scores = [s for s in scores if s > 0]  # Filter out zero/invalid scores
        if not scores:
            st.warning("No valid scores to display in chart.")
            return
        labels = [f"Q{i + 1}" for i in range(len(scores))]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
        ax.set_title("📊 Performance Distribution Across Questions")
        st.pyplot(fig)


    with st.spinner("🤖 Analyzing your answers..."):
        feedback_data, score_data = generate_feedback(questions, st.session_state.hr_answers)

    st.subheader("📝 Feedback Summary")
    if score_data:
        total_score = sum(score_data)
        max_score = len(score_data) * 10
        st.markdown(f"### 🏁 Total Score: **{total_score / max_score * 10:.2f} / 10**")
    else:
        st.warning("No valid scores were generated.")


    for fb in feedback_data:
        st.markdown(f"**❓ Question:** {fb['question']}")
        st.markdown(f"**✅ Your Answer:** {fb['answer']}")
        st.markdown(f"**🔢 Score:** {fb['score']}/10")
        st.markdown(f"**💬 Feedback:** {fb['feedback']}")
        if fb['suggested_answer']:
            st.markdown(f"**💡 Suggested Answer:** {fb['suggested_answer']}")
        st.markdown("---")

    # Chart
    show_feedback_visualization(score_data)

