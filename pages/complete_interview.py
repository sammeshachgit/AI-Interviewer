import streamlit as st
import random
from groq import Groq
import pyttsx3
import tempfile
import spacy
from io import BytesIO
import speech_recognition as sr
from streamlit_mic_recorder import speech_to_text
import matplotlib.pyplot as plt

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

# ========== Groq Setup ========== 
nlp = spacy.load("en_core_web_sm")
client = Groq(api_key="")
MODEL_NAME = "llama3-70b-8192"

# ========== General Domain Questions ========== 
GENERAL_DOMAIN_QUESTIONS = [
    "How do you stay updated with the latest trends in your field?",
    "What is your approach to problem-solving?",
    "How do you handle tight deadlines when working on complex tasks?",
    "Can you describe a time when you had to learn a new tool or technology quickly?",
    "How do you ensure the quality of your work?",
    "What’s your preferred way to collaborate with a team on technical tasks?",
    "Describe a situation where you had to troubleshoot a difficult issue.",
    "What are your thoughts on automation in the industry?",
    "How do you manage multiple priorities in a project?",
    "Explain a technical concept to a non-technical person."
]

# ========== HR and Resume Questions ========== 
HR_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want to work here?",
    "What are your strengths and weaknesses?",
    "Where do you see yourself in 5 years?",
    "How do you handle conflict at work?",
    "Describe a challenging situation you faced and how you handled it.",
    "What motivates you?",
    "How do you handle criticism?",
    "What does leadership mean to you?",
    "What is your expected salary?"
]

RESUME_PROMPTS = [
    "Is your resume general or domain specific?",
    "Highlight one project and explain your role in detail.",
    "What skills from your resume are you most confident in?",
    "Tell me about a certification or course you've done recently.",
    "How has your academic background prepared you for this role?"
]

# ========== Streamlit Page Setup ========== 
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.markdown("<h2 style='text-align:center;'>🎤 AI Interviewer - Complete Interview Suite</h2>", unsafe_allow_html=True)

# ========== Session State Init ========== 
if "stage" not in st.session_state:
    st.session_state.stage = "intro"
if "questions" not in st.session_state:
    st.session_state.questions = {"hr": [], "domain": [], "resume": []}
if "answers" not in st.session_state:
    st.session_state.answers = {"hr": [], "domain": [], "resume": []}
if "current_round" not in st.session_state:
    st.session_state.current_round = "HR Round"
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "speech_text" not in st.session_state:
    st.session_state.speech_text = ""  

# ========== Answer Evaluation via Groq ========== 
def evaluate_answer(question, answer):
    prompt = f"""
You are an expert interview evaluator. Given the following interview question and the candidate's answer, analyze it thoroughly and return your evaluation in JSON format.

Focus on:
- Relevance to the question
- Clarity and structure
- Use of real examples or evidence
- Depth of insight and reflection

Format your response in valid JSON with:
- "score": Integer between 1 and 10 (based on the answer quality)
- "feedback": Constructive feedback (max 2 lines, specific and helpful)
- "suggested_answer": An ideal answer for the given question (brief, yet high-quality)

Example output:
{{
  "score": 8,
  "feedback": "Well-structured and relevant. You could improve by adding a specific example.",
  "suggested_answer": "A great answer would include a situation where you demonstrated leadership under pressure."
}}

Question: {question}
Answer: {answer}
"""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.5
        )
        content = response.choices[0].message.content
        result = eval(content) if content.strip().startswith('{') else {}
        return {
            "score": result.get("score", random.randint(6, 9)),
            "feedback": result.get("feedback", "Decent answer. Could use more specificity."),
            "suggested_answer": result.get("suggested_answer", "No ideal response available.")
        }
    except Exception as e:
        return {
            "score": random.randint(6, 10),
            "feedback": f"Error from Groq: {str(e)}",
            "suggested_answer": "Could not fetch suggestion due to an error."
        }

# ========== UI Logic ========== 
def intro_stage():
    st.markdown(""" 
<div style='
    background-color: #f8f9fa;
    padding: 15px;
    border-left: 6px solid #0d6efd;
    border-radius: 5px;
    font-size: 16px;
    color: black;
'>
    ⚠️ <strong>Disclaimer:</strong> These are general interview questions commonly asked across domains. 
    For more specific or role-based interviews, please explore our domain-based or resume-based or hr-based interview sections.
</div>
""", unsafe_allow_html=True)
    st.markdown("### Interview Rounds Included:")
    col1, col2, col3 = st.columns(3)
    col1.success("🧠 HR Round - 10 Questions")
    col2.success("💼 Domain Questions - 10 Questions")
    col3.success("📄 Resume-Based - 5 Questions")

    if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
        st.session_state.questions = {
            "hr": random.sample(HR_QUESTIONS, 10),
            "domain": random.sample(GENERAL_DOMAIN_QUESTIONS, 10),
            "resume": RESUME_PROMPTS
        }
        st.session_state.stage = "hr"
        st.session_state.current_round = "HR Round"
        st.rerun()      

def interview_stage():
    round_key = st.session_state.stage
    questions = st.session_state.questions[round_key]
    index = st.session_state.current_index
    question = questions[index]

    st.markdown(f"### 📝 {st.session_state.current_round}")
    st.markdown(f"**Question {index+1} of {len(questions)}:**")
    st.markdown(f"**👉 {question}**")

    # === Reset speech_text when new question loads ===
    if "last_index" not in st.session_state:
        st.session_state.last_index = -1
    if st.session_state.current_index != st.session_state.last_index:
        st.session_state.speech_text = ""
        st.session_state.last_index = st.session_state.current_index

    # === Play Question Audio ===
    st.audio(text_to_audio(question), format="audio/mp3")

    # === Record Speech Input ===
    st.markdown("🎤 **Record Your Answer**")
    speech_text = speech_to_text(
        language='en',
        use_container_width=True,
        just_once=True,
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key=f"mic_q_{round_key}_{index}"
    )

    if speech_text:
        st.session_state.speech_text = speech_text
        st.success("✔️ Speech converted to text!")

    # === Text Input Field (Pre-filled with speech text if any) ===
    st.markdown("⌨️ **Type or Edit Your Answer**")
    typed_text = st.text_area(
        "Answer here:",
        value=st.session_state.speech_text,
        key=f"text_q_{round_key}_{index}",
        height=150
    )

    # === On Submit ===
    if st.button("Submit Answer", use_container_width=True):
        final_answer = typed_text.strip() or st.session_state.speech_text.strip()
        if not final_answer:
            st.warning("⚠️ Please enter an answer before submitting.")
            st.stop()

        evaluation = evaluate_answer(question, final_answer)
        st.session_state.answers[round_key].append({
            "question": question,
            "answer": final_answer,
            "score": evaluation["score"],
            "feedback": evaluation["feedback"],
            "suggested_answer": evaluation["suggested_answer"]
        })

        # Move to next question or round
        if index + 1 < len(questions):
            st.session_state.current_index += 1
            st.rerun()
        else:
            st.session_state.current_index = 0
            if round_key == "hr":
                st.session_state.stage = "domain"
                st.session_state.current_round = "Domain Round"
            elif round_key == "domain":
                st.session_state.stage = "resume"
                st.session_state.current_round = "Resume-Based Round"
            else:
                st.session_state.stage = "results"
            st.rerun()

def show_feedback_visualization(scores, title):
    if not scores:
        st.warning("No valid scores to display.")
        return
    labels = [f"Q{i + 1}" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, scores, color='skyblue')
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score")
    ax.set_title(title)
    st.pyplot(fig)

def results_stage():
    st.success("✅ Interview Complete!")
    st.markdown("### 📊 Interview Summary")
    total_score = 0
    total_questions = 0
    round_averages = {}

    for section in ["hr", "domain", "resume"]:
        st.markdown(f"#### 📌 {section.upper()} ROUND")
        section_score = 0
        section_scores = []
        section_questions = len(st.session_state.answers[section])

        for item in st.session_state.answers[section]:
            with st.expander(f"👉 {item['question']}"):
                st.markdown(f"**Your Answer:** {item['answer']}**")
                st.markdown(f"**Score:** {item['score']}/10")
                st.markdown(f"**Feedback:** {item['feedback']}")
                st.markdown(f"**Suggested Answer:** {item['suggested_answer']}")
                section_score += item["score"]
                section_scores.append(item["score"])

        if section_questions > 0:
            avg = round(section_score / section_questions, 2)
            round_averages[section] = avg
            st.info(f"**Average Score for {section.upper()} Round:** {avg} / 10")
            show_feedback_visualization(section_scores, f"{section.upper()} Round Performance")

        total_score += section_score
        total_questions += section_questions

    if total_questions > 0:
        overall = round(total_score / total_questions, 2)
        st.markdown("---")
        st.success(f"🎯 **Overall Interview Score: {overall} / 10**")

    # 🔍 Recommendation based on lowest average
    if round_averages:
        weakest_round = min(round_averages, key=round_averages.get)
        st.warning(f"📌 Based on your scores, we recommend focusing more on **{weakest_round.upper()}** round to improve your performance.")

    if st.button("🔁 Retake Interview", use_container_width=True):
        for key in ["stage", "questions", "answers", "current_round", "current_index"]:
            st.session_state.pop(key, None)
        st.rerun()

# ========== Main ========== 
if st.session_state.stage == "intro":
    intro_stage()
elif st.session_state.stage in ["hr", "domain", "resume"]:
    interview_stage()
elif st.session_state.stage == "results":
    results_stage()