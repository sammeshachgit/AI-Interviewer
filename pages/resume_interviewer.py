import streamlit as st
import docx
import spacy
import re
import matplotlib.pyplot as plt
from groq import Groq
from pathlib import Path
from io import StringIO
from pdfminer.high_level import extract_text
from fpdf import FPDF
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from io import BytesIO
import pyttsx3
import tempfile

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

# ----------- Setup ----------- 
st.set_page_config(page_title="📄 AI Resume-Based Interview", layout="wide")
st.title("📄 AI Resume-Based Interview")
nlp = spacy.load("en_core_web_sm")
client = Groq(api_key="")  # Replace with your key

# ----------- Resume Processing Functions -----------
def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_resume_text(uploaded_file):
    ext = Path(uploaded_file.name).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(uploaded_file)
    else:
        return None

def extract_info(text):
    doc_top = nlp("\n".join(text.splitlines()[:10]))
    extracted = {
        "name": next((ent.text.strip() for ent in doc_top.ents if ent.label_ == "PERSON"), ""),
        "email": re.search(r"[\w.-]+@[\w.-]+", text).group(0) if re.search(r"[\w.-]+@[\w.-]+", text) else "",
        "phone": re.search(r"\b\d{10}\b", text).group(0) if re.search(r"\b\d{10}\b", text) else "",
        "education": [], "skills": [], "experience": [], "projects": [], 
        "certifications": [], "internships": [], "achievements": [],
        "github": re.search(r"https://github\.com/[^\s]+", text).group(0) if re.search(r"https://github\.com/[^\s]+", text) else "",
        "linkedin": re.search(r"https://www\.linkedin\.com/[^\s]+", text).group(0) if re.search(r"https://www\.linkedin\.com/[^\s]+", text) else ""
    }

    blocks = {
        "education": ["education", "degree", "university", "college"],
        "experience": ["experience", "work", "employment", "professional"],
        "projects": ["project", "portfolio"],
        "certifications": ["certified", "certification", "nptel", "udemy", "coursera"],
        "internships": ["intern", "internship", "training"],
        "achievements": ["achievement", "hackathon", "award", "honor"],
        "skills": ["skills", "technical", "technologies", "programming", "languages"]
    }

    current_block = None
    for line in text.splitlines():
        l = line.strip().lower()
        for key, keywords in blocks.items():
            if any(k in l for k in keywords):
                current_block = key
        if current_block and line.strip():
            if current_block == "skills":
                extracted[current_block] += [s.strip() for s in re.split(r"[,|\u2022\-•]", line) if s.strip()]
            else:
                extracted[current_block].append(line.strip())

    for k in extracted:
        if isinstance(extracted[k], list):
            extracted[k] = [item for item in sorted(set(extracted[k])) if item]
    
    return extracted

# ----------- Resume-Specific Question Generator -----------
def generate_resume_specific_questions(resume_data):
    """Generate questions specifically tailored to the resume content"""
    prompt = f"""
    You are a technical interview coach. Analyze this resume data and generate exactly 10 technical 
    and behavioral interview questions that would be specifically relevant to this candidate.Just provide me the questions without any general commands.
    
    Resume Summary:
    - Name: {resume_data.get('name', 'N/A')}
    - Skills: {', '.join(resume_data.get('skills', [])) if resume_data.get('skills') else 'N/A'}
    - Experience: {', '.join(resume_data.get('experience', [])) if resume_data.get('experience') else 'N/A'}
    - Projects: {', '.join(resume_data.get('projects', [])) if resume_data.get('projects') else 'N/A'}
    - Education: {', '.join(resume_data.get('education', [])) if resume_data.get('education') else 'N/A'}
    - Certifications: {', '.join(resume_data.get('certifications', [])) if resume_data.get('certifications') else 'N/A'}
    
    Requirements:
    1. Generate exactly 11 questions
    2. Questions must be specifically relevant to the resume content
    3. Include both technical and behavioral questions
    4. For technical roles, focus on their specific skills and tools
    5. For each experience item, generate a follow-up question
    6. Make questions realistic like a real technical interview
    7. More Importantly just generate questions without any general informations like Here are the 10 technical and behavioral interview questions:.
    8. Analyze the extracted information correctly and then produce the questions correctly.

    Return ONLY the questions as a numbered list (1-11), nothing else. """
    
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        questions = response.choices[0].message.content.split('\n')
        # Clean the questions
        questions = [q.split('. ', 1)[1] if '. ' in q else q for q in questions if q.strip()]
        return questions[1:11]  # Ensure we only get 10 questions
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        # Fallback to resume-specific questions
        questions = []
        if resume_data.get('skills'):
            questions.append(f"Can you walk us through your experience with {resume_data['skills'][0]}?")
            if len(resume_data['skills']) > 1:
                questions.append(f"How would you compare {resume_data['skills'][0]} and {resume_data['skills'][1]} in terms of performance and use cases?")
        
        if resume_data.get('experience'):
            questions.append(f"Tell me about your experience at {resume_data['experience'][0].split(' at ')[-1].split(' from ')[0] if ' at ' in resume_data['experience'][0] else resume_data['experience'][0]}")
        
        if resume_data.get('projects'):
            questions.append(f"Describe your project '{resume_data['projects'][0].split(':')[0] if ':' in resume_data['projects'][0] else resume_data['projects'][0]}' and what technologies you used")
        
        if resume_data.get('certifications'):
            questions.append(f"How has your {resume_data['certifications'][0]} certification helped you in practical situations?")
        
        # Fill remaining questions with general but relevant ones
        remaining = 10 - len(questions)
        general_questions = [
            "What technical challenge are you most proud of solving?",
            "Describe a time you had to learn a new technology quickly for a project",
            "How do you stay updated with the latest developments in your field?",
            "Describe a situation where you had to explain a technical concept to a non-technical person",
            "What's your approach to debugging complex issues?",
            "How do you balance quality with deadlines in your work?",
            "Describe your ideal technical stack for a new project",
            "What metrics do you use to measure the success of your technical work?"
        ]
        questions.extend(general_questions[:remaining])
        return questions[0:11]

# ----------- Feedback System -----------
def generate_feedback(questions, answers):
    feedback = []
    scores = []
    
    with st.spinner("Analyzing your answers..."):
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
                st.error(f"Error analyzing answer {i+1}: {e}")
                feedback.append({
                    "question": q,
                    "answer": a,
                    "score": 5,
                    "feedback": "Could not analyze this answer due to an error.",
                    "suggested_answer": ""
                })
                scores.append(5)
    
    return feedback, scores

# def show_feedback_visualization(scores):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.bar(range(1, len(scores)+1), scores, color='skyblue')
#     ax.set_xlabel('Question Number')
#     ax.set_ylabel('Score (out of 10)')
#     ax.set_title('Performance Across Questions')
#     ax.set_ylim(0, 10)
#     ax.set_xticks(range(1, len(scores)+1))
#     st.pyplot(fig)

def show_feedback_visualization(scores):
    labels = [f"Q{i+1}" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    ax.set_title("Performance Distribution Across Questions")
    st.pyplot(fig)


# ----------- Main Application Flow -----------
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    raw_text = extract_resume_text(uploaded_file)
    resume_data = extract_info(raw_text)

    with st.expander("View Extracted Resume Info", expanded=False):
        st.json(resume_data)

    if "questions" not in st.session_state:
        with st.spinner("Generating interview questions specific to your resume..."):
            st.session_state.questions = generate_resume_specific_questions(resume_data)
        st.session_state.q_idx = 0
        st.session_state.answers = []
        st.session_state.speech_text = ""

    questions = st.session_state.questions

    if st.session_state.q_idx < len(questions):
        current_q = st.session_state.q_idx
        st.subheader(f"❓ Question {current_q+1}/{len(questions)}")
        st.info(questions[current_q])
        st.audio(text_to_audio(questions[current_q]), format="audio/mp3", start_time=0)

        # Speech input
        st.markdown("**🎤 Record Your Answer**")
        speech_text = speech_to_text(
            language='en',
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=True,
            key=f"stt_{current_q}"
        )
        
        if speech_text:
            st.session_state.speech_text = speech_text
            st.success("Speech converted to text!")

        # Text input (pre-populated with speech text if available)
        st.markdown("**⌨️ Your Answer**")
        typed_text = st.text_area(
            "Type or edit your answer:",
            value=st.session_state.speech_text,
            key=f"answer_{current_q}",
            height=150,
            label_visibility="collapsed"
        )

        # Submit button
        if st.button("Submit Answer", type="primary"):
            if typed_text.strip():
                st.session_state.answers.append(typed_text)
                st.session_state.q_idx += 1
                st.session_state.speech_text = ""
                st.rerun()
            else:
                st.warning("Please provide an answer before submitting.")

    elif len(st.session_state.answers) == len(questions):
        st.success("🎉 Interview Complete! Generating Feedback...")

        if "scores" not in st.session_state:
            st.session_state.feedback, st.session_state.scores = generate_feedback(questions, st.session_state.answers)

        st.subheader("📊 Your Interview Performance")
        show_feedback_visualization(st.session_state.scores)

        st.subheader("📝 Detailed Feedback")
        for i, item in enumerate(st.session_state.feedback, 1):
            with st.expander(f"Question {i}: {item['question']}", expanded=(i==1)):
                st.markdown(f"**Your Answer:** {item['answer']}")
                st.markdown(f"**Score:** {item['score']}/10")
                st.markdown(f"**Feedback:** {item['feedback']}")
                st.markdown(f"**Suggested Answer:** {item['suggested_answer']}")

        avg_score = sum(score for score in st.session_state.scores) / len(st.session_state.scores)
        st.metric("Overall Average Score", f"{avg_score:.1f}/10")

        if st.button("🔄 Start New Interview"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()