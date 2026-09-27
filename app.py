from flask import Flask, render_template, request
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import re
import html

app = Flask(__name__)

# AI model
model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# SKILL DATABASE
# =========================================================

SKILLS = [
    "Python", "Java", "SQL", "C++", "C#", "C",
    "JavaScript", "HTML", "CSS",
    "Data Structures", "Machine Learning",
    "Artificial Intelligence", "IoT",
    "Embedded Systems", "MATLAB", "Arduino",
    "Git", "GitHub", "AWS", "Docker",
    "React", "Flask", "Django", "Spring Boot",
    "MySQL", "REST API",
    "Communication", "Teamwork", "Leadership",
    "Critical Thinking", "Problem Solving",
    "Project Management", "Time Management"
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_pdf_text(file):
    """Extract text from uploaded PDF."""
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()


def detect_skills(text):
    """Detect known skills from text."""
    text_lower = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, text_lower):
            found.append(skill)

    return found


def detect_education(text):
    """Detect common educational qualifications."""
    text_lower = text.lower()

    if re.search(r"\b(b\.?e\.?|b\.?tech|bachelor of engineering|bachelor of technology)\b", text_lower):
        return "B.E / B.Tech"

    if re.search(r"\b(b\.?sc|bachelor of science)\b", text_lower):
        return "B.Sc"

    if re.search(r"\b(b\.?com|bachelor of commerce)\b", text_lower):
        return "B.Com"

    if re.search(r"\b(b\.?ba|bachelor of business administration)\b", text_lower):
        return "BBA"

    if re.search(r"\b(b\.?ca|bachelor of computer applications)\b", text_lower):
        return "BCA"

    if re.search(r"\b(b\.?a\.?|bachelor of arts)\b", text_lower):
        return "B.A"

    if re.search(r"\b(m\.?e\.?|m\.?tech|master of engineering|master of technology)\b", text_lower):
        return "M.E / M.Tech"

    if re.search(r"\b(m\.?sc|master of science)\b", text_lower):
        return "M.Sc"

    if re.search(r"\b(mba|master of business administration)\b", text_lower):
        return "MBA"

    if re.search(r"\bdiploma\b", text_lower):
        return "Diploma"

    return "Not detected"


def calculate_semantic_similarity(resume_text, job_description):
    """Calculate semantic similarity using SentenceTransformer."""
    resume_embedding = model.encode(
        resume_text,
        convert_to_tensor=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        resume_embedding,
        job_embedding
    ).item()

    similarity = max(0, min(1, similarity))

    return similarity * 100


def career_suggestions(education, skills):
    """Generate simple career directions."""

    skill_text = " ".join(skills).lower()

    suggestions = []

    if education in ["B.E / B.Tech", "BCA"]:
        if any(x in skill_text for x in ["python", "java", "c++", "javascript"]):
            suggestions.append("Software Development")

        if any(x in skill_text for x in ["iot", "embedded systems", "arduino", "c"]):
            suggestions.append("Embedded / IoT")

        if any(x in skill_text for x in ["machine learning", "artificial intelligence", "python"]):
            suggestions.append("AI / Machine Learning")

        if any(x in skill_text for x in ["matlab", "c", "c++"]):
            suggestions.append("Core Engineering / Technical Roles")

        if not suggestions:
            suggestions.extend([
                "Software Development",
                "Data & Technology Roles",
                "Technical Engineering Roles"
            ])

    elif education == "B.Com":
        suggestions.extend([
            "Finance & Accounting",
            "Banking & Financial Services",
            "Business Analytics"
        ])

    elif education == "BBA":
        suggestions.extend([
            "Business Management",
            "Human Resources",
            "Marketing & Operations"
        ])

    elif education == "B.Sc":
        suggestions.extend([
            "Data & Analytics",
            "Research & Technical Roles",
            "Government Sector"
        ])

    elif education == "B.A":
        suggestions.extend([
            "Government Sector",
            "Administration",
            "Education & Communication"
        ])

    elif education == "Diploma":
        suggestions.extend([
            "Technical Support",
            "Engineering Technician Roles",
            "Government Technical Jobs"
        ])

    else:
        suggestions.extend([
            "Government Sector",
            "Technology Roles",
            "Business & Administration"
        ])

    return suggestions[:4]


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# JOB MATCH ANALYZER
# =========================================================

@app.route("/job-match")
def job_match():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Job Match Analyzer - ResumeAI</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #080d1d;
            color: #ffffff;
        }

        .container {
            width: 90%;
            max-width: 900px;
            margin: 50px auto;
        }

        .back {
            color: #8ea0ff;
            text-decoration: none;
            display: inline-block;
            margin-bottom: 25px;
        }

        .card {
            background: #141b33;
            border: 1px solid #29365d;
            border-radius: 18px;
            padding: 35px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        }

        h1 {
            text-align: center;
            margin-bottom: 12px;
        }

        .subtitle {
            text-align: center;
            color: #aab3d1;
            margin-bottom: 35px;
        }

        label {
            display: block;
            margin: 18px 0 8px;
            font-weight: bold;
        }

        input[type="file"],
        textarea {
            width: 100%;
            padding: 14px;
            border-radius: 10px;
            border: 1px solid #34436f;
            background: #0d1328;
            color: white;
        }

        textarea {
            min-height: 180px;
            resize: vertical;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: #6675ff;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #7885ff;
        }
    </style>
</head>

<body>

<div class="container">

    <a class="back" href="/">← Back to ResumeAI</a>

    <div class="card">

        <h1>🎯 Job Match Analyzer</h1>

        <p class="subtitle">
            Upload your resume and enter a job description.
            ResumeAI will analyse the semantic similarity and skill match.
        </p>

        <form action="/analyze"
              method="POST"
              enctype="multipart/form-data">

            <label>Upload Resume (PDF)</label>

            <input
                type="file"
                name="resume"
                accept=".pdf"
                required
            >

            <label>Job Description</label>

            <textarea
                name="job_description"
                placeholder="Paste the job description here..."
                required
            ></textarea>

            <button type="submit">
                Analyse Resume
            </button>

        </form>

    </div>

</div>

</body>
</html>
"""


# =========================================================
# JOB MATCH ANALYSIS
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    resume_file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not resume_file or not job_description:
        return """
        <h2 style="font-family:Arial;text-align:center;margin-top:100px;">
            Please upload your resume and enter a job description.
        </h2>
        """

    try:
        resume_text = extract_pdf_text(resume_file)

        if not resume_text:
            return """
            <h2 style="font-family:Arial;text-align:center;margin-top:100px;">
                Could not extract text from the uploaded PDF.
            </h2>
            """

        # -------------------------------------------------
        # Semantic similarity
        # -------------------------------------------------

        semantic_score = calculate_semantic_similarity(
            resume_text,
            job_description
        )

        # -------------------------------------------------
        # Skill matching
        # -------------------------------------------------

        resume_skills = detect_skills(resume_text)
        job_skills = detect_skills(job_description)

        if job_skills:
            matched_skills = [
                skill for skill in job_skills
                if skill in resume_skills
            ]

            missing_skills = [
                skill for skill in job_skills
                if skill not in resume_skills
            ]

            skill_score = (
                len(matched_skills) / len(job_skills)
            ) * 100

        else:
            matched_skills = []
            missing_skills = []
            skill_score = 0

        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        final_score = (
            semantic_score * 0.60
            + skill_score * 0.40
        )

        final_score = round(final_score, 2)
        semantic_score = round(semantic_score, 2)
        skill_score = round(skill_score, 2)

        # -------------------------------------------------
        # Display text
        # -------------------------------------------------

        if matched_skills:
            matched_html = "".join(
                f"<span class='tag'>{html.escape(skill)}</span>"
                for skill in matched_skills
            )
        else:
            matched_html = "<span class='muted'>No direct skill match detected</span>"

        if missing_skills:
            missing_html = "".join(
                f"<span class='tag missing'>{html.escape(skill)}</span>"
                for skill in missing_skills
            )
        else:
            missing_html = "<span class='muted'>No major missing skills detected</span>"

        # -------------------------------------------------
        # PROFESSIONAL CONCISE RESULT PAGE
        # -------------------------------------------------

        return f"""
<!DOCTYPE html>
<html>
<head>

    <title>Job Match Result - ResumeAI</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background:
                radial-gradient(circle at top right,
                rgba(102,117,255,0.12),
                transparent 35%),
                #080d1d;
            color: #ffffff;
        }}

        .container {{
            width: 90%;
            max-width: 960px;
            margin: 42px auto;
        }}

        .back {{
            color: #9aa7ff;
            text-decoration: none;
            font-size: 15px;
            display: inline-block;
            margin-bottom: 22px;
        }}

        .back:hover {{
            color: #ffffff;
        }}

        .main-card {{
            background: rgba(20,27,51,0.96);
            border: 1px solid #2d3b66;
            border-radius: 20px;
            padding: 32px;
            box-shadow: 0 20px 55px rgba(0,0,0,0.30);
        }}

        .title {{
            text-align: center;
            margin: 0;
            font-size: 30px;
        }}

        .score {{
            text-align: center;
            font-size: 62px;
            font-weight: 800;
            color: #8290ff;
            margin: 12px 0 5px;
        }}

        .description {{
            text-align: center;
            color: #aeb7d4;
            margin: 0 auto 24px;
            font-size: 14px;
        }}

        .metrics {{
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 25px;
        }}

        .metric {{
            background: #0d142b;
            border: 1px solid #2c3962;
            padding: 11px 18px;
            border-radius: 10px;
            color: #cbd2ea;
            font-size: 14px;
        }}

        .metric strong {{
            color: #ffffff;
        }}

        .sections {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }}

        .section {{
            background: #10172e;
            border: 1px solid #29375f;
            border-radius: 15px;
            padding: 22px;
        }}

        .section h2 {{
            font-size: 18px;
            margin: 0 0 15px;
        }}

        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .tag {{
            display: inline-block;
            padding: 7px 11px;
            border-radius: 8px;
            background: rgba(35, 211, 139, 0.12);
            border: 1px solid rgba(35, 211, 139, 0.35);
            color: #61e4ad;
            font-size: 13px;
        }}

        .tag.missing {{
            background: rgba(255, 181, 71, 0.10);
            border-color: rgba(255, 181, 71, 0.30);
            color: #ffc56c;
        }}

        .muted {{
            color: #8993b3;
            font-size: 14px;
        }}

        .footer-note {{
            text-align: center;
            color: #687394;
            font-size: 12px;
            margin-top: 22px;
        }}

        @media (max-width: 700px) {{

            .container {{
                width: 94%;
                margin: 25px auto;
            }}

            .main-card {{
                padding: 22px;
            }}

            .title {{
                font-size: 24px;
            }}

            .score {{
                font-size: 50px;
            }}

            .sections {{
                grid-template-columns: 1fr;
            }}

        }}

    </style>

</head>

<body>

<div class="container">

    <a class="back" href="/job-match">
        ← Analyse another resume
    </a>

    <div class="main-card">

        <h1 class="title">
            🎯 Job Match Result
        </h1>

        <div class="score">
            {final_score}%
        </div>

        <p class="description">
            AI-estimated compatibility between your resume and the provided job description.
        </p>

        <div class="metrics">

            <div class="metric">
                Semantic Similarity:
                <strong>{semantic_score}%</strong>
            </div>

            <div class="metric">
                Skill Match:
                <strong>{skill_score}%</strong>
            </div>

        </div>

        <div class="sections">

            <div class="section">

                <h2>✅ Matched Skills</h2>

                <div class="tags">
                    {matched_html}
                </div>

            </div>

            <div class="section">

                <h2>📌 Skills to Develop</h2>

                <div class="tags">
                    {missing_html}
                </div>

            </div>

        </div>

        <div class="footer-note">
            ResumeAI • AI-powered career analysis
        </div>

    </div>

</div>

</body>
</html>
"""

    except Exception as e:

        return f"""
        <div style="
            font-family:Arial;
            max-width:700px;
            margin:100px auto;
            padding:30px;
            background:#141b33;
            color:white;
            border-radius:15px;
        ">

            <h2>Something went wrong</h2>

            <p>{html.escape(str(e))}</p>

            <a href="/job-match"
               style="color:#8ea0ff;">
               ← Try again
            </a>

        </div>
        """


# =========================================================
# CAREER SKILL ANALYZER
# =========================================================

@app.route("/career-analysis")
def career_analysis():

    return """
<!DOCTYPE html>
<html>
<head>

    <title>Career Skill Analyzer - ResumeAI</title>

    <style>

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #080d1d;
            color: white;
        }

        .container {
            width: 90%;
            max-width: 900px;
            margin: 50px auto;
        }

        .back {
            color: #8ea0ff;
            text-decoration: none;
        }

        .card {
            margin-top: 25px;
            background: #141b33;
            border: 1px solid #29365d;
            border-radius: 18px;
            padding: 35px;
        }

        h1 {
            text-align: center;
        }

        .subtitle {
            text-align: center;
            color: #aab3d1;
            margin-bottom: 30px;
        }

        label {
            display: block;
            margin: 18px 0 8px;
            font-weight: bold;
        }

        input[type="file"] {
            width: 100%;
            padding: 14px;
            background: #0d1328;
            border: 1px solid #34436f;
            color: white;
            border-radius: 10px;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: #6675ff;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }

    </style>

</head>

<body>

<div class="container">

    <a class="back" href="/">← Back to ResumeAI</a>

    <div class="card">

        <h1>🚀 Career Skill Analyzer</h1>

        <p class="subtitle">
            Upload your resume to discover your skills, education and possible career directions.
        </p>

        <form action="/career-analyze"
              method="POST"
              enctype="multipart/form-data">

            <label>Upload Resume (PDF)</label>

            <input
                type="file"
                name="resume"
                accept=".pdf"
                required
            >

            <button type="submit">
                Analyse Career Profile
            </button>

        </form>

    </div>

</div>

</body>
</html>
"""


# =========================================================
# CAREER ANALYSIS RESULT
# =========================================================

@app.route("/career-analyze", methods=["POST"])
def career_analyze():

    resume_file = request.files.get("resume")

    if not resume_file:
        return "Please upload a resume."

    try:

        resume_text = extract_pdf_text(resume_file)

        education = detect_education(resume_text)
        skills = detect_skills(resume_text)

        suggestions = career_suggestions(
            education,
            skills
        )

        if skills:
            skills_html = "".join(
                f"<span class='tag'>{html.escape(skill)}</span>"
                for skill in skills
            )
        else:
            skills_html = "<span class='muted'>No recognised skills detected.</span>"

        suggestions_html = "".join(
            f"<div class='career'>{html.escape(item)}</div>"
            for item in suggestions
        )

        return f"""
<!DOCTYPE html>
<html>

<head>

<title>Career Analysis - ResumeAI</title>

<meta name="viewport" content="width=device-width, initial-scale=1">

<style>

body {{
    margin:0;
    font-family:Arial,sans-serif;
    background:#080d1d;
    color:white;
}}

.container {{
    width:90%;
    max-width:950px;
    margin:45px auto;
}}

.back {{
    color:#9aa7ff;
    text-decoration:none;
}}

.card {{
    margin-top:22px;
    background:#141b33;
    border:1px solid #2d3b66;
    border-radius:18px;
    padding:30px;
}}

h1 {{
    margin-top:0;
}}

.subtitle {{
    color:#aab3d1;
}}

.info {{
    background:#0d142b;
    border:1px solid #29375f;
    padding:18px;
    border-radius:12px;
    margin-top:20px;
}}

.tags {{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
}}

.tag {{
    padding:7px 11px;
    border-radius:8px;
    background:rgba(102,117,255,.12);
    border:1px solid #4452a0;
    color:#aab5ff;
}}

.career {{
    padding:14px;
    margin-top:10px;
    background:#10172e;
    border:1px solid #29375f;
    border-radius:10px;
}}

.muted {{
    color:#8993b3;
}}

</style>

</head>

<body>

<div class="container">

<a class="back" href="/career-analysis">
← Analyse another resume
</a>

<div class="card">

<h1>🚀 Career Profile</h1>

<p class="subtitle">
AI-generated overview based on your resume.
</p>

<div class="info">
<strong>Education</strong>
<p>{html.escape(education)}</p>
</div>

<div class="info">

<strong>Detected Skills</strong>

<div class="tags" style="margin-top:12px;">
{skills_html}
</div>

</div>

<div class="info">

<strong>Possible Career Directions</strong>

<div style="margin-top:12px;">
{suggestions_html}
</div>

</div>

</div>

</div>

</body>

</html>
"""

    except Exception as e:

        return f"""
        <h2 style="font-family:Arial;text-align:center;margin-top:100px;">
            Error: {html.escape(str(e))}
        </h2>
        """


# =========================================================
# GOVERNMENT EXAM GUIDE
# =========================================================
# =========================================================
# GOVERNMENT EXAM INFORMATION
# =========================================================

@app.route("/exam-info")
def exam_info():

    exam = request.args.get("exam", "").strip().lower()

    exam_data = {
        "upsc-cse": {
            "title": "UPSC Civil Services Examination",
            "category": "Civil Services",
            "about": "UPSC Civil Services Examination is conducted for recruitment to various Group A and Group B civil services of the Government of India.",
            "eligibility": "Generally requires a recognised bachelor's degree. Exact conditions depend on the current UPSC notification.",
            "pattern": "Preliminary Examination → Main Examination → Personality Test.",
            "careers": "Civil services including IAS, IPS, Indian Foreign Service and other services, subject to rank and service allocation.",
            "official": "https://upsc.gov.in/"
        },

        "upsc-ese": {
            "title": "UPSC Engineering Services Examination",
            "category": "Engineering",
            "about": "A UPSC examination for recruitment to engineering services in the Government of India.",
            "eligibility": "Engineering degree or other qualification specified in the current notification.",
            "pattern": "Engineering Services Preliminary → Main → Personality Test.",
            "careers": "Engineering and technical government services.",
            "official": "https://upsc.gov.in/"
        },

        "cds": {
            "title": "Combined Defence Services (CDS)",
            "category": "Defence",
            "about": "UPSC conducts the CDS examination for entry into selected defence academies.",
            "eligibility": "Educational and age requirements vary according to the academy and notification.",
            "pattern": "Written Examination → SSB Interview → Medical and other requirements.",
            "careers": "Officer entry into selected branches of the Indian Armed Forces.",
            "official": "https://upsc.gov.in/"
        },

        "nda": {
            "title": "National Defence Academy (NDA)",
            "category": "Defence",
            "about": "NDA examination provides an entry route for candidates seeking training for officer careers in the Armed Forces.",
            "eligibility": "Educational, age and subject requirements depend on the current notification.",
            "pattern": "Written Examination → SSB → Medical and other selection stages.",
            "careers": "Officer training and careers in the Armed Forces.",
            "official": "https://upsc.gov.in/"
        },

        "capf": {
            "title": "UPSC CAPF",
            "category": "Defence",
            "about": "Central Armed Police Forces examination for recruitment of Assistant Commandants.",
            "eligibility": "Bachelor's degree and other conditions specified in the current notification.",
            "pattern": "Written Examination → Physical Standards / Physical Efficiency → Medical → Interview.",
            "careers": "Assistant Commandant roles in participating Central Armed Police Forces.",
            "official": "https://upsc.gov.in/"
        },

        "ifs": {
            "title": "Indian Forest Service",
            "category": "Environment & Government",
            "about": "UPSC conducts the Indian Forest Service examination for recruitment to the forest service.",
            "eligibility": "Specific degree subjects and other requirements are defined in the current notification.",
            "pattern": "Civil Services Preliminary → IFS Main → Personality Test.",
            "careers": "Forest and environmental administration roles.",
            "official": "https://upsc.gov.in/"
        },

        "ssc-je": {
            "title": "SSC Junior Engineer",
            "category": "Engineering",
            "about": "SSC Junior Engineer recruitment is conducted for technical posts in eligible government organisations.",
            "eligibility": "Diploma or engineering qualification depending on the post and current notification.",
            "pattern": "Computer Based Examination and subsequent selection stages as notified.",
            "careers": "Junior Engineer roles in eligible government departments.",
            "official": "https://ssc.gov.in/"
        },

        "rrb-je": {
            "title": "RRB Junior Engineer",
            "category": "Railways",
            "about": "Railway Recruitment Boards conduct recruitment for Junior Engineer and related technical positions.",
            "eligibility": "Diploma or engineering qualification depending on the notified post.",
            "pattern": "Computer Based Tests followed by applicable selection stages.",
            "careers": "Technical engineering roles in Indian Railways.",
            "official": "https://www.rrbapply.gov.in/"
        },

        "drdo": {
            "title": "DRDO Careers",
            "category": "Science & Technology",
            "about": "DRDO offers scientific, technical and administrative career opportunities through different recruitment routes.",
            "eligibility": "Requirements vary by recruitment, post and qualification.",
            "pattern": "Selection method varies by recruitment and post.",
            "careers": "Scientific, engineering, technical and other roles in defence research.",
            "official": "https://www.drdo.gov.in/"
        },

        "isro": {
            "title": "ISRO Careers",
            "category": "Space & Technology",
            "about": "ISRO provides engineering, scientific and technical career opportunities through recruitment and selection processes.",
            "eligibility": "Requirements vary by post and recruitment notification.",
            "pattern": "Selection stages vary by recruitment.",
            "careers": "Engineering, scientific and technical roles in India's space programme.",
            "official": "https://www.isro.gov.in/"
        },

        "ssc-cgl": {
            "title": "SSC Combined Graduate Level (CGL)",
            "category": "Central Government",
            "about": "SSC CGL is a graduate-level recruitment examination for various central government posts.",
            "eligibility": "Generally requires a bachelor's degree, with post-specific conditions where applicable.",
            "pattern": "Multiple Computer Based Examination tiers as notified.",
            "careers": "Various Group B and Group C posts in central government departments.",
            "official": "https://ssc.gov.in/"
        },

        "ssc-chsl": {
            "title": "SSC CHSL",
            "category": "Central Government",
            "about": "SSC CHSL recruits candidates for various posts such as clerical and data-entry related positions.",
            "eligibility": "Generally based on 12th-standard qualification, subject to post-specific requirements.",
            "pattern": "Computer Based Examination and subsequent stages as notified.",
            "careers": "Clerical, assistant and data-entry related government positions.",
            "official": "https://ssc.gov.in/"
        },

        "ssc-mts": {
            "title": "SSC Multi-Tasking Staff",
            "category": "Central Government",
            "about": "SSC MTS is a recruitment examination for Multi-Tasking Staff and related posts.",
            "eligibility": "Generally based on matriculation-level qualification.",
            "pattern": "Computer Based Examination and other stages as applicable.",
            "careers": "Support and administrative roles in central government offices.",
            "official": "https://ssc.gov.in/"
        },

        "ssc-selection": {
            "title": "SSC Selection Post",
            "category": "Central Government",
            "about": "SSC Selection Post recruitment covers various posts across government departments.",
            "eligibility": "Qualification varies according to the specific post.",
            "pattern": "Computer Based Examination followed by applicable document verification and other stages.",
            "careers": "Different government department positions.",
            "official": "https://ssc.gov.in/"
        },

        "ssc-steno": {
            "title": "SSC Stenographer",
            "category": "Central Government",
            "about": "SSC conducts recruitment for Stenographer Grade C and Grade D posts.",
            "eligibility": "Generally requires 12th-standard qualification and prescribed skill requirements.",
            "pattern": "Computer Based Examination followed by a skill test.",
            "careers": "Stenographer roles in government departments.",
            "official": "https://ssc.gov.in/"
        },

        "ibps-po": {
            "title": "IBPS Probationary Officer",
            "category": "Banking",
            "about": "IBPS PO recruitment is conducted for Probationary Officer / Management Trainee positions in participating banks.",
            "eligibility": "Generally requires a graduation degree.",
            "pattern": "Preliminary Examination → Main Examination → Interview.",
            "careers": "Probationary Officer / Management Trainee roles in participating banks.",
            "official": "https://www.ibps.in/"
        },

        "ibps-clerk": {
            "title": "IBPS Clerk",
            "category": "Banking",
            "about": "IBPS conducts recruitment for clerical positions in participating banks.",
            "eligibility": "Generally requires graduation and other conditions in the notification.",
            "pattern": "Preliminary Examination → Main Examination.",
            "careers": "Clerical roles in participating banks.",
            "official": "https://www.ibps.in/"
        },

        "ibps-rrb": {
            "title": "IBPS RRB",
            "category": "Banking",
            "about": "IBPS RRB recruitment covers various positions in Regional Rural Banks.",
            "eligibility": "Qualification varies according to the post.",
            "pattern": "Online examinations and applicable selection stages.",
            "careers": "Officer and office-assistant roles in Regional Rural Banks.",
            "official": "https://www.ibps.in/"
        },

        "sbi-po": {
            "title": "SBI Probationary Officer",
            "category": "Banking",
            "about": "State Bank of India conducts recruitment for Probationary Officers.",
            "eligibility": "Generally requires graduation, subject to the current recruitment notification.",
            "pattern": "Preliminary → Main → Interview / Group Exercise as notified.",
            "careers": "Probationary Officer roles in SBI.",
            "official": "https://sbi.co.in/"
        },

        "sbi-clerk": {
            "title": "SBI Clerk",
            "category": "Banking",
            "about": "SBI recruits Junior Associates through its clerk-level recruitment process.",
            "eligibility": "Generally requires graduation and other notification-specific conditions.",
            "pattern": "Preliminary Examination → Main Examination.",
            "careers": "Junior Associate / clerical banking roles.",
            "official": "https://sbi.co.in/"
        },

        "rbi": {
            "title": "RBI Opportunities",
            "category": "Banking & Finance",
            "about": "The Reserve Bank of India conducts different recruitment processes for various professional and administrative roles.",
            "eligibility": "Qualification varies according to the specific recruitment.",
            "pattern": "Selection process varies by post.",
            "careers": "Financial, economic, technical, administrative and other roles.",
            "official": "https://www.rbi.org.in/"
        },

        "rrb-ntpc": {
            "title": "RRB NTPC",
            "category": "Railways",
            "about": "RRB NTPC covers recruitment for various non-technical railway positions.",
            "eligibility": "Qualification varies by notified post.",
            "pattern": "Computer Based Tests and applicable subsequent stages.",
            "careers": "Various non-technical roles in Indian Railways.",
            "official": "https://www.rrbapply.gov.in/"
        },

        "rrb-alp": {
            "title": "RRB Assistant Loco Pilot",
            "category": "Railways",
            "about": "Railway Recruitment Boards conduct recruitment for Assistant Loco Pilot positions.",
            "eligibility": "Technical qualification requirements depend on the current notification.",
            "pattern": "Computer Based Tests and applicable aptitude / document stages.",
            "careers": "Assistant Loco Pilot and related railway technical roles.",
            "official": "https://www.rrbapply.gov.in/"
        },

        "rrb-group-d": {
            "title": "RRB Group D",
            "category": "Railways",
            "about": "RRB Group D recruitment covers various Level-1 railway positions.",
            "eligibility": "Qualification and other requirements are defined in the recruitment notification.",
            "pattern": "Computer Based Examination followed by applicable physical and other stages.",
            "careers": "Level-1 operational and support roles in Indian Railways.",
            "official": "https://www.rrbapply.gov.in/"
        },

        "afcat": {
            "title": "AFCAT",
            "category": "Defence",
            "about": "AFCAT is an Indian Air Force entrance examination for eligible officer-entry branches.",
            "eligibility": "Branch-specific educational, age and other conditions apply.",
            "pattern": "Online Examination → AFSB / applicable selection stages → Medical.",
            "careers": "Officer-entry opportunities in eligible Indian Air Force branches.",
            "official": "https://afcat.cdac.in/"
        },

        "coast-guard": {
            "title": "Indian Coast Guard",
            "category": "Defence",
            "about": "Indian Coast Guard recruitment includes officer and technical opportunities through different selection processes.",
            "eligibility": "Depends on the specific post and recruitment notification.",
            "pattern": "Selection stages vary by recruitment.",
            "careers": "Officer, technical and other Coast Guard roles.",
            "official": "https://joinindiancoastguard.cdac.in/"
        },

        "tnpsc-group1": {
            "title": "TNPSC Group 1",
            "category": "Tamil Nadu Government",
            "about": "TNPSC Group 1 recruitment is conducted for higher-level services in the Tamil Nadu government.",
            "eligibility": "Educational and age requirements depend on the current notification.",
            "pattern": "Preliminary Examination → Main Examination → Interview where applicable.",
            "careers": "Higher-level administrative services in Tamil Nadu.",
            "official": "https://www.tnpsc.gov.in/"
        },

        "tnpsc-group2": {
            "title": "TNPSC Group 2",
            "category": "Tamil Nadu Government",
            "about": "TNPSC Group 2 covers multiple Tamil Nadu government services.",
            "eligibility": "Qualification requirements vary by post.",
            "pattern": "Preliminary and Main stages as prescribed.",
            "careers": "Various state government administrative and related positions.",
            "official": "https://www.tnpsc.gov.in/"
        },

        "tnpsc-group4": {
            "title": "TNPSC Group 4",
            "category": "Tamil Nadu Government",
            "about": "TNPSC Group 4 is a major recruitment route for several Tamil Nadu government posts.",
            "eligibility": "Qualification varies by post and current notification.",
            "pattern": "Written examination followed by applicable certificate verification / selection stages.",
            "careers": "Clerical, assistant, village administration and other notified state posts.",
            "official": "https://www.tnpsc.gov.in/"
        },

        "tnpsc-technical": {
            "title": "TNPSC Technical Recruitment",
            "category": "Tamil Nadu Government",
            "about": "TNPSC conducts recruitment for various technical and engineering-related posts.",
            "eligibility": "Technical qualification depends on the specific post.",
            "pattern": "Selection process varies according to the recruitment notification.",
            "careers": "Engineering and technical roles in Tamil Nadu government departments.",
            "official": "https://www.tnpsc.gov.in/"
        },

        "ctet": {
            "title": "Central Teacher Eligibility Test",
            "category": "Teaching",
            "about": "CTET is a national-level teacher eligibility examination conducted by CBSE.",
            "eligibility": "Educational and teacher-training requirements depend on the paper and current rules.",
            "pattern": "Paper I and Paper II with subject-specific sections.",
            "careers": "Eligibility for teaching opportunities where CTET is applicable.",
            "official": "https://ctet.nic.in/"
        },

        "ugc-net": {
            "title": "UGC NET",
            "category": "Teaching & Research",
            "about": "UGC NET determines eligibility for Assistant Professor and/or Junior Research Fellowship according to applicable rules.",
            "eligibility": "Generally requires a relevant postgraduate qualification and other conditions.",
            "pattern": "Computer Based Test with two papers.",
            "careers": "Academic teaching and research opportunities.",
            "official": "https://ugcnet.nta.ac.in/"
        },

        "state-tet": {
            "title": "State Teacher Eligibility Tests",
            "category": "Teaching",
            "about": "State TET examinations assess eligibility for teaching positions in participating state education systems.",
            "eligibility": "Varies according to the state and teaching level.",
            "pattern": "Paper structure varies by state.",
            "careers": "School teaching opportunities where the relevant TET is required.",
            "official": "https://www.education.gov.in/"
        },

        "csir-net": {
            "title": "CSIR-UGC NET",
            "category": "Science & Research",
            "about": "CSIR-UGC NET is a national examination for eligibility related to Junior Research Fellowship and teaching in specified science subjects.",
            "eligibility": "Relevant postgraduate qualification and other conditions apply.",
            "pattern": "Computer Based Test with subject-specific sections.",
            "careers": "Research and academic opportunities in eligible science disciplines.",
            "official": "https://csirnet.nta.ac.in/"
        },

        "icmr": {
            "title": "ICMR Recruitment",
            "category": "Science & Research",
            "about": "ICMR and its institutes recruit for scientific, technical and other positions through different notifications.",
            "eligibility": "Depends on the specific position.",
            "pattern": "Selection process varies by recruitment.",
            "careers": "Biomedical, health research, technical and administrative roles.",
            "official": "https://main.icmr.nic.in/"
        },

        "technical-govt": {
            "title": "Technical Government Careers",
            "category": "Engineering",
            "about": "Government technical careers include engineering, scientific and technical positions across multiple organisations.",
            "eligibility": "Depends on the organisation, discipline and post.",
            "pattern": "Selection method varies by recruitment.",
            "careers": "Engineering, technical and scientific government positions.",
            "official": "https://www.india.gov.in/"
        }
    }

    data = exam_data.get(exam)

    if not data:
        return """
        <h2 style="font-family:Arial;text-align:center;margin-top:100px;color:white;">
            Exam information not available.
        </h2>
        """

    return render_template(
        "exam-info.html",
        data=data
    )
@app.route("/government-guide")
def government_guide():

    return render_template("government-guide.html")


# =========================================================
# GATE EXAM PAGE
# =========================================================

@app.route("/exam/gate")
def gate_exam():

    return render_template("gate.html")


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)