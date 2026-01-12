from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import TypedDict, NotRequired ,Annotated ,Literal
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import Docx2txtLoader
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
# overall state 
from typing import TypedDict, List, Dict, Optional, NotRequired

from langchain_community.document_loaders import PyPDFLoader






# -----------------------------
# Job Description TypedDict
# -----------------------------
class SkillsState(TypedDict):
    technical_skills: NotRequired[List[str]]
    soft_skills: NotRequired[List[str]]

class EducationState(TypedDict):
    degree: NotRequired[str]
    field_of_study: NotRequired[str]
    institution: NotRequired[str]

class ProjectState(TypedDict):
    project_title: NotRequired[str]
    description: NotRequired[str]
    technologies_used: NotRequired[List[str]]
    outcome: NotRequired[str]
    domain: NotRequired[str]

class JDState(TypedDict):
    job_title: NotRequired[str]
    company: NotRequired[str]
    location: NotRequired[str]
    required_experience_years: NotRequired[int]
    target_roles: NotRequired[List[str]]
    skills: NotRequired[SkillsState]
    education: NotRequired[List[EducationState]]
    responsibilities: NotRequired[List[str]]
    projects: NotRequired[List[ProjectState]]
    certifications: NotRequired[List[str]]
    publications: NotRequired[List[str]]



class ComparisonResult(TypedDict):
    score: NotRequired[int]       # score between 0 and 1
    feedback: NotRequired[str]  # textual comments

class CVEvaluationState(TypedDict):
    input_cv_path: str                     # path to CV PDF
    api_key: NotRequired[str]              # LLM API key
    candidate_cv: NotRequired[dict]        # parsed CV JSON
    job_description: NotRequired[dict]     # JD JSON / TypedDict

    # Node outputs
    education_comparison: NotRequired[ComparisonResult]
    experience_comparison: NotRequired[ComparisonResult]
    skills_comparison: NotRequired[ComparisonResult]
    # summary_comparison: NotRequired[ComparisonResult]

    # Optional: final aggregated score
    overall_fit_score: NotRequired[int]
    final_call:NotRequired[str]
    final_status:NotRequired[str]

# -----------------------------



def upload_cv(state:CVEvaluationState) ->CVEvaluationState:
    print(f"we have sucssfully upload your cv {state["input_cv_path"]} to the pipline ")
    return state


def cv_parsing(state:CVEvaluationState) ->CVEvaluationState:
    def cv_reader(pdf_path):
        
        # Create a loader instance
        loader = PyPDFLoader(pdf_path)
        
        # Load the data
        documents = loader.load()
        all_contents = [doc.page_content for doc in documents]
        single_string = "\n".join(all_contents)
        
        
        return single_string
    
    
    
    
    
    def cv_extraction(cv_path: str, api_key1: str):
        cv_text=cv_reader(cv_path)
    
        
        # -----------------------------
        # CV Schema
        # -----------------------------
        class PersonalInfo(BaseModel):
            full_name: Optional[str] = None
            email: Optional[str] = None
            phone: Optional[str] = None
            location: Optional[str] = None
            links: Dict[str, Optional[str]] = Field(default_factory=dict)
        
        
        class ProfessionalSummary(BaseModel):
            summary_text: Optional[str] = None
            inferred_experience_years: Optional[float] = None
            target_roles: List[str] = Field(default_factory=list)
            domain_keywords: List[str] = Field(default_factory=list)
        
        
        class Skills(BaseModel):
            technical_skills: List[str] = Field(default_factory=list)
            soft_skills: List[str] = Field(default_factory=list)
        
        
        class WorkExperience(BaseModel):
            job_title: Optional[str] = None
            company: Optional[str] = None
            start_date: Optional[str] = None
            end_date: Optional[str] = None
            duration_months: Optional[int] = None
            responsibilities: List[str] = Field(default_factory=list)
            achievements: List[str] = Field(default_factory=list)
        
        
        class Education(BaseModel):
            degree: Optional[str] = None
            field_of_study: Optional[str] = None
            institution: Optional[str] = None
            graduation_year: Optional[int] = None
            gpa: Optional[float] = None
        
        
        class Project(BaseModel):
            project_title: Optional[str] = None
            description: Optional[str] = None
            technologies_used: List[str] = Field(default_factory=list)
            outcome: Optional[str] = None
            domain: Optional[str] = None
        
        
        class CVParsedOutput(BaseModel):
            personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
            professional_summary: ProfessionalSummary = Field(default_factory=ProfessionalSummary)
            skills: Skills = Field(default_factory=Skills)
            work_experience: List[WorkExperience] = Field(default_factory=list)
            education: List[Education] = Field(default_factory=list)
            projects: List[Project] = Field(default_factory=list)
            certifications: List[str] = Field(default_factory=list)
            publications: List[str] = Field(default_factory=list)
        
        
        # -----------------------------
        # CV Parsing Function
        # -----------------------------
        
            """
            Parses raw CV text and outputs structured JSON according to CVParsedOutput.
            Handles missing fields safely and ensures complete JSON output.
            """
    
        # Create parser
        parser = JsonOutputParser(pydantic_object=CVParsedOutput)
    
        # Initialize Gemini model
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.0,
            max_output_tokens=2048,
            top_p=1.0,
            api_key=api_key1
        )
    
        # Build prompt
        prompt = ChatPromptTemplate.from_template(
                                """
                                    {format_instructions}
                                    
                                    You are an expert CV parsing system.
                                    
                                    You MUST extract information for ALL of the following sections:
                                    - personal_info
                                    - professional_summary
                                    - skills
                                    - work_experience
                                    - education
                                    - projects
                                    - certifications
                                    - publications
                                    
                                    If a section is not present in the CV, return it as:
                                    - null for single values
                                    - empty arrays for lists
                                    - empty objects for nested objects
                                    
                                    IMPORTANT:
                                    - Do NOT stop early
                                    - Populate every top-level key in the schema
                                    - Return a COMPLETE JSON object
                                    - Extract each section carefully and provide as much detail as possible.
                                    
                                    CV TEXT:
                                    ----------------
                                    {cv_text}
                                    ----------------
                                    """
        )
    
        # Combine LLM + parser
        chain = prompt | model | parser
    
        # Invoke chain
        return chain.invoke(
            {
                "cv_text": cv_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )

    path = state["input_cv_path"]
    orignal_api_key = state["api_key"]
    parsed_cv = cv_extraction(path, orignal_api_key)

    state["candidate_cv"] = parsed_cv

    return state

def Calculation_node(state: CVEvaluationState) -> dict:
    education_comparison_score= state["education_comparison"]["score"]
    experience_comparison_score= state["experience_comparison"]["score"]
    skills_comparison_score= state["skills_comparison"]["score"]

    state["overall_fit_score"] = education_comparison_score + experience_comparison_score + skills_comparison_score

    return state





def final_evaluation(state: CVEvaluationState) -> Literal["interview_node", "consider_node", "reject_node"]:

    overall_fit_score = state["overall_fit_score"]

    if overall_fit_score >= 60:
        return "interview_node"
    
    elif 30 <= overall_fit_score < 60:
        return "consider_node"
    
    else:
        return "reject_node"


def interview_node(state: CVEvaluationState) -> CVEvaluationState:
    state["final_call"] = "Selected for interview"
    state["final_status"] = "interview"
    return state


def consider_node(state: CVEvaluationState) -> CVEvaluationState:
    state["final_call"] = "Candidate is under consideration"
    state["final_status"] = "consider"
    return state

def reject_node(state: CVEvaluationState) -> CVEvaluationState:
    state["final_call"] = "Application rejected"
    state["final_status"] = "reject"
    return state

def education_comparison_node(state: CVEvaluationState) -> dict:
    candidate_education = state["candidate_cv"]["education"]
    jd_education = state["job_description"]["education"]
    api_key = state["api_key"]

    # -----------------------------
    # LLM Output Schema
    # -----------------------------
    class EducationComparisonResult(BaseModel):
        score: int      # 0–25
        feedback: str

    parser = JsonOutputParser(pydantic_object=EducationComparisonResult)

    # -----------------------------
    # LLM
    # -----------------------------
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=1500,
        api_key=api_key
    )

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = ChatPromptTemplate.from_template(
        """
You are an expert recruitment assistant.

Compare the candidate's education with the job description education requirements.

Candidate Education:
{candidate_education}

Job Description Education Requirements:
{jd_education}

Instructions:
- Evaluate education **technically and semantically**, not word-by-word.
- Closely related degrees (e.g., Software Engineering vs Computer Science) should be treated as strong matches.
- Assign a score between 0 and 33:
    * 25 = Fully matching
    * 15–24 = Closely related
    * 5–14 = Partially related
    * 0 = Not relevant or missing
- If multiple degrees exist, consider the highest relevant one.
- Provide clear feedback.
- Return ONLY valid JSON.

{format_instructions}
"""
    )

    chain = prompt | model | parser

    result = chain.invoke({
        "candidate_education": candidate_education,
        "jd_education": jd_education,
        "format_instructions": parser.get_format_instructions()
    })

    # ✅ Return ONLY what this node updates
    return {
        "education_comparison": {
            "score": result["score"],
            "feedback": result["feedback"]
        }
    }


def skills_comparison_node(state: CVEvaluationState) -> dict:
    
    candidate_skills = state["candidate_cv"]["skills"]
    jd_skills = state["job_description"]["skills"]
    api_key = state["api_key"]

    # LLM Output format
    class SkillsComparisonResult(BaseModel):
        score: int      # 0 to 25
        feedback: str

    parser = JsonOutputParser(pydantic_object=SkillsComparisonResult)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=1500,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert technical recruiter specializing in Machine Learning roles.

Compare the candidate's skills with the job description skill requirements.

Candidate Skills:
{candidate_skills}

Job Description Required Skills:
{jd_skills}

Instructions:
- Perform **semantic and technical matching**, not exact keyword matching.
- Consider related technologies as partial or full matches
  (e.g., PyTorch ↔ TensorFlow, CNNs ↔ Deep Learning, Pandas ↔ Data Analysis).
- Consider depth, breadth, and relevance of skills.
- Assign a score between 0 and 25:
    * 25 = Strong match across most required skills
    * 15–24 = Good match with minor gaps
    * 5–14 = Partial match with significant gaps
    * 0 = Poor or no relevant skills
- Provide clear, professional feedback.
- Return ONLY valid JSON.

{format_instructions}
"""
    )

    chain = prompt | model | parser

    result= chain.invoke({
        "candidate_skills": candidate_skills,
        "jd_skills": jd_skills,
        "format_instructions": parser.get_format_instructions()
    })

    

    # ✅ Return ONLY what this node updates
    return {
        "skills_comparison": {
            "score": result["score"],
            "feedback": result["feedback"]
        }
    }

def experience_comparison_node(state: CVEvaluationState) -> dict:
    
    candidate_experience = state["candidate_cv"]["work_experience"]
    jd_experience =state["job_description"]["required_experience_years"]
    api_key = state["api_key"]

    # LLM Output format
    class ExperienceComparisonResult(BaseModel):
        score: int      # 0 to 25
        feedback: str

    parser = JsonOutputParser(pydantic_object=ExperienceComparisonResult)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=1500,
        api_key=api_key
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert recruitment assistant evaluating Machine Learning experience.

Compare the candidate's professional experience with the job description requirements.

Candidate Experience:
{candidate_experience}

Job Description Experience Requirements:
{jd_experience}

Instructions:
- Evaluate experience **semantically and practically**, not by job title alone.
- Consider:
    * Years of experience
    * Relevance to Machine Learning / AI
    * Type of work performed (model training, deployment, data pipelines)
    * Industry or domain relevance
- Assign a score between 0 and 25:
    * 25 = Fully meets or exceeds experience requirements
    * 15–24 = Relevant experience with minor gaps
    * 5–14 = Limited or partially relevant experience
    * 0 = No relevant experience
- Provide concise, professional feedback.
- Return ONLY valid JSON.

{format_instructions}
"""
    )

    chain = prompt | model | parser

    result= chain.invoke({
        "candidate_experience": candidate_experience,
        "jd_experience": jd_experience,
        "format_instructions": parser.get_format_instructions()
    })

   # ✅ Return ONLY what this node updates
    return {
        "experience_comparison": {
            "score": result["score"],
            "feedback": result["feedback"]
        }
    }
    


# Example JD JSON
ml_engineer_jd = {
    "job_title": "Machine Learning Engineer",
    "company": "TechAI Solutions",
    "location": "Remote / Flexible",
    "required_experience_years": 2,
    "target_roles": ["Machine Learning Engineer", "AI Developer", "Data Scientist"],
    "skills": {
        "technical_skills": [
            "Python","TensorFlow","PyTorch","scikit-learn","Machine Learning",
            "Deep Learning","NLP","Computer Vision","Data Preprocessing","Model Deployment"
        ],
        "soft_skills": [
            "Problem Solving","Team Collaboration","Communication",
            "Analytical Thinking","Time Management"
        ]
    },
    "education": [
        {
            "degree": "Bachelor's or Master's",
            "field_of_study": "Computer Science, Data Science, AI, or related",
            "institution": None
        }
    ],
    "responsibilities": [
        "Design, develop, and deploy machine learning models.",
        "Collaborate with data engineers and software developers.",
        "Optimize ML models for performance and scalability.",
        "Analyze large datasets to extract insights.",
        "Implement data preprocessing and feature engineering pipelines."
    ],
    "projects": [
        {
            "project_title": "ML Model Deployment",
            "description": "Deploy ML models into production using Flask, FastAPI, or cloud services.",
            "technologies_used": ["Python", "Flask", "FastAPI", "Docker", "AWS/GCP"],
            "outcome": None,
            "domain": "Machine Learning"
        }
    ],
    "certifications": [
        "Certified TensorFlow Developer",
        "AWS Machine Learning Specialty",
        "Professional Data Scientist Certificate"
    ],
    "publications": []
}

# Convert to TypedDict instance
jd_state: JDState = ml_engineer_jd




graph = StateGraph(CVEvaluationState)

graph.add_node("upload_cv",upload_cv)
graph.add_node("cv_parsing",cv_parsing)
graph.add_node("education_comparison_node",education_comparison_node)
graph.add_node("skills_comparison_node",skills_comparison_node)
graph.add_node("experience_comparison_node",experience_comparison_node)
graph.add_node("Calculation_node",Calculation_node)
graph.add_node("interview_node",interview_node)
graph.add_node("consider_node",consider_node)
graph.add_node("reject_node",reject_node)


graph.add_edge(START,"upload_cv")
graph.add_edge("upload_cv","cv_parsing")
graph.add_edge("cv_parsing","education_comparison_node")
graph.add_edge("cv_parsing","skills_comparison_node")
graph.add_edge("cv_parsing","experience_comparison_node")


graph.add_edge("skills_comparison_node","Calculation_node")
graph.add_edge("experience_comparison_node","Calculation_node")
graph.add_edge("education_comparison_node","Calculation_node")


graph.add_conditional_edges(

    "Calculation_node",
    final_evaluation,
    {
        "interview_node": "interview_node",
        "consider_node" : "consider_node",
        "reject_node" : "reject_node"
    }
)

graph.add_edge("interview_node",END)
graph.add_edge("consider_node",END)
graph.add_edge("reject_node",END)




workflow = graph.compile()
intail_state = {

    "input_cv_path":"cvs\MUAZZAM CV-.pdf",
    "api_key":orignal_api_key,
    "job_description":jd_state


    
}

final_state = workflow.invoke(intail_state)
print(final_state)



    

    
