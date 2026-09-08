from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("ResumeGenie")
resume_text = st.text_input("Resume Text:")
job_description = st.text_input("Job Description:")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Using chain Extract important skills, experience, technologies, and qualifications from the resume_text.
class ResumeInfoModel(BaseModel):
    skills: list[str] = Field(description="List of skills of the candidate")
    experience: list[str] = Field(description="List of experience of the candidate")
    technologies: list[str] = Field(description="List of technologies known by the candidate")
    qualifications: list[str] = Field(description="List of qualifications of the candidate")

resume_info_prompt = load_prompt('./prompt_library/resume_prompt.json')

structured_model_resume_info = llm.with_structured_output(ResumeInfoModel)

resume_info_chain = resume_info_prompt | structured_model_resume_info

# Using a separate chain, analyze the job description to identify required skills and responsibilities.
class JobDescriptionInfoModel(BaseModel):
    required_skills: list[str] = Field(description="List of skills required for the job")
    responsibilities: list[str] = Field(description="List of responsibilities required for the job")
    qualifications: list[str] = Field(description="List of qualifications required for the job")

job_description_info_prompt = load_prompt('./prompt_library/job_description_prompt.json')

structured_model_job_description_info = llm.with_structured_output(JobDescriptionInfoModel)

job_description_info_chain = job_description_info_prompt | structured_model_job_description_info

# outputs are then passed sequentially to a comparison chain that evaluates the candidate-job match, identifies missing skills, highlights strengths, and generates an overall suitability assessment.
class ComparisonInfoModel(BaseModel):
    missing_skills: list[str] = Field(description="List of skills missing in the candidate's resume compared to the job description")
    strengths: list[str] = Field(description="List of strengths of the candidate based on the resume and job description comparison")
    overall_suitability: str = Field(description="Overall suitability assessment of the candidate for the role")

comparison_info_prompt = load_prompt('./prompt_library/comparison_prompt.json')

structured_model_comparison_info = llm.with_structured_output(ComparisonInfoModel)

comparison_chain = comparison_info_prompt | structured_model_comparison_info

# A final chain provides personalized recommendations for improving the resume or preparing for the role.
class RecommendationsInfoModel(BaseModel):
    recommendations: list[str] = Field(description="List of personalized recommendations for improving the resume or preparing for the role")

recommendations_prompt = load_prompt('./prompt_library/recommendations_prompt.json')

structured_model_recommendations_info = llm.with_structured_output(RecommendationsInfoModel)

recommendations_chain = recommendations_prompt | structured_model_recommendations_info


if st.button("Assess Suitability"):
    with st.spinner("Processing your request... Please wait."):
        # The Streamlit UI can display the skill analysis, match assessment, missing skills, and improvement suggestions.
        resume_JD = {
            "resume_info": resume_info_chain.invoke({'resume_text': resume_text}),
            "job_description_info": job_description_info_chain.invoke({'job_description': job_description})
        }

        comparison_chain_info = comparison_chain.invoke({'resume_info': resume_JD['resume_info'], 'job_description_info': resume_JD['job_description_info']})

        recommendations_chain_info = recommendations_chain.invoke({'resume_info': resume_JD['resume_info'], 'job_description_info': resume_JD['job_description_info'], 'comparison_info': comparison_chain_info})

        st.subheader("Missing Skills:")
        # lambda function to format the missing skills list into a string for display add a newline after each skill for better readability
        st.dataframe(comparison_chain_info.missing_skills, column_config={"missing_skills": "Missing Skills"})
        st.subheader("Strengths:")
        st.dataframe(comparison_chain_info.strengths)
        st.subheader("Overall Suitability Assessment:")
        st.write(comparison_chain_info.overall_suitability)
        st.subheader("Recommendations:")
        st.write("Recommendations:")
        st.dataframe(recommendations_chain_info.recommendations)
    st.success("Done!")

def main():
    pass

if __name__ == "__main__":
    main()
