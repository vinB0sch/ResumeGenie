from langchain_core.prompts import PromptTemplate

template_resume = PromptTemplate(
    template="""You are a professional Career and Resume advisor. Based on the user's resume, extract skills, experience, technologies and qualifications.
User Info:
- Resume Text: {resume_text}
Keep the tone Professional, concise and informative.""", input_variables=['resume_text'])

template_resume.save('./prompt_library/resume_prompt.json')

template_job_description = PromptTemplate(
    template="""You are a professional Career and Resume advisor. Based on the provided job description text, extract required skills, responsibilities, and qualifications.
Job Description Info:
- Job Description: {job_description}
Keep the tone Professional, concise and informative.""", input_variables=['job_description'])

template_job_description.save('./prompt_library/job_description_prompt.json')

template_comparison = PromptTemplate(
    template="""You are a professional Career and Resume advisor. Based on the extracted information from the user's resume and the job description, compare the two and provide an assessment of the candidate's suitability for the role. Highlight missing skills, strengths, and provide an overall suitability assessment.
Resume Info:
- Skills, Experience, Technologies, Qualifications: {resume_info}
Job Description Info:
- Required Skills, Responsibilities, Qualifications: {job_description_info}
Keep the tone Professional, concise and informative.""", input_variables=['resume_info', 'job_description_info'])

template_comparison.save('./prompt_library/comparison_prompt.json')

template_recommendations = PromptTemplate(
    template="""You are a professional Career and Resume advisor. Based on the comparison of the user's resume and the job description, provide personalized recommendations for improving the resume or preparing for the role.
Recommendations should focus on addressing missing skills, enhancing strengths, and improving overall suitability for the role.
Resume Info:
- Skills, Experience, Technologies, Qualifications: {resume_info}
Job Description Info:
- Required Skills, Responsibilities, Qualifications: {job_description_info}
Comparison Info:
- Assessment of Candidate's Suitability: {comparison_info}
Keep the tone Professional, concise and informative.""", input_variables=['resume_info', 'job_description_info', 'comparison_info'])

template_recommendations.save('./prompt_library/recommendations_prompt.json')
