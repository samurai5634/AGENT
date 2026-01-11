from crewai import Agent,Task,Crew,LLM,Process
import fi
from crewai_tools import tool

@tool("TriageTool")
def triage_tool(query: str):
    """Predicts Department, Priority, and Sentiment for a support query."""
    # This is where you call your existing model_1_pipeline
    prediction = fi.model_1_pipeline.predict([query])[0]
    return {
        "dept": fi.label_encoders['Assigned Department'].inverse_transform([prediction[0]])[0],
        "priority": fi.label_encoders['Priority'].inverse_transform([prediction[1]])[0],
        "sentiment" : fi.label_encoders['Sentiment'].inverse_transform([prediction[2]])[0]
    }

@tool("KnowledgeBaseTool")
def knowledge_base_tool(query: str):
    """Finds historical resolutions for similar customer issues."""
    # Call your resolution_recommender function here
    return fi.resolution_recommender(query, n=3)


triager = Agent(
    role='Support Triager',
    goal='Categorize incoming tickets and assess urgency'
            'Your task is to predict Department, Priority, and Sentiment for a support query',
    backstory='You are an expert at identifying the core issue in customer messages.',
    tools=[triage_tool],
    # llm=local_llm
)

researcher = Agent(
    role='Knowledge Specialist',
    goal='Find the best historical solution for the query',
    backstory='You have access to all past resolved tickets and technical docs.',
    tools=[knowledge_base_tool],
    memory = True,
    verbose = True,
    # llm=local_llm
)

complexity_analyst = Agent(
    role='Complexity Auditor',
    goal='Analyze customer queries to determine technical difficulty on a scale of 1-10.',
    backstory="""You are a Senior Technical Lead. You evaluate how many resources 
    and how much expertise is needed to solve a ticket. You look for technical 
    keywords and the severity of the mentioned issue.""",
    #llm=local_llm, # Using your Ollama instance
    verbose=True,
    reasoning=True # This makes the agent "think" before scoring
)

from pydantic import BaseModel

class ComplexOutput(BaseModel):
    score: float
    reason: str

complexity_task = Task(
    description="Analyze this customer query: '{query}'. Provide a complexity score from 1-10."
                "Take the below rubric into account"
                "1 to 3 (Low): Routine request (e.g., password reset, hours of operation)."
                "4 to 7 (Medium): Requires investigation (e.g., internet slow, billing discrepancy)."
                "8 to 10 (High): Critical/Infrastructure failure (e.g.server down, data breach, API integration failing).",

    expected_output="A JSON object with a float score and a brief reason.",
    agent=complexity_analyst,
    output_pydantic=ComplexOutput # Forces structured output
)

@tool("TimeEstimationTool")
def estimate_resolution_time(complexity_score: float, dept: str, priority: str,sentiment : str):
    """Predicts resolution time using a pre-trained Regression model."""
    # Convert labels back to encoded values as your model expects
    dept_enc = fi.label_encoders['Assigned Department'].transform([dept])[0]
    priority_enc = fi.label_encoders['Priority'].transform([priority])[0]
    senti_enc = fi.label_encoders['Sentiment'].transform([sentiment])[0]
    
    # Run your existing Scikit-Learn regression model
    # prediction = fi.model_3_pipeline.predict([[complexity_score, priority_enc, dept_enc,senti_enc]])
    min = fi.time_estimator(dept_enc, priority_enc, senti_enc, complexity_score)

    return f"The estimated resolution time is {min:.2f} minutes."


