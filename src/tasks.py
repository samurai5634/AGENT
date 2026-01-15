from pydantic import BaseModel
from crewai import Task
import agent

class ComplexOutput(BaseModel):
    score: float
    reason: str

complexity_task = Task(
    description="Based on the summary provided, evaluate the technical complexity on a scale of 1-10."
                "Take the below rubric into account"
                "1 to 3 (Low): Routine request (e.g., password reset, hours of operation)."
                "4 to 7 (Medium): Requires investigation (e.g., internet slow, billing discrepancy)."
                "8 to 10 (High): Critical/Infrastructure failure (e.g.server down, data breach, API integration failing).",

    expected_output="A JSON object with a float score and a brief reason.",
    agent=agent.complexity_analyst,
    output_pydantic=ComplexOutput 
)

summary_task = Task(
    description=(
        "Analyze the following customer query: '{query}'. "
        "Remove all greetings (like 'Hello', 'Hope you are well'), emotional filler, "
        "and non-technical information. Extract the core technical issue."
    ),
    expected_output="A single sentence of 10-12 words that summarizes the technical problem.",
    agent=agent.summary_specialist
)

triage_task = Task(
    description=(
        "Analyze the raw customer query: '{query}'. "
        "Use the TriageTool to perform a multi-output classification. "
        "Retrieve the following labels: Action Type, Priority, Sentiment, and Assigned Department. "
        "Do not guess; use the tool to ensure consistency with historical training data."
    ),
    expected_output="A structured record of Predicted Action, Priority, Sentiment, and Department.",
    agent=agent.triager
)

override_task = Task(
    description=(
        "Review the outputs for query: '{query}'. "
        "1. Predict resolution time using the TimeEstimationTool. "
        "2. Evaluate time against SLAs (High: 240m, Med: 540m, Low: 1080m). "
        "3. Apply Dynamic Alterations: "
        "- If SLA Breach Risk is detected, change Action to 'Escalate'. "
        "- If 'Follow-Up' and Complexity > 6, apply 'Technical Priority Commitment'. "
        "- Apply a 20% Realism Buffer if Complexity is above 8. "
        "Provide a clear justification for any overrides."
    ),
    expected_output="The final resolution time and the definitive 'Action Type', including the logic applied.",
    agent=agent.time_agent,
    context=[triage_task,complexity_task] 
)




