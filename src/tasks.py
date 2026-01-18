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


research_task = Task(
    description=(
        "Using the raw customer query: '{query}', search the historical database "
        "using the KnowledgeBaseTool. You must identify the 3 most similar past cases. "
        "Analyze the 'Resolution_Steps' from these cases to see if they can be applied "
        "to the current situation. "
        "\n\nFocus specifically on identifying technical commonalities—if past "
        "similar queries resulted in an 'Escalation', note that for the Orchestrator."
    ),
    expected_output="""A technical briefing containing:
    1. A list of 3 similar historical queries.
    2. The exact resolution steps used in those cases.
    3. A synthesized recommendation on how to solve the current query based on history.""",
    agent=agent.researcher 
    #  it helps the Orchestrator synthesize the final report.
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

orchestrator_task = Task(
    description="""
    1. Look at the Summary from the Summary Agent.
    2. Look at the ML Labels from the Triage Agent.
    3. Look at the SLA Override and Final Action from the Time Agent.
    4. Look at the Historical Fixes from the Knowledge Specialist.
    
    Synthesize all this into a ONE-PAGE report for the supervisor. 
    Ensure the 'FINAL ACTION' is clearly highlighted based on the Overrider's decision.

    Take this template into account : 
    
    "original_insights": {
        "summary": "Short 10-word technical summary",
        "initial_action": "Predicted by Model 1",
        "priority": "High/Medium/Low"
    },
    "technical_analysis": {
        "complexity_score": 8.5,
        "base_prediction": "320.5 mins",
        "adjusted_prediction": "384.6 mins (20% Complexity Buffer Applied)"
    },
    "final_decision": {
        "action_type": "Escalate",  # The Altered Field
        "sla_status": "CRITICAL BREACH RISK",
        "routing_reason": "Complexity-induced delay exceeds 240m High-Priority SLA."
    }

    """,
    expected_output="A professional ticket brief containing: Summary, Final Action, Routing Dept, and Suggested Fix.",
    agent=agent.orchestrator_agent,
    context=[summary_task, triage_task, override_task] 
)


