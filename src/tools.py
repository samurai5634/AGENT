# tools.py
from pydantic import BaseModel, Field
from crewai.tools import tool
from utils import fi 
from prediction import resolution_recommender,time_estimator

@tool("TriageTool")
def triage_tool(query: str) -> dict:
    """Predicts Department, Priority, Sentiment and action type for a support query."""
    try:
        prediction = fi.triage.predict([query])[0]
        dept_prediction = fi.action.predict([query])[0]
        
        return {
            "dept": fi.encoders['Assigned Department'].inverse_transform([prediction[0]])[0],
            "priority": fi.encoders['Priority'].inverse_transform([prediction[1]])[0],
            "sentiment" : fi.encoders['Sentiment'].inverse_transform([prediction[2]])[0],
            "actiontype" : fi.encoders['Assigned Department'].inverse_transform([dept_prediction])[0]
        }
    except Exception:
        return {"dept": "IT Support", "priority": "Medium", "sentiment": "Neutral", "actiontype": "Resolve"}

@tool("KnowledgeBaseTool")
def knowledge_base_tool(query: str) -> list:
    """Finds historical resolutions for similar customer issues using the ChromaDB vector database."""
    # This now executes the query against the actual ChromaDB collection instead of Scikit-learn KNN
    return resolution_recommender(query, n=3)

@tool("TimeEstimationTool")
def estimate_resolution_time(department: str, complexity_score: float, priority: str, sentiment: str) -> float:
    """
    Evaluates whether an SLA override is needed.
    
    CRITICAL: 'predicted_mins' MUST be a raw numerical float value (e.g., 45.0, 120.5). 
    DO NOT pass string descriptors or other tool names like 'time_estimation_tool' into this argument.
    """
    # """Predicts resolution time using a pre-trained Regression model using direct string attributes."""
    mins = time_estimator(department, priority, sentiment, complexity_score)
    return round(mins, 2)

@tool("OverridingTool")
def overriding_tool(predicted_mins: float, priority: str, complexity_score: float, orig_act: str) -> dict:
    """Analyzes execution metrics against operational SLAs to deploy necessary escalation triggers."""
    if complexity_score > 8.0:
        adjusted_time = predicted_mins * 1.2
    else:
        adjusted_time = predicted_mins
        
    sla_limits = {"High": 240, "Medium": 540, "Low": 1080}
    limit = sla_limits.get(priority, 1080)
    
    final_action = orig_act
    override_reason = "No override needed."
    
    if predicted_mins > limit:
        final_action = "Escalate"
        override_reason = f"CRITICAL: Predicted time ({predicted_mins:.1f}m) exceeds SLA ({limit}m)."
    elif orig_act == "Follow-Up":
        if complexity_score >= 6.0:
           override_reason = "High Complexity Follow-Up: Committed to Senior Technical Review."
           adjusted_time = min(predicted_mins, limit * 0.8)
        else:
            override_reason = "Standard Override Buffer"
            adjusted_time = predicted_mins + 60
            
    return {
        "Final_action": final_action,
        "Final_time": round(adjusted_time, 2),
        "Override_status": "Active" if final_action != orig_act else "Inactive",
        "Reasoning": override_reason 
    }