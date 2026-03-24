from pydantic import BaseModel, Field
from crewai.tools import tool
from utils import fi 

class TriageInput(BaseModel):
    query: str = Field(description="The raw customer support query text to be classified.")

class OverrideInput(BaseModel):
    predicted_mins: float = Field(..., description="The base minutes predicted by the ML model.")
    priority: str = Field(..., description="The priority level (High, Medium, Low).")
    complexity_score: float = Field(..., description="The technical complexity score (1-10).")
    orig_act: str = Field(..., description="The initial action predicted by the ML model.")

class KnowledgeInput(BaseModel):
    query: str = Field(..., description="The raw query string used to find similar historical cases.")

class TimeEstimationInput(BaseModel):
    # action_label: int = Field(..., description="The encoded integer value for the Action (from Model 2).")
    dept_label: int = Field(..., description="The encoded integer value for the Department (from Model 2).")
    complexity_score: float = Field(..., description="The complexity score (1-10) provided by the Auditor.")
    priority: int = Field(..., description="The encoded integer value for the Department (from Model 2).")
    senti :  int = Field(..., description="The encoded integer value for the Department (from Model 2).")

@tool("TriageTool")
def triage_tool(**kwargs):
    """Predicts Department, Priority, Sentiment and action type for a support query."""
    # This is where you call your existing model_1_pipeline and model_2_pipeline

    props = kwargs.get("properties", {})

    # Step 2: Validate using Pydantic
    data = TriageInput(**props)

    # Step 3: Use it safely
    query = data.query

    prediction = fi.triage.predict([query])[0]
    dept_prediction = fi.action.predict([query])[0]
    return {
        "dept": fi.encoders['Assigned Department'].inverse_transform([prediction[0]])[0],
        "priority": fi.encoders['Priority'].inverse_transform([prediction[1]])[0],
        "sentiment" : fi.encoders['Sentiment'].inverse_transform([prediction[2]])[0],
        "actiontype" : fi.encoders['Assigned Department'].inverse_transform([dept_prediction])[0]
    }




@tool("KnowledgeBaseTool")
def knowledge_base_tool(**kwargs):
    """Finds historical resolutions for similar customer issues."""
    # Call your resolution_recommender function here
    props = kwargs.get("properties", {})

    # Step 2: Validate using Pydantic
    data = TriageInput(**props)

    # Step 3: Use it safely
    query = data.query

    return fi.knn.resolution_recommender(query, n=3)


@tool("TimeEstimationTool")
def estimate_resolution_time(**kwargs):
    """Predicts resolution time using a pre-trained Regression model."""
    # Convert labels back to encoded values as your model expects
    props = kwargs.get("properties", {})

    # Step 2: Validate using Pydantic
    data = TriageInput(**props)

    # Step 3: Use it safely
    query = data.query


    dept_enc = fi.encoders['Assigned Department'].transform([data.dept_label])[0]
    priority_enc = fi.encoders['Priority'].transform([data.priority])[0]
    senti_enc = fi.encoders['Sentiment'].transform([data.senti])[0]
    
    # Run your existing Scikit-Learn regression model
    # prediction = fi.model_3_pipeline.predict([[complexity_score, priority_enc, dept_enc,senti_enc]])
    mins = fi.timer(dept_enc, priority_enc, senti_enc, data.complexity_score)
    f"The estimated resolution time is {mins:.2f} minutes."



@tool("OverridingTool")
def overriding_tool(**kwargs):
    """Analyzes the prediction against SLAs and applies overrides for Escalations or Follow-Ups."""
    # If complexity is high, we add 20 % buffer


    #mins: float, priority: str, action_type: str, complexity_score: float

    props = kwargs.get("properties", {})

    # Step 2: Validate using Pydantic
    data = TriageInput(**props)


    if data.complexity_score > 8.0:
        adjusted_time = data.mins * 1.2 
        
    else:
        adjusted_time = data.mins
        
    # 1. Define Business Policy (SLA Limits)
    sla_limits = {"High": 240, "Medium": 540, "Low": 1080}
    limit = sla_limits.get(data.priority, 1080)               ## anything other than priority would get 1080 mins
    
    
    # 2. Logic for SLA Breach (Escalation Override)
    if data.mins > limit:
        final_action = "Escalate"
        override_reason = f"CRITICAL: Predicted time ({data.mins:.1f}m) exceeds SLA ({limit}m)."
    
    # 3. Logic for Follow-Up Optimization (Complexity Override)
    elif data.orig_act == "Follow-Up":
        if data.complexity_score >= 6.0:
           override_reason = "High Complexity Follow-Up: Committed to Senior Technical Review."
           adjusted_time = min(data.mins, limit * 0.8)
            # We don't change the action here, but we flag it for the supervisor and update time restriction
        else:
            override_reason = "Standard Override Buffer"
            adjusted_time = data.mins + 60  ## add 60 mins buffer
    
    else:
          final_action = data.orig_act
          adjusted_time = data.mins
          override_reason = "No override needed."
            
    return {
        "Final_action": final_action,
        "Final_time": round(adjusted_time, 2),
        "Override_status": "Active" if final_action != data.orig_act else "Inactive",
        "Reasoning": override_reason 
    }
