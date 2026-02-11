
from crewai_tools import tool
import prediction


@tool("TriageTool")
def triage_tool(query: str):
    """Predicts Department, Priority, Sentiment and action type for a support query."""
    # This is where you call your existing model_1_pipeline and model_2_pipeline
    prediction = prediction.model_1_pipeline.predict([query])[0]
    dept_prediction = prediction.model_2_pipeline.predict([query])[0]

    return {
        "dept": prediction.label_encoders['Assigned Department'].inverse_transform([prediction[0]])[0],
        "priority": prediction.label_encoders['Priority'].inverse_transform([prediction[1]])[0],
        "sentiment" : prediction.label_encoders['Sentiment'].inverse_transform([prediction[2]])[0],
        "actiontype" : prediction.label_encoders['Assigned Department'].inverse_transform([dept_prediction])[0]
    }



@tool("KnowledgeBaseTool")
def knowledge_base_tool(query: str):
    """Finds historical resolutions for similar customer issues."""
    # Call your resolution_recommender function here
    return prediction.resolution_recommender(query, n=3)

@tool("TimeEstimationTool")
def estimate_resolution_time(complexity_score: float, dept: str, priority: str,sentiment : str):
    """Predicts resolution time using a pre-trained Regression model."""
    # Convert labels back to encoded values as your model expects
    dept_enc = prediction.label_encoders['Assigned Department'].transform([dept])[0]
    priority_enc = prediction.label_encoders['Priority'].transform([priority])[0]
    senti_enc = prediction.label_encoders['Sentiment'].transform([sentiment])[0]
    
    # Run your existing Scikit-Learn regression model
    # prediction = fi.model_3_pipeline.predict([[complexity_score, priority_enc, dept_enc,senti_enc]])
    mins = prediction.time_estimator(dept_enc, priority_enc, senti_enc, complexity_score)
    f"The estimated resolution time is {mins:.2f} minutes."



@tool("OverridingTool")
def overriding_tool(mins: float, priority: str, action_type: str, complexity_score: float):
    """Analyzes the prediction against SLAs and applies overrides for Escalations or Follow-Ups."""
    # If complexity is high, we assume 'Queue Friction'
    if complexity_score > 8.0:
        adjusted_time = mins * 1.2 
        
    else:
        adjusted_time = mins
        
    # 1. Define Business Policy (SLA Limits)
    sla_limits = {"High": 240, "Medium": 540, "Low": 1080}
    limit = sla_limits.get(priority, 1080)               ## anything other than priority would get 1080 mins
    
    
    # 2. Logic for SLA Breach (Escalation Override)
    if mins > limit:
        final_action = "Escalate"
        override_reason = f"CRITICAL: Predicted time ({min:.1f}m) exceeds SLA ({limit}m)."
    
    # 3. Logic for Follow-Up Optimization (Complexity Override)
    elif action_type == "Follow-Up":
        if complexity_score >= 6.0:
           override_reason = "High Complexity Follow-Up: Committed to Senior Technical Review."
           adjusted_time = min(mins, limit * 0.8)
            # We don't change the action here, but we flag it for the supervisor and update time restriction
        else:
            override_reason = "Standard Override Buffer"
            adjusted_time = min + 60  ## add 60 mins buffer
    
    else:
          final_action = action_type
          adjusted_time = mins
          override_reason = "No override needed."
            
    return {
        "Final_action": final_action,
        "Final_time": round(adjusted_time, 2),
        "Override_status": "Active" if final_action != action_type else "Inactive",
        "Reasoning": override_reason 
    }

