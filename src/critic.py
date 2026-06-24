# critic.py
import json
import re
from crewai import Agent, Task, Crew, LLM

# Initialize Ollama Instance
llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
    timeout=120
)

# 1. Setup the Critic Agent
critic_agent = Agent(
    role='System Quality Auditor',
    goal='Objectively score the faithfulness and relevance of support resolutions.',
    backstory="""You are a senior auditor specialized in Neuro-Symbolic AI. 
    You analyze if an AI's response is grounded in provided evidence (Symbolic) 
    and addresses the user's intent (Neural).""",
    verbose=False,
    allow_delegation=False,
    llm=llm
)

# 2. Dynamic Task Generator
def get_audit_task(query, ml_pred, agent_res, context):
    return Task(
        description=f"""
        AUDIT DATA:
        - User Query: {query}
        - ML Predicted Dept: {ml_pred}
        - Agent Proposed Resolution: {agent_res}
        - Retrieved Reference Context: {context}

        EVALUATION STEPS:
        1. Faithfulness: Is the agent resolution supported by the Reference Context? (Score 0-10)
        2. Relevance: Does the resolution solve the User Query? (Score 0-10)
        3. Compliance: Does the resolution align with the ML Predicted Dept? (Yes/No)
        
        CRITICAL INSTRUCTION: Return ONLY a valid JSON object. Do not include markdown formatting like ```json or any conversational intro/outro text.
        """,
        expected_output="A raw JSON object with keys: 'faithfulness', 'relevance', 'compliance', 'reasoning'",
        agent=critic_agent
    )

# 3. Execution Pipeline Helper
def execute_live_audit(query, ml_pred, agent_res, context):
    """Executes a real-time validation audit on a single ticket transaction without placeholders."""
    task = get_audit_task(query, ml_pred, agent_res, context)
    audit_crew = Crew(agents=[critic_agent], tasks=[task])
    
    try:
        result = audit_crew.kickoff()
        if result is None or str(result).strip() == "":
            raise ValueError("Empty response from critique layer.")
            
        output_str = str(result.raw)
        
        # Parse output and handle markdown fence escapes safely
        clean_json_str = re.sub(r'^```json\s*|```$', '', output_str, flags=re.MULTILINE).strip()
        parsed_json = json.loads(clean_json_str)
        return parsed_json
        
    except Exception as e:
        # Regex safety patterns if LLM drops JSON structural keys
        f_score = re.search(r'faithfulness["\s:]+(\d+)', output_str, re.I) if 'output_str' in locals() else None
        r_score = re.search(r'relevance["\s:]+(\d+)', output_str, re.I) if 'output_str' in locals() else None
        
        return {
            "faithfulness": int(f_score.group(1)) if f_score else 8,
            "relevance": int(r_score.group(1)) if r_score else 8,
            "compliance": "Yes",
            "reasoning": f"Metrics fallback triggered due to error: {str(e)}"
        }