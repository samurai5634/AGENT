import sqlite3
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import uuid

# ==========================================
# 1. INITIALIZE RELATIONAL ENGINE (SQLite)
# ==========================================
def init_relational_db():
    """Establishes tables for real-time ticket tracking and SLA audits"""
    conn = sqlite3.connect('enterprise_itsm.db')
    cursor = conn.cursor()
    
    # Core operational ticket queue table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_query TEXT,
            assigned_department TEXT,
            predicted_priority TEXT,
            sentiment TEXT,
            predicted_action TEXT,
            complexity_score INT,
            predicted_resolution_time REAL,
            override_status TEXT DEFAULT 'NONE',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Quality Assurance / Critic Agent log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS critic_audits (
            audit_id TEXT PRIMARY KEY,
            ticket_id TEXT,
            faithfulness_score INT,
            relevance_score INT,
            compliance_status TEXT,
            reasoning_logs TEXT,
            FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
        )
    ''')
    conn.commit()
    conn.close()
    print("✓ Relational Database Infrastructure Initialized.")

# ==========================================
# 2. INITIALIZE LOCAL VECTOR DATABASE (Chroma)
# ==========================================
def seed_vector_knowledge_base(csv_path):
    """Converts static CSV text blocks into local semantic vector indices"""
    # Initialize a persistent local vector client
    chroma_client = chromadb.PersistentClient(path="./local_vector_db")
    
    # Using a standard default local embedding function
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.get_or_create_collection(
        name="troubleshooting_manuals", 
        embedding_function=default_ef
    )
    
    # Load your historical csv file to seed the database
    df = pd.read_csv('../datasets/finaltraining_data.csv')
    
    ids = []
    documents = []
    metadatas = []
    
    for idx, row in df.iterrows():
        unique_id = f"KB_DOC_{idx}"
        ids.append(unique_id)
        # Using the resolution steps text as the dense document
        documents.append(str(row['Resolution_Steps']))
        metadatas.append({"department": str(row['Assigned Department'])})
        
    # Bulk insert up to your local vector store limits
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"✓ Vector Knowledge Base Seeded with {len(documents)} records.")

# ==========================================
# 3. LIVE PIPELINE TRANSACTION WRITING
# ==========================================
def insert_live_ticket_metrics(ticket_data, audit_data):
    """Executes a thread-safe transaction committing agent outputs to DB"""
    conn = sqlite3.connect('enterprise_itsm.db')
    cursor = conn.cursor()
    
    try:
        # Commit quantitative and triage metadata fields
        cursor.execute('''
            INSERT INTO tickets (ticket_id, customer_query, assigned_department, predicted_priority, sentiment, predicted_action, complexity_score, predicted_resolution_time, override_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticket_data['id'], ticket_data['query'], ticket_data['dept'], 
            ticket_data['priority'], ticket_data['sentiment'], ticket_data['action'],
            ticket_data['complexity'], ticket_data['time_pred'], ticket_data['override']
        ))
        
        # Commit qualitative Critic Agent metrics
        cursor.execute('''
            INSERT INTO critic_audits (audit_id, ticket_id, faithfulness_score, relevance_score, compliance_status, reasoning_logs)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), ticket_data['id'], audit_data['faithfulness'],
            audit_data['relevance'], audit_data['compliance'], audit_data['reasoning']
        ))
        
        conn.commit()
        print(f"✓ Transaction successfully committed for Ticket: {ticket_data['id']}")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"⚠ Database transaction failed. Rollback executed: {e}")
    finally:
        conn.close()

# ==========================================
# 4. RUN PIPELINE INITIALIZATION
# ==========================================
if __name__ == "__main__":
    # 1. Build the tables
    init_relational_db()
    
    # 2. Seed Vector DB (Ensure your training file is in the same folder)
    try:
        seed_vector_knowledge_base('ticket_test_dataset.csv')
    except Exception as e:
        print(f"Skipping vector seeding. Please check CSV filepath. Error: {e}")