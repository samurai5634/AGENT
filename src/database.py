# database.py
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

def init_ticket_db():
    """Initializes a local relational database with extended audit metrics."""
    conn = sqlite3.connect('enterprise_itsm.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            customer_query TEXT,
            predicted_department TEXT,
            predicted_priority TEXT,
            predicted_sentiment TEXT,
            predicted_action TEXT,
            predicted_resolution_time REAL,
            final_override_status TEXT,
            faithfulness_score TEXT,       -- Integrated from Critic Agent
            relevance_score TEXT,          -- Integrated from Critic Agent
            audit_status TEXT,             -- Integrated from Critic Agent
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✓ Local Production Relational Database Initialized.")

def commit_processed_ticket(ticket_payload):
    """Inserts processed ticket telemetry safely using a secure rollback transaction."""
    conn = sqlite3.connect('enterprise_itsm.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO tickets (
                ticket_id, customer_query, predicted_department, predicted_priority, 
                predicted_sentiment, predicted_action, predicted_resolution_time, final_override_status,
                faithfulness_score, relevance_score, audit_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticket_payload['id'],
            ticket_payload['query'],
            ticket_payload['dept'],
            ticket_payload['priority'],
            ticket_payload['sentiment'],
            ticket_payload['action'],
            ticket_payload['time_pred'],
            ticket_payload['override'],
            ticket_payload['faithfulness'],
            ticket_payload['relevance'],
            ticket_payload['audit_status']
        ))
        conn.commit()
        print(f"✓ Ticket {ticket_payload['id']} committed safely to database queue.")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"⚠ Database transaction failed. Rollback executed: {e}")
    finally:
        conn.close()

def seed_vector_knowledge_base(csv_path):
    """Ingests historical data into vector indices in chunks under SQLite limits."""
    chroma_client = chromadb.PersistentClient(path="./local_vector_db")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = chroma_client.get_or_create_collection(
        name="troubleshooting_manuals", 
        embedding_function=default_ef
    )
    
    df = pd.read_csv('../datasets/finaltraining_data.csv')
    
    all_ids = [f"KB_DOC_{idx}" for idx in range(len(df))]
    all_documents = df['Resolution_Steps'].astype(str).tolist()
    all_metadatas = [{"department": str(dept)} for dept in df['Assigned Department']]
    
    BATCH_SIZE = 5000 
    total_records = len(all_documents)
    
    print(f"Starting vector ingestion loop: Processing {total_records} rows in batches of {BATCH_SIZE}...")
    for i in range(0, total_records, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, total_records)
        collection.upsert(
            ids=all_ids[i:end_idx],
            documents=all_documents[i:end_idx],
            metadatas=all_metadatas[i:end_idx]
        )
        print(f"✓ Successfully indexed rows: {i} to {end_idx}")

    print("✓ Vector Knowledge Base Seeded and Safe from Batch Errors!")