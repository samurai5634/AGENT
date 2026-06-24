import sqlite3
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
            complexity_score REAL,          
            predicted_resolution_time REAL,
            final_override_status TEXT,
            faithfulness_score TEXT,       
            relevance_score TEXT,          
            audit_status TEXT,             
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
                predicted_sentiment, predicted_action, complexity_score, predicted_resolution_time, 
                final_override_status, faithfulness_score, relevance_score, audit_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticket_payload['id'],
            ticket_payload['query'],
            ticket_payload['dept'],
            ticket_payload['priority'],
            ticket_payload['sentiment'],
            ticket_payload['action'],
            ticket_payload['complexity'],  
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