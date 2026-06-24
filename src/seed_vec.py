from database import seed_vector_knowledge_base
import chromadb

if __name__ == "__main__":
    # Call the batch function using your heavy 6,600+ row dataset
    # (Make sure finaltraining_data.csv is in your project folder)
    seed_vector_knowledge_base('../datasets/finaltraining_data.csv')


#### Retreival on test query 

# Connect to the persistent vector folder
chroma_client = chromadb.PersistentClient(path="./local_vector_db")
collection = chroma_client.get_collection("troubleshooting_manuals")

# Simulate a live, incoming query text string
live_user_query = "I am locked out of my corporate network login and getting an authentication error."

# Run a semantic lookup for the single closest matching historical document
search_results = collection.query(
    query_texts=[live_user_query],
    n_results=1
)

# Extract the retrieved step-by-step resolution text snippet
retrieved_resolution = search_results['documents'][0][0]

print("\n--- SEMANTIC SEARCH RESULT ---")
print(f"Live Query: {live_user_query}")
print(f"Retrieved Historical Fix: {retrieved_resolution}")