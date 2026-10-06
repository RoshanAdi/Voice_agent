from vector_db import update_database, search, get_all_documents, delete_document_by_id, delete_search_results

# Add new documents
update_database([
    {"document": "Reset password via settings.", "source": "FAQ"},
    {"document": "Live chat available 24/7.", "source": "FAQ"}
])

# Search
results = search("How do I reset my password?")
for r in results:
    print(r)

# Get all documents
all_docs = get_all_documents()
for doc in all_docs:
    print(doc)

# Delete one by ID
delete_document_by_id(all_docs[0]["id"])

# Delete search results
deleted_ids = delete_search_results("reset password")
print("Deleted IDs:", deleted_ids)
