# db_gui.py
import os
import sys
import subprocess

if __name__ == "__main__":
    import start_DB_container
    start_DB_container.ensure_qdrant_running()
    if os.environ.get("STREAMLIT_RUN") != "1":
        env = os.environ.copy()
        env["STREAMLIT_RUN"] = "1"
        subprocess.run(
            ["streamlit", "run", sys.argv[0]],
            env=env,
            check=True
        )
        sys.exit(0)

import streamlit as st
from vector_db import update_database, search, get_all_documents, delete_document_by_id
st.set_page_config(page_title="Vector DB Admin", layout="wide")

st.title("Qdrant Vector Database Admin")

tab_add, tab_view, tab_search = st.tabs(["Add Data", "View All", "Search"])

with tab_add:
    text = st.text_area("Document text")
    if st.button("Add"):
        update_database([{"document": text}])
        st.success("Added")

with tab_view:
    docs = get_all_documents()
    for d in docs:
        col1, col2 = st.columns([8, 1])
        col1.write(d["document"])
        if col2.button("Delete", key=f"view_{d['id']}"):
            delete_document_by_id(d["id"])
            st.rerun()

with tab_search:
    query = st.text_input("Search")
    if query:
        results = search(query)
        for r in results:
            col1, col2 = st.columns([8, 1])
            col1.write(f"{r['document']}  \n(score={r['score']:.3f})")
            if col2.button("Delete", key=f"search_{r['id']}"):
                delete_document_by_id(r["id"])
                st.rerun()

