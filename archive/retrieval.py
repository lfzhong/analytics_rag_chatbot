import sqlite3
from collections import defaultdict
import pandas as pd

def get_feedback_score_map():
    conn = sqlite3.connect("feedback_log.db")
    df = pd.read_sql_query("SELECT retrieved_doc_ids, rating FROM feedback", conn)
    conn.close()

    score_map = defaultdict(int)
    for _, row in df.iterrows():
        if not row["retrieved_doc_ids"]:
            continue
        for doc_id in row["retrieved_doc_ids"].split(","):
            if row["rating"] == "positive":
                score_map[doc_id] += 1
            elif row["rating"] == "negative":
                score_map[doc_id] -= 1
    return score_map

def retrieve_relevant_data(question, vector_store):
    results = vector_store.similarity_search(question, k=8)
    score_map = get_feedback_score_map()

    for doc in results:
        boost = score_map.get(doc.metadata.get("doc_id", ""), 0)
        doc.metadata["adjusted_score"] = doc.metadata.get("score", 0) + boost

    results.sort(key=lambda d: d.metadata.get("adjusted_score", 0), reverse=True)

    retrieved_doc_ids = ",".join([doc.metadata.get("doc_id", "") for doc in results])
    retrieved_data_text = "\n".join([doc.page_content for doc in results])

    return results, retrieved_data_text, retrieved_doc_ids

