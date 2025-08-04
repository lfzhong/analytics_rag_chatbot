import sqlite3
from datetime import datetime

class FeedbackLogger:
    def __init__(self, db_path="feedback_log.db"):
        print("[INIT] FeedbackLogger initialized. DB path:", db_path)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                context TEXT,
                feedback TEXT,
                llm_model TEXT,
                prompt_template_name TEXT,
                rating TEXT,
                retrieved_doc_ids TEXT
            )
        """)
        self.conn.commit()

    def log(self, question, answer, context, feedback, llm_model, prompt_template_name, session_id, rating="neutral",
            retrieved_doc_ids=""):
        try:
            self.conn.execute("""
                INSERT INTO feedback (
                    timestamp, session_id, question, answer, context, feedback,
                    llm_model, prompt_template_name, rating, retrieved_doc_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                session_id,
                question,
                answer,
                context,
                feedback,
                llm_model,
                prompt_template_name,
                rating,
                retrieved_doc_ids
            ))
            self.conn.commit()
            print("[DEBUG] Feedback logged successfully.")
        except Exception as e:
            print("[LOGGING ERROR]", e)  # 👈 this will print the root cause to terminal
