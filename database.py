import sqlite3
from datetime import datetime
import os

class Database:
    def __init__(self, db_path="waste_classifier.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes the database and creates tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_correct INTEGER,
                    actual_class TEXT
                )
            """)
            conn.commit()

    def log_prediction(self, image_name, predicted_class, confidence):
        """Logs a prediction to the database."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (timestamp, image_name, predicted_class, confidence)
                VALUES (?, ?, ?, ?)
            """, (timestamp, image_name, predicted_class, confidence))
            conn.commit()
            return cursor.lastrowid

    def update_feedback(self, prediction_id, is_correct, actual_class=None):
        """Updates a prediction record with user feedback."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE predictions
                SET is_correct = ?, actual_class = ?
                WHERE id = ?
            """, (1 if is_correct else 0, actual_class, prediction_id))
            conn.commit()

    def get_recent_predictions(self, limit=10):
        """Retrieves recent predictions from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, image_name, predicted_class, confidence 
                FROM predictions ORDER BY id DESC LIMIT ?
            """, (limit,))
            return cursor.fetchall()

    def get_recent_history(self, limit=10):
        """Alias for get_recent_predictions for UI consistency."""
        return self.get_recent_predictions(limit)

    def get_stats(self):
        """Returns summary statistics for the dashboard."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), AVG(confidence) FROM predictions")
            count, avg_conf = cursor.fetchone()
            return {
                "total_scans": count or 0,
                "avg_confidence": round((avg_conf or 0) * 100, 1)
            }

if __name__ == "__main__":
    # Simple test
    db = Database("test.db")
    pred_id = db.log_prediction("test.jpg", "plastic", 0.95)
    print(f"Logged prediction with ID: {pred_id}")
    db.update_feedback(pred_id, True)
    print("Updated feedback.")
    print("Recent predictions:", db.get_recent_predictions())
    os.remove("test.db")
