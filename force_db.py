from database import Database
import os

db = Database("final_test.db")
db.log_prediction("test.jpg", "plastic", 0.99)
print("Database file should be created now.")
