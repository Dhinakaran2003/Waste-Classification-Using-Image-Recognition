from database import Database
import os

def test_database():
    db_name = "test_verify.db"
    db = Database(db_name)
    
    print("Testing log_prediction...")
    pred_id = db.log_prediction("test_image.jpg", "plastic", 0.98)
    assert pred_id is not None
    print(f"Prediction logged with ID: {pred_id}")
    
    print("Testing update_feedback...")
    db.update_feedback(pred_id, True)
    
    print("Testing get_recent_predictions...")
    predictions = db.get_recent_predictions(1)
    assert len(predictions) == 1
    assert predictions[0][3] == "plastic"
    assert predictions[0][5] == 1 # Correct
    print("Records verified successfully.")
    
    # Cleanup
    if os.path.exists(db_name):
        os.remove(db_name)
    print("Database verification passed!")

if __name__ == "__main__":
    test_database()
