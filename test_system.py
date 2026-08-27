import pytest
from pydantic import ValidationError
from schemas import ImageUnderstanding
from vector_db import VectorDB
import shutil
import os

# --- 1. Schema Validation Tests ---
def test_schema_validation_success():
    """Test that valid JSON parses correctly into the Pydantic schema."""
    valid_data = {
        "subject": "red fox",
        "category": "animal",
        "attributes": ["wild", "orange fur", "forest"],
        "caption": "A red fox stands in the snow.",
        "confidence": 0.94
    }
    model = ImageUnderstanding.model_validate(valid_data)
    assert model.subject == "red fox"
    assert model.confidence == 0.94

def test_schema_validation_failure():
    """Test that invalid JSON fails schema validation (Never trust invalid responses)."""
    invalid_data = {
        "subject": "red fox",
        # missing category, attributes, caption
        "confidence": "not-a-number"
    }
    with pytest.raises(ValidationError):
        ImageUnderstanding.model_validate(invalid_data)

# --- 2. Matching and Mismatch Rejection Tests ---
import tempfile

@pytest.fixture
def test_db():
    """Create a temporary vector database for testing."""
    test_dir = tempfile.mkdtemp()
    db = VectorDB(collection_name="test_collection", persist_directory=test_dir)
    
    # Add mock data
    mock_fox = ImageUnderstanding(
        subject="red fox",
        category="animal",
        attributes=["orange", "wild", "forest"],
        caption="A red fox running in the forest",
        confidence=0.95
    )
    mock_wolf = ImageUnderstanding(
        subject="gray wolf",
        category="animal",
        attributes=["gray", "wild", "forest", "predator"],
        caption="A gray wolf staring in the forest",
        confidence=0.90
    )
    
    db.add_image_understanding(image_id="test_fox.jpg", understanding=mock_fox)
    db.add_image_understanding(image_id="test_wolf.jpg", understanding=mock_wolf)
    
    yield db
    
    # Teardown
    shutil.rmtree(test_dir, ignore_errors=True)

def test_matching_accuracy(test_db):
    """Test that a semantically related post ranks the correct image as ACCEPTED."""
    # Semantic match for fox
    results = test_db.rank_images_for_post("how the wild fox hunts in the snow", top_k=2, max_cosine_distance=0.7)
    
    fox_result = next((r for r in results if r["id"] == "test_fox.jpg"), None)
    assert fox_result is not None
    assert fox_result["status"] == "ACCEPTED"
    assert fox_result["reason"] == "Good match"

def test_mismatch_rejection_subject(test_db):
    """Test that the Mismatch Guard explicitly rejects wrong subjects."""
    results = test_db.rank_images_for_post(
        "how the wild fox hunts in the snow", 
        top_k=2, 
        max_cosine_distance=0.7,
        required_subject="fox"
    )
    
    wolf_result = next((r for r in results if r["id"] == "test_wolf.jpg"), None)
    assert wolf_result is not None
    assert wolf_result["status"] == "REJECTED"
    assert "Subject mismatch" in wolf_result["reason"]

def test_mismatch_rejection_confidence():
    """Test that images with low vision AI confidence are rejected."""
    # Create DB in memory or temp
    test_dir = tempfile.mkdtemp()
    db = VectorDB(collection_name="test_conf", persist_directory=test_dir)
    
    low_conf_dog = ImageUnderstanding(
        subject="dog",
        category="animal",
        attributes=["blurry"],
        caption="A blurry shape that might be a dog",
        confidence=0.40 # Below 0.7 threshold
    )
    db.add_image_understanding(image_id="blurry_dog.jpg", understanding=low_conf_dog)
    
    results = db.rank_images_for_post("a blurry dog", top_k=1, max_cosine_distance=0.99)
    assert results[0]["status"] == "REJECTED"
    assert "confidence too low" in results[0]["reason"]
    
    shutil.rmtree(test_dir, ignore_errors=True)
