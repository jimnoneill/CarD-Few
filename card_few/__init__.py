# __init__.py for CarD-Few package

# Import the main classifier from the model module
from .model import CarDFewClassifier

# Define what gets imported with a "from card_few import *"
__all__ = [
    'CarDFewClassifier',
]
