import re

# Supported Intents
INTENT_HARVEST = "HARVEST"
INTENT_CHECK_STATUS = "CHECK_STATUS"
INTENT_FORECAST_DEMAND = "FORECAST_DEMAND"
INTENT_IDENTIFY_RIPENESS = "IDENTIFY_RIPENESS"
INTENT_UNKNOWN = "UNKNOWN"

def parse_command(command_text: str) -> dict:
    """
    Parses a natural language command into a structured intent and extracted entities.
    This is a simple rule-based NLP system for demonstration purposes.
    """
    text = command_text.lower()
    
    # 1. Intent Detection using keyword heuristics
    intent = INTENT_UNKNOWN
    
    if any(word in text for word in ["harvest", "pick", "collect", "gather"]):
        intent = INTENT_HARVEST
    elif any(word in text for word in ["forecast", "demand", "expected", "future"]):
        intent = INTENT_FORECAST_DEMAND
    elif any(word in text for word in ["identify", "are these", "is this", "what is"]):
        # Disambiguate between identify ripeness and check status
        if "ripe" in text or "unripe" in text or "ripeness" in text:
            intent = INTENT_IDENTIFY_RIPENESS
        else:
            intent = INTENT_CHECK_STATUS
    elif any(word in text for word in ["check", "status", "show"]):
        # Disambiguate check status and demand forecast
        if "demand" in text:
            intent = INTENT_FORECAST_DEMAND
        else:
            intent = INTENT_CHECK_STATUS
            
    # 2. Entity Extraction
    entities = {}
    
    # Crop Entity
    if "tomato" in text or "tomatoes" in text:
        entities["crop"] = "TOMATO"
        
    # Ripeness Entity
    # Use regex word boundaries to differentiate 'ripe' from 'unripe'
    if re.search(r'\bunripe\b', text):
        entities["ripeness"] = "UNRIPE"
    elif re.search(r'\bripe\b', text):
        entities["ripeness"] = "RIPE"
        
    # Location Entity
    # Looking for patterns like "greenhouse 1", "greenhouse 2", "section a", etc.
    location_match = re.search(r'(greenhouse\s*\d+|section\s*[a-z])', text)
    if location_match:
        # Standardize formatting (e.g. "greenhouse 2" -> "GREENHOUSE_2")
        loc_str = location_match.group(1).upper().replace(" ", "_")
        entities["location"] = loc_str
        
    # Handle negation specifically for harvest (e.g. "Do not harvest unripe tomatoes")
    if intent == INTENT_HARVEST and ("do not" in text or "don't" in text or "stop" in text):
        intent = "DO_NOT_HARVEST"
        
    # 3. Assemble Structured Command
    structured_command = {
        "intent": intent,
    }
    structured_command.update(entities)
    
    return structured_command

if __name__ == "__main__":
    # Quick tests
    print(parse_command("Harvest the ripe tomatoes in greenhouse 2."))
    print(parse_command("Check the tomatoes in section A."))
    print(parse_command("Do not harvest unripe tomatoes."))
    print(parse_command("Forecast tomato demand."))
