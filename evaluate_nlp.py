from nlp import parse_command
import json

test_commands = [
    "Harvest the ripe tomatoes.",
    "Harvest the ripe tomatoes in greenhouse 2.",
    "Check the tomatoes in section A.",
    "Show me the expected tomato demand.",
    "Are these tomatoes ripe?",
    "Do not harvest unripe tomatoes.",
    "Water the crops in section B." # A test for an unknown/unsupported action
]

print("==================================================")
print("DAY 6 NLP EVALUATION")
print("==================================================\\n")

for cmd in test_commands:
    print(f"Input: '{cmd}'")
    result = parse_command(cmd)
    
    intent = result.get("intent")
    
    # Format entities string
    entities = {k: v for k, v in result.items() if k != "intent"}
    ent_str = json.dumps(entities)
    
    print("↓")
    print(f"Detected Intent: {intent}")
    print("↓")
    print(f"Extracted Entities: {ent_str}")
    print("↓")
    print(f"Structured Command: {json.dumps(result, indent=2)}")
    print("\\n--------------------------------------------------\\n")
