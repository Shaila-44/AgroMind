# AgroMind Day 6: Natural Language Interface

## 1. Objective
The objective of Day 6 is to introduce a simple Natural Language Processing (NLP) component as an interface layer for the AgroMind system. This allows a human operator (e.g., a farmer or manager) to issue text-based commands that the system can parse into structured instructions.

## 2. Why NLP is Relevant to AgroMind
While Computer Vision (Day 3) answers the question *"What do I see?"* (is the tomato ripe or unripe?), NLP answers the question *"What does the operator want to do?"*. By combining these two modalities, we enable a future robotic system that can receive high-level instructions (e.g., "Harvest ripe tomatoes in greenhouse 2") and execute them autonomously based on visual perception. NLP acts strictly as a human-computer interaction layer and does NOT replace the computer vision system.

## 3. NLP Approach
To keep the system lightweight, transparent, and explainable, we implemented a **rule-based NLP parser** (`nlp.py`). 
We explicitly avoided Large Language Models (LLMs) or complex Transformer architectures because:
1. They introduce unnecessary latency and computational overhead.
2. The domain (tomato harvesting) has a very small, restricted vocabulary.
3. Rule-based systems provide 100% deterministic, explainable command parsing.

The parser uses text preprocessing, keyword/phrase matching for intent detection, and Regex/keyword matching for entity extraction.

## 4. Supported Intents
The system currently supports the following Intents:
- **`HARVEST`**: Command to pick tomatoes.
- **`DO_NOT_HARVEST`**: Explicit command to avoid harvesting (handling negations).
- **`FORECAST_DEMAND`**: Request to view the Day 4 LSTM demand predictions.
- **`CHECK_STATUS`**: Request to check the general status of a location or crop.
- **`IDENTIFY_RIPENESS`**: Request to identify if specific tomatoes are ripe.
- **`UNKNOWN`**: Fallback when a command is not understood.

## 5. Entity Extraction
The parser extracts relevant parameters (entities) to pass to downstream systems:
- **`crop`**: Identifies the crop (e.g., `TOMATO`).
- **`ripeness`**: Desired state (e.g., `RIPE`, `UNRIPE`).
- **`location`**: Specific physical zones extracted via Regex (e.g., `GREENHOUSE_2`, `SECTION_A`).

## 6. Example Commands & 7. Structured Outputs

**Example 1:**
- **User:** "Harvest the ripe tomatoes in greenhouse 2."
- **Structured Output:** 
  ```json
  {
    "intent": "HARVEST",
    "crop": "TOMATO",
    "ripeness": "RIPE",
    "location": "GREENHOUSE_2"
  }
  ```

**Example 2:**
- **User:** "Do not harvest unripe tomatoes."
- **Structured Output:** 
  ```json
  {
    "intent": "DO_NOT_HARVEST",
    "crop": "TOMATO",
    "ripeness": "UNRIPE"
  }
  ```

## 8. Evaluation
The NLP module was evaluated using a manually defined test dataset (`evaluate_nlp.py`).
- **Intent Accuracy**: 100% on the restricted test set.
- **Successful Interpretations**: Successfully disambiguated "Check tomatoes" (`CHECK_STATUS`) vs "Are these tomatoes ripe?" (`IDENTIFY_RIPENESS`). Successfully mapped negation to `DO_NOT_HARVEST`.
- **Failure Cases (Handled)**: When asked to "Water the crops in section B", it correctly identified the `location` but gracefully defaulted to the `UNKNOWN` intent, preventing hallucinated actions.

*Note: Because this is a small rule-based system, this evaluation serves as a functional demonstration rather than a large-scale statistical benchmark.*

## 9. Limitations
- **Vocabulary Constraints**: The system relies on exact keywords. It cannot understand synonyms unless explicitly programmed (e.g., if a user says "Pluck the red ones", it will fail).
- **No Typo Tolerance**: Basic string matching is brittle against spelling mistakes.
- **No Physical Control**: The generated structured commands are merely displayed in the Streamlit UI. They do not physically actuate robotic hardware.

## 10. Future Improvements
- **Fuzzy Matching**: Introduce Levenshtein distance or TF-IDF matching to handle typos and synonyms.
- **Robot Controller Integration**: Route the structured JSON output into an automated planner (like ROS) to physically drive the robot to the specified `location` and begin the `HARVEST` loop using the CNN.
