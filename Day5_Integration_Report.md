# AgroMind Day 5: Simple System Integration

## 1. Objective
The objective of Day 5 is to demonstrate how the existing components of the AgroMind project fit together conceptually into a cohesive, user-facing application. This phase integrates the core visual perception module (Day 3 CNN) with a supporting supply-planning module (Day 4 LSTM), presenting a unified interface without overcomplicating the architecture.

## 2. Existing Models Reused
- **CNN (Day 3)**: Used for the core task of classifying a tomato image as Ripe or Unripe.
- **LSTM (Day 4)**: Conceptually integrated to provide a supporting demand forecast (simulated in the API for robust demonstration).

*Note: No new ML models were introduced or retrained for Day 5. We strictly reused existing logic.*

## 3. Simple System Workflow
The integration follows a clear, unidirectional workflow:

```text
Camera / Uploaded Image
        ↓
CNN (Perception)
        ↓
Ripe or Unripe (Classification)
        ↓
Harvest Recommendation
```

### Optional Supporting Module:
```text
Historical Sales Data
        ↓
LSTM (Sequence Modeling)
        ↓
Demand Forecast (Supply Planning)
```

## 4. CNN's Role
The Convolutional Neural Network serves as the "eyes" of the AgroMind system. Its sole responsibility is visual perception: analyzing an image and determining the ripeness of the tomato. This is the core prerequisite for any automated harvesting action.

## 5. LSTM's Supporting Role
The LSTM sequence model provides a macroeconomic forecast of upcoming tomato demand. This demand forecast is **supporting information**. It does NOT alter the CNN's ripeness prediction, nor does it control the robotic arm. Instead, it informs farm managers (or high-level orchestration software) about when to scale harvesting operations up or down to meet market needs and reduce waste.

## 6. Example Output
When a user uploads an image to the Streamlit app, the system displays:
- **Predicted Class**: Ripe
- **Confidence**: 98.45%
- **Harvest Recommendation**: "Tomato is ready for harvesting."
- **Supporting Information**: "Forecasted tomato demand: 1250.5 kg" (Labelled clearly to show it does not affect ripeness).

## 7. Limitations
- **Simulated Integration**: While the CNN directly processes the uploaded image, the LSTM forecast is presented as a static daily metric rather than a dynamically updating real-time data stream.
- **No Physical Actuation**: The current project performs software-based classification and recommendation. It does not output physical control signals (like ROS topics) to hardware motors.

## 8. Future Integration with a Real Robotic Arm
In a future robotics phase, the system would evolve from a "Recommendation" to an "Action". The architecture would look like this:
1. The **CNN** processes a live camera feed to detect and localize a ripe tomato in 3D space.
2. The **LSTM** confirms that harvesting is economically viable based on current demand.
3. The system translates the CNN's bounding box coordinates into inverse kinematics for the **Robotic Arm**.
4. The arm physically executes the harvest sequence.
