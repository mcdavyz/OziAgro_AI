def generate_dss_recommendation(
    predicted_probability,
    irrigation_available,
    threshold=0.25
):
    """
    Convert predicted dry-spell risk into a planting
    recommendation based on risk and irrigation availability.
    """

    if predicted_probability >= threshold:
        risk_level = "High"
    else:
        risk_level = "Low"

    if risk_level == "Low" and not irrigation_available:
        recommendation = "Planting can proceed."

    elif risk_level == "Low" and irrigation_available:
        recommendation = (
            "Planting can proceed; irrigation can serve as backup."
        )

    elif risk_level == "High" and irrigation_available:
        recommendation = (
            "Planting may proceed with supplemental irrigation."
        )

    else:
        recommendation = (
            "Delay planting until dry-spell risk decreases."
        )

    return {
        "predicted_probability": predicted_probability,
        "risk_level": risk_level,
        "irrigation_available": irrigation_available,
        "recommendation": recommendation
    }
def validate_dss_logic(threshold=0.25):
    """
    Test all four combinations of predicted risk
    and irrigation availability.
    """

    test_cases = [
        {
            "predicted_probability": threshold - 0.01,
            "irrigation_available": False
        },
        {
            "predicted_probability": threshold - 0.01,
            "irrigation_available": True
        },
        {
            "predicted_probability": threshold + 0.01,
            "irrigation_available": True
        },
        {
            "predicted_probability": threshold + 0.01,
            "irrigation_available": False
        }
    ]

    print("\n" + "=" * 50)
    print("OZIAGRO DSS LOGIC VALIDATION")
    print("=" * 50)

    for i, case in enumerate(test_cases, start=1):

        result = generate_dss_recommendation(
            predicted_probability=case["predicted_probability"],
            irrigation_available=case["irrigation_available"],
            threshold=threshold
        )

        print(f"\nTEST CASE {i}")
        print(
            "Predicted probability:",
            result["predicted_probability"]
        )
        print(
            "Risk level:",
            result["risk_level"]
        )
        print(
            "Irrigation available:",
            result["irrigation_available"]
        )
        print(
            "Recommendation:",
            result["recommendation"]
        )