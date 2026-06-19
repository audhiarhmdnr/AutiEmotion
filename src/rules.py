def get_rule(emotion: str):
    if not emotion:
        return {"therapy": "Standard ABA Activities", "avatar": "default.png"}

    emotion = emotion.strip().lower()

    RULES = {
        "joy": {
            "therapy": "Social Play & Interaction",
            "avatar": "happy.png"
        },
        "sadness": {
            "therapy": "Calming Music & Guided Breathing",
            "avatar": "sad.png"
        },
        "anger": {
            "therapy": "Relaxation Exercises",
            "avatar": "angry.png"
        },
        "natural": {
            "therapy": "Standard ABA Activities",
            "avatar": "neutral.png"
        },
        "surprise": {
            "therapy": "Exploratory Play",
            "avatar": "surprised.png"
        },
        "fear": {
            "therapy": "Calming Safety Exercises",
            "avatar": "fear.png"
        }
    }

    return RULES.get(emotion, {
        "therapy": "Standard ABA Activities",
        "avatar": "default.png"
    })