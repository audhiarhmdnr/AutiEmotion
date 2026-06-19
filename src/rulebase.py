def get_recommendation(emotion, stress, focus):

    if emotion == "sadness":
        return "Terapi Sedih"

    elif emotion == "anger":
        return "Terapi Marah"

    elif emotion == "fear":
        return "Terapi Takut"

    elif emotion == "joy":
        return "Terapi Senang"

    elif emotion == "Natural":
        return "Terapi Netral"

    return "Tidak ada"