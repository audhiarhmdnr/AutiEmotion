from eeg_module import predict_stress, get_focus

sample = {
    "Raw": 37,
    "Attention": 69,
    "Meditation": 35,
    "delta": 797033,
    "low-alpha": 220853,
    "high-alpha": 15504,
    "low-beta": 8871,
    "high-beta": 202271
}

stress = predict_stress(sample)
focus = get_focus(sample["Attention"])

print("Stress:", stress)
print("Focus :", focus)