function updateDashboard() {

    fetch('/status')
        .then(response => response.json())
        .then(data => {

            document.getElementById('emotion').innerText =
                data.emotion;

            document.getElementById('confidence').innerText =
                data.confidence + "%";

            document.getElementById('stress').innerText =
                data.stress;

            document.getElementById('focus').innerText =
                data.focus;

            document.getElementById('therapy').innerText =
                data.therapy;

            const emotionCard =
                document.getElementById("emotion");

            if (data.emotion === "joy") {
                emotionCard.style.color = "#16a34a";
            }

            else if (data.emotion === "anger") {
                emotionCard.style.color = "#dc2626";
            }

            else if (data.emotion === "fear") {
                emotionCard.style.color = "#ea580c";
            }

            else if (data.emotion === "sadness") {
                emotionCard.style.color = "#2563eb";
            }

            else {
                emotionCard.style.color = "#7c3aed";
            }

        })

        .catch(error => {
            console.log(error);
        });

}

updateDashboard();

setInterval(
    updateDashboard,
    1000
);