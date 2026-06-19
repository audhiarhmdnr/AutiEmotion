fetch('/history_data')
    .then(response => response.json())
    .then(data => {

        new Chart(
            document.getElementById('emotionChart'),
            {
                type: 'doughnut',
                data: {
                    labels: data.labels,
                    datasets: [
                        {
                            data: data.values
                        }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            }
        );

    });