const API_URL = "http://127.0.0.1:8000";

async function loadTeams() {
    const res = await fetch(`${API_URL}/teams`);
    const data = await res.json();
    const teams = data.teams;

    const homeSelect = document.getElementById("homeTeam");
    const awaySelect = document.getElementById("awayTeam");

    teams.forEach(t => {
        const opt1 = document.createElement("option");
        opt1.value = t;
        opt1.textContent = t;

        const opt2 = document.createElement("option");
        opt2.value = t;
        opt2.textContent = t;

        homeSelect.appendChild(opt1);
        awaySelect.appendChild(opt2);
    });
}

async function predict() {
    const home = document.getElementById("homeTeam").value;
    const away = document.getElementById("awayTeam").value;

    const res = await fetch(`${API_URL}/predict?home=${home}&away=${away}`);
    const data = await res.json();

    document.getElementById("result").classList.remove("hidden");

    document.getElementById("probability").innerHTML =
        `<strong>${home}</strong>: ${(data.prob_home * 100).toFixed(2)}%  
         &nbsp;&nbsp; | &nbsp;&nbsp; 
         <strong>${away}</strong>: ${(data.prob_away * 100).toFixed(2)}%`;

    const tbody = document.querySelector("#playersTable tbody");
    tbody.innerHTML = "";

    data.players.forEach(p => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${p.name}</td>
            <td>${p.pred_pts}</td>
            <td>${p.REB}</td>
            <td>${p.AST}</td>
            <td>${p.STL}</td>
            <td>${p.BLK}</td>
            <td>${p.TOV}</td>
        `;
        tbody.appendChild(row);
    });
}

async function loadStats() {
    const res = await fetch(`${API_URL}/stats`);
    const data = await res.json();

    let text = "";

    if (data.team_model) {
        text += "🏀 TEAM MODEL\n";
        text += `Accuracy:  ${(data.team_model.accuracy * 100).toFixed(2)}%\n`;
        text += `Precision: ${(data.team_model.precision * 100).toFixed(2)}%\n`;
        text += `Recall:    ${(data.team_model.recall * 100).toFixed(2)}%\n`;
        text += `F1-Score:  ${data.team_model.f1.toFixed(4)}\n\n`;
    }

    if (data.player_model) {
        text += "👤 PLAYER MODEL\n";
        text += `R²:    ${data.player_model.r2.toFixed(4)}\n`;
        text += `RMSE:  ${data.player_model.rmse.toFixed(3)} pts\n`;
        text += `MAE:   ${data.player_model.mae.toFixed(3)} pts\n`;
    }

    document.getElementById("statsOutput").textContent = text;
    document.getElementById("statsBox").classList.remove("hidden");
}

loadTeams();
loadStats();
