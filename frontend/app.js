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

loadTeams();