// Global State
let activeTab = "tab-baseline";

// Baseline Mode State
let baselineInterval = null;
let baselineRunning = false;
let chartBaseline = null;

// Self-Play Mode State
let selfplayInterval = null;
let selfplayRunning = false;
let chartSelfplay = null;

// Saved Game State
let savedSymbol = "X";

// Live Learning State
let liveSymbol = "X";


// Helper: Render 3x3 Board
function renderBoard(containerId, boardMatrix, probsDict = {}, isClickable = false, clickCallback = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";

    // Find highest probability for styling
    let maxProb = 0;
    if (probsDict) {
        Object.values(probsDict).forEach(p => {
            if (p > maxProb) maxProb = p;
        });
    }

    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const val = boardMatrix[r][c];
            const idx = r * 3 + c;
            const cell = document.createElement("div");
            cell.className = "cell";
            cell.dataset.row = r;
            cell.dataset.col = c;
            cell.dataset.idx = idx;

            if (val === 1) {
                cell.classList.add("taken");
                cell.innerHTML = `<span class="cell-symbol x">X</span>`;
            } else if (val === -1) {
                cell.classList.add("taken");
                cell.innerHTML = `<span class="cell-symbol o">O</span>`;
            } else {
                // Empty cell
                if (probsDict && probsDict[idx] !== undefined) {
                    const probVal = probsDict[idx];
                    const isHigh = probVal > 0 && probVal === maxProb;
                    if (isHigh) cell.classList.add("prob-high");
                    cell.innerHTML = `<span class="cell-prob">${probVal}%</span>`;
                }

                if (isClickable && clickCallback) {
                    cell.addEventListener("click", () => clickCallback(r, c));
                }
            }

            container.appendChild(cell);
        }
    }
}


// Tab Switching
document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        const target = btn.dataset.tab;
        if (target === activeTab) return;

        // Stop running loops when navigating away
        if (activeTab === "tab-baseline" && baselineRunning) toggleBaselineAuto();
        if (activeTab === "tab-selfplay" && selfplayRunning) toggleSelfplayAuto();

        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

        btn.classList.add("active");
        document.getElementById(target).classList.add("active");
        activeTab = target;

        // Initialize target tab if needed
        if (target === "tab-saved") {
            startSavedGame();
        } else if (target === "tab-live") {
            startLiveGame();
        }
    });
});


// ==========================================
// MODE 1: BASELINE TRAINING
// ==========================================
function initBaselineChart() {
    const ctx = document.getElementById("chart-baseline").getContext("2d");
    chartBaseline = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Win Rate %",
                    data: [],
                    borderColor: "#10b981",
                    backgroundColor: "rgba(16, 185, 129, 0.1)",
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: "yWin"
                },
                {
                    label: "Loss",
                    data: [],
                    borderColor: "#f43f5e",
                    backgroundColor: "rgba(244, 63, 94, 0.1)",
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: "yLoss"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    grid: { color: "#26334a" },
                    ticks: { color: "#94a3b8", maxTicksLimit: 8 }
                },
                yWin: {
                    type: 'linear',
                    position: 'left',
                    min: 0,
                    max: 100,
                    grid: { color: "#26334a" },
                    ticks: { color: "#10b981", callback: v => v + "%" }
                },
                yLoss: {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: "#f43f5e" }
                }
            },
            plugins: {
                legend: { labels: { color: "#f1f5f9" } }
            }
        }
    });
}

async function stepBaseline() {
    try {
        const res = await fetch("/api/train/baseline/step", { method: "POST" });
        const data = await res.json();
        renderBoard("board-baseline", data.board, data.probs);

        document.getElementById("stat-baseline-games").innerText = data.games_played;
        document.getElementById("stat-baseline-winrate").innerText = data.win_rate + "%";
        document.getElementById("stat-baseline-draws").innerText = data.draws;
        document.getElementById("stat-baseline-loss").innerText = data.last_loss.toFixed(4);

        const statusPill = document.getElementById("baseline-status");
        if (data.game_ended) {
            statusPill.innerText = data.game_result === 1 ? "Net Won!" : data.game_result === -1 ? "Net Lost!" : "Draw!";
        } else {
            statusPill.innerText = data.turn === 1 ? "Net Turn (X)" : "Random Turn (O)";
        }

        if (data.update_happened) {
            updateBaselineChart(data.games_played, data.win_rate, data.last_loss);
        }
    } catch (err) {
        console.error("Baseline step error:", err);
    }
}

function updateBaselineChart(gameCount, winRate, loss) {
    if (!chartBaseline) return;
    chartBaseline.data.labels.push(String(gameCount));
    chartBaseline.data.datasets[0].data.push(winRate);
    chartBaseline.data.datasets[1].data.push(loss);
    if (chartBaseline.data.labels.length > 25) {
        chartBaseline.data.labels.shift();
        chartBaseline.data.datasets[0].data.shift();
        chartBaseline.data.datasets[1].data.shift();
    }
    chartBaseline.update();
}

function toggleBaselineAuto() {
    const btn = document.getElementById("btn-baseline-auto");
    if (baselineRunning) {
        clearInterval(baselineInterval);
        baselineRunning = false;
        btn.innerText = "▶ Resume Auto-Play";
        btn.classList.remove("btn-outline");
        btn.classList.add("btn-primary");
    } else {
        const speed = parseInt(document.getElementById("speed-baseline").value);
        baselineInterval = setInterval(stepBaseline, speed);
        baselineRunning = true;
        btn.innerText = "⏸ Pause";
        btn.classList.remove("btn-primary");
        btn.classList.add("btn-outline");
    }
}

document.getElementById("btn-baseline-auto").addEventListener("click", toggleBaselineAuto);
document.getElementById("btn-baseline-step").addEventListener("click", stepBaseline);

document.getElementById("btn-baseline-batch").addEventListener("click", async () => {
    const btn = document.getElementById("btn-baseline-batch");
    btn.disabled = true;
    btn.innerText = "Training...";
    try {
        const res = await fetch("/api/train/baseline/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ n_games: 20 })
        });
        const data = await res.json();
        renderBoard("board-baseline", data.board, data.probs);
        document.getElementById("stat-baseline-games").innerText = data.games_played;
        document.getElementById("stat-baseline-winrate").innerText = data.win_rate + "%";
        document.getElementById("stat-baseline-draws").innerText = data.draws;
        document.getElementById("stat-baseline-loss").innerText = data.last_loss.toFixed(4);

        if (chartBaseline && data.history_labels) {
            chartBaseline.data.labels = data.history_labels;
            chartBaseline.data.datasets[0].data = data.history_win_rate;
            chartBaseline.data.datasets[1].data = data.history_loss;
            chartBaseline.update();
        }
    } finally {
        btn.disabled = false;
        btn.innerText = "+20 Games Turbo";
    }
});

document.getElementById("btn-baseline-reset").addEventListener("click", async () => {
    if (baselineRunning) toggleBaselineAuto();
    await fetch("/api/train/baseline/reset", { method: "POST" });
    document.getElementById("stat-baseline-games").innerText = "0";
    document.getElementById("stat-baseline-winrate").innerText = "0%";
    document.getElementById("stat-baseline-draws").innerText = "0";
    document.getElementById("stat-baseline-loss").innerText = "0.0000";
    if (chartBaseline) {
        chartBaseline.data.labels = [];
        chartBaseline.data.datasets[0].data = [];
        chartBaseline.data.datasets[1].data = [];
        chartBaseline.update();
    }
    renderBoard("board-baseline", [[0,0,0],[0,0,0],[0,0,0]]);
});

document.getElementById("speed-baseline").addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    document.getElementById("speed-label-baseline").innerText = val < 200 ? "Fast" : val < 500 ? "Medium" : "Slow";
    if (baselineRunning) {
        clearInterval(baselineInterval);
        baselineInterval = setInterval(stepBaseline, val);
    }
});

document.getElementById("btn-baseline-save").addEventListener("click", async () => {
    const res = await fetch("/api/model/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "baseline" })
    });
    const d = await res.json();
    alert("Model successfully saved to " + d.path);
});


// ==========================================
// MODE 2: SELF-PLAY TRAINING
// ==========================================
function initSelfplayChart() {
    const ctx = document.getElementById("chart-selfplay").getContext("2d");
    chartSelfplay = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Draw Rate % (Defense)",
                    data: [],
                    borderColor: "#fbbf24",
                    backgroundColor: "rgba(251, 191, 36, 0.1)",
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: "X Win Rate %",
                    data: [],
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.1)",
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: "O Win Rate %",
                    data: [],
                    borderColor: "#f43f5e",
                    backgroundColor: "rgba(244, 63, 94, 0.1)",
                    borderWidth: 2,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: "#26334a" },
                    ticks: { color: "#94a3b8", maxTicksLimit: 8 }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: "#26334a" },
                    ticks: { color: "#94a3b8", callback: v => v + "%" }
                }
            },
            plugins: {
                legend: { labels: { color: "#f1f5f9" } }
            }
        }
    });
}

async function stepSelfplay() {
    try {
        const res = await fetch("/api/train/selfplay/step", { method: "POST" });
        const data = await res.json();
        renderBoard("board-selfplay", data.board, data.probs);

        document.getElementById("stat-selfplay-games").innerText = data.games_played;
        document.getElementById("stat-selfplay-drawrate").innerText = data.draw_rate + "%";
        document.getElementById("stat-selfplay-xwin").innerText = data.x_win_rate + "%";
        document.getElementById("stat-selfplay-owin").innerText = data.o_win_rate + "%";

        const statusPill = document.getElementById("selfplay-status");
        if (data.game_ended) {
            statusPill.innerText = data.game_result === 1 ? "X Won!" : data.game_result === -1 ? "O Won!" : "Tied (Optimal Defense)!";
        } else {
            statusPill.innerText = data.turn === 1 ? "Self-Play: X Thinking" : "Self-Play: O Thinking";
        }

        if (data.update_happened && chartSelfplay) {
            chartSelfplay.data.labels.push(String(data.games_played));
            chartSelfplay.data.datasets[0].data.push(data.draw_rate);
            chartSelfplay.data.datasets[1].data.push(data.x_win_rate);
            chartSelfplay.data.datasets[2].data.push(data.o_win_rate);
            if (chartSelfplay.data.labels.length > 25) {
                chartSelfplay.data.labels.shift();
                chartSelfplay.data.datasets[0].data.shift();
                chartSelfplay.data.datasets[1].data.shift();
                chartSelfplay.data.datasets[2].data.shift();
            }
            chartSelfplay.update();
        }
    } catch (err) {
        console.error("Self-play step error:", err);
    }
}

function toggleSelfplayAuto() {
    const btn = document.getElementById("btn-selfplay-auto");
    if (selfplayRunning) {
        clearInterval(selfplayInterval);
        selfplayRunning = false;
        btn.innerText = "▶ Resume Self-Play";
        btn.classList.remove("btn-outline");
        btn.classList.add("btn-primary");
    } else {
        const speed = parseInt(document.getElementById("speed-selfplay").value);
        selfplayInterval = setInterval(stepSelfplay, speed);
        selfplayRunning = true;
        btn.innerText = "⏸ Pause";
        btn.classList.remove("btn-primary");
        btn.classList.add("btn-outline");
    }
}

document.getElementById("btn-selfplay-auto").addEventListener("click", toggleSelfplayAuto);
document.getElementById("btn-selfplay-step").addEventListener("click", stepSelfplay);

document.getElementById("btn-selfplay-batch").addEventListener("click", async () => {
    const btn = document.getElementById("btn-selfplay-batch");
    btn.disabled = true;
    btn.innerText = "Training...";
    try {
        const res = await fetch("/api/train/selfplay/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ n_games: 20 })
        });
        const data = await res.json();
        renderBoard("board-selfplay", data.board, data.probs);
        document.getElementById("stat-selfplay-games").innerText = data.games_played;
        document.getElementById("stat-selfplay-drawrate").innerText = data.draw_rate + "%";
        document.getElementById("stat-selfplay-xwin").innerText = data.x_win_rate + "%";
        document.getElementById("stat-selfplay-owin").innerText = data.o_win_rate + "%";

        if (chartSelfplay && data.history_labels) {
            chartSelfplay.data.labels = data.history_labels;
            chartSelfplay.data.datasets[0].data = data.history_draw_rate;
            chartSelfplay.data.datasets[1].data = data.history_x_win_rate;
            chartSelfplay.data.datasets[2].data = data.history_o_win_rate;
            chartSelfplay.update();
        }
    } finally {
        btn.disabled = false;
        btn.innerText = "+20 Games Turbo";
    }
});

document.getElementById("btn-selfplay-reset").addEventListener("click", async () => {
    if (selfplayRunning) toggleSelfplayAuto();
    await fetch("/api/train/selfplay/reset", { method: "POST" });
    document.getElementById("stat-selfplay-games").innerText = "0";
    document.getElementById("stat-selfplay-drawrate").innerText = "0%";
    document.getElementById("stat-selfplay-xwin").innerText = "0%";
    document.getElementById("stat-selfplay-owin").innerText = "0%";
    if (chartSelfplay) {
        chartSelfplay.data.labels = [];
        chartSelfplay.data.datasets[0].data = [];
        chartSelfplay.data.datasets[1].data = [];
        chartSelfplay.data.datasets[2].data = [];
        chartSelfplay.update();
    }
    renderBoard("board-selfplay", [[0,0,0],[0,0,0],[0,0,0]]);
});

document.getElementById("speed-selfplay").addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    document.getElementById("speed-label-selfplay").innerText = val < 200 ? "Fast" : val < 500 ? "Medium" : "Slow";
    if (selfplayRunning) {
        clearInterval(selfplayInterval);
        selfplayInterval = setInterval(stepSelfplay, val);
    }
});

document.getElementById("btn-selfplay-save").addEventListener("click", async () => {
    const res = await fetch("/api/model/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "selfplay" })
    });
    const d = await res.json();
    alert("Self-play model successfully saved to " + d.path);
});


// ==========================================
// MODE 3: PLAY VS SAVED MODEL
// ==========================================
async function startSavedGame() {
    const res = await fetch("/api/play/saved/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: savedSymbol })
    });
    const data = await res.json();
    handleSavedGameState(data);
}

function handleSavedGameState(data) {
    const banner = document.getElementById("saved-game-status");
    banner.className = "game-banner";

    if (data.game_ended) {
        const userCode = savedSymbol === "X" ? 1 : -1;
        if (data.game_result === userCode) {
            banner.innerText = "🎉 You Won! Excellent move!";
            banner.classList.add("win");
        } else if (data.game_result === -userCode) {
            banner.innerText = "🤖 The AI Won! Better luck next time.";
            banner.classList.add("loss");
        } else {
            banner.innerText = "🤝 It's a Draw!";
            banner.classList.add("draw");
        }
        renderBoard("board-saved", data.board, {}, false);
    } else {
        banner.innerText = "Your turn! Click an empty square.";
        renderBoard("board-saved", data.board, data.probs, true, onSavedCellClick);
    }

    if (data.scores) {
        document.getElementById("score-saved-user").innerText = data.scores.user;
        document.getElementById("score-saved-ai").innerText = data.scores.ai;
        document.getElementById("score-saved-draw").innerText = data.scores.draw;
    }
}

async function onSavedCellClick(row, col) {
    const banner = document.getElementById("saved-game-status");
    banner.innerText = "AI is thinking...";
    const res = await fetch("/api/play/saved/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row, col })
    });
    const data = await res.json();
    if (data.error) {
        banner.innerText = data.error;
        return;
    }
    handleSavedGameState(data);
}

document.getElementById("btn-saved-symbol-x").addEventListener("click", () => {
    savedSymbol = "X";
    document.getElementById("btn-saved-symbol-x").classList.add("active");
    document.getElementById("btn-saved-symbol-o").classList.remove("active");
    startSavedGame();
});

document.getElementById("btn-saved-symbol-o").addEventListener("click", () => {
    savedSymbol = "O";
    document.getElementById("btn-saved-symbol-o").classList.add("active");
    document.getElementById("btn-saved-symbol-x").classList.remove("active");
    startSavedGame();
});

document.getElementById("btn-saved-restart").addEventListener("click", startSavedGame);


// ==========================================
// MODE 4: PLAY & TEACH MODEL LIVE
// ==========================================
async function startLiveGame() {
    const res = await fetch("/api/play/live-learning/new", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: liveSymbol })
    });
    const data = await res.json();
    handleLiveGameState(data);
}

function handleLiveGameState(data) {
    const banner = document.getElementById("live-game-status");
    banner.className = "game-banner";

    if (data.game_ended) {
        const userCode = liveSymbol === "X" ? 1 : -1;
        if (data.game_result === userCode) {
            banner.innerText = "🎉 You Won! The model received negative reward and is updating its weights...";
            banner.classList.add("win");
        } else if (data.game_result === -userCode) {
            banner.innerText = "🤖 The AI Won! The model received positive reward and reinforced its winning moves!";
            banner.classList.add("loss");
        } else {
            banner.innerText = "🤝 Draw! The model is adjusting its defensive strategy.";
            banner.classList.add("draw");
        }
        renderBoard("board-live", data.board, {}, false);
    } else {
        banner.innerText = "Your turn! Click an empty square.";
        renderBoard("board-live", data.board, data.probs, true, onLiveCellClick);
    }

    if (data.stats) {
        document.getElementById("score-live-games").innerText = data.stats.games_played;
        document.getElementById("score-live-user").innerText = data.stats.user_wins;
        document.getElementById("score-live-ai").innerText = data.stats.ai_wins;
        document.getElementById("score-live-draw").innerText = data.stats.draws;
    }

    if (data.learning_update) {
        const up = data.learning_update;
        document.getElementById("brain-status-pill").innerText = "Updated!";
        document.getElementById("brain-status-pill").className = "status-pill";
        document.getElementById("brain-update-details").innerHTML = `
            <div class="brain-stat-row">
                <span>Reward Assigned:</span>
                <strong style="color: ${up.reward > 0 ? '#10b981' : '#f43f5e'}">${up.reward > 0 ? '+' + up.reward : up.reward}</strong>
            </div>
            <div class="brain-stat-row">
                <span>Backprop Loss:</span>
                <strong>${up.loss.toFixed(4)}</strong>
            </div>
            <div class="brain-stat-row">
                <span>Moves Backpropagated:</span>
                <strong>${up.moves_learned} moves</strong>
            </div>
            <div class="brain-status-msg">
                🧠 ${up.status}
            </div>
        `;
    }
}

async function onLiveCellClick(row, col) {
    const banner = document.getElementById("live-game-status");
    banner.innerText = "AI is thinking...";
    const res = await fetch("/api/play/live-learning/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row, col })
    });
    const data = await res.json();
    if (data.error) {
        banner.innerText = data.error;
        return;
    }
    handleLiveGameState(data);
}

document.getElementById("btn-live-symbol-x").addEventListener("click", () => {
    liveSymbol = "X";
    document.getElementById("btn-live-symbol-x").classList.add("active");
    document.getElementById("btn-live-symbol-o").classList.remove("active");
    startLiveGame();
});

document.getElementById("btn-live-symbol-o").addEventListener("click", () => {
    liveSymbol = "O";
    document.getElementById("btn-live-symbol-o").classList.add("active");
    document.getElementById("btn-live-symbol-x").classList.remove("active");
    startLiveGame();
});

document.getElementById("btn-live-restart").addEventListener("click", startLiveGame);

document.getElementById("btn-live-reset-scratch").addEventListener("click", async () => {
    if (confirm("Reset model brain to completely random initial weights?")) {
        await fetch("/api/play/live-learning/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ load_checkpoint: false })
        });
        document.getElementById("brain-update-details").innerHTML = `<p class="placeholder-text">Model brain reset to fresh random weights. Play to train it from scratch!</p>`;
        document.getElementById("brain-status-pill").innerText = "Fresh Brain";
        startLiveGame();
    }
});

document.getElementById("btn-live-reset-ckpt").addEventListener("click", async () => {
    await fetch("/api/play/live-learning/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ load_checkpoint: true })
    });
    document.getElementById("brain-update-details").innerHTML = `<p class="placeholder-text">Loaded saved model weights from model.npz. It will now fine-tune live against you!</p>`;
    document.getElementById("brain-status-pill").innerText = "Model Loaded";
    startLiveGame();
});

document.getElementById("btn-live-save").addEventListener("click", async () => {
    const res = await fetch("/api/model/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "live" })
    });
    const d = await res.json();
    alert("Live-trained model successfully saved to " + d.path);
});


// Initialization on load
window.addEventListener("DOMContentLoaded", () => {
    initBaselineChart();
    initSelfplayChart();
    renderBoard("board-baseline", [[0,0,0],[0,0,0],[0,0,0]]);
    renderBoard("board-selfplay", [[0,0,0],[0,0,0],[0,0,0]]);
});
