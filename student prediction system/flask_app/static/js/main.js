// ── State ──
let currentPage = 1;
let statsData = null;

// ── Navigation ──
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        item.classList.add('active');
        document.getElementById(item.dataset.page).classList.add('active');

        // Load page-specific data
        const page = item.dataset.page;
        if (page === 'overview') loadOverview();
        else if (page === 'dataset') loadDataset(1);
        else if (page === 'visualizations') loadCharts();
        else if (page === 'performance') loadPerformance();
    });
});

// ── Overview ──
async function loadOverview() {
    if (statsData) return updateOverviewUI(statsData);
    const res = await fetch('/api/stats');
    statsData = await res.json();
    updateOverviewUI(statsData);
}

function updateOverviewUI(data) {
    document.getElementById('stat-records').textContent = data.dataset.total_records.toLocaleString();
    document.getElementById('stat-features').textContent = data.dataset.features;
    document.getElementById('stat-missing').textContent = data.dataset.missing_values;
    document.getElementById('stat-r2').textContent = data.metrics.r2_pct + '%';
    document.getElementById('stat-mae').textContent = data.metrics.mae;
    document.getElementById('stat-train').textContent = data.metrics.train_size.toLocaleString();
    document.getElementById('stat-test').textContent = data.metrics.test_size.toLocaleString();
    document.getElementById('stat-rmse').textContent = data.metrics.rmse;

    // Mini correlation chart
    const corr = data.correlation.FinalScore;
    const features = Object.keys(corr).filter(k => k !== 'FinalScore');
    const values = features.map(f => corr[f]);
    createBarChart('corrChart', features, values, 'Correlation with FinalScore',
        values.map(v => v > 0.5 ? '#10b981' : v > 0.2 ? '#3b82f6' : '#f59e0b'));
}

// ── Dataset ──
async function loadDataset(page) {
    currentPage = page;
    const search = document.getElementById('searchInput')?.value || '';
    const res = await fetch(`/api/dataset?page=${page}&per_page=15&search=${search}`);
    const data = await res.json();

    const tbody = document.getElementById('datasetBody');
    tbody.innerHTML = data.data.map((row, i) => `
        <tr>
            <td>${(page - 1) * 15 + i + 1}</td>
            <td>${row.StudyHours}</td>
            <td>${row.Attendance}</td>
            <td>${row.SleepHours}</td>
            <td>${row.PreviousScore}</td>
            <td style="font-weight:700;color:var(--accent)">${row.FinalScore}</td>
        </tr>
    `).join('');

    const pag = document.getElementById('pagination');
    let html = `<button onclick="loadDataset(1)" ${page === 1 ? 'disabled' : ''}>&laquo;</button>`;
    html += `<button onclick="loadDataset(${Math.max(1, page - 1)})">&lsaquo;</button>`;
    const start = Math.max(1, page - 2), end = Math.min(data.pages, page + 2);
    for (let i = start; i <= end; i++)
        html += `<button class="${i === page ? 'active' : ''}" onclick="loadDataset(${i})">${i}</button>`;
    html += `<button onclick="loadDataset(${Math.min(data.pages, page + 1)})">&rsaquo;</button>`;
    html += `<button onclick="loadDataset(${data.pages})">&raquo;</button>`;
    html += `<span>Page ${page} of ${data.pages} (${data.total} records)</span>`;
    pag.innerHTML = html;
}

// ── Charts ──
const chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function createBarChart(id, labels, data, label, colors) {
    destroyChart(id);
    const ctx = document.getElementById(id)?.getContext('2d');
    if (!ctx) return;
    chartInstances[id] = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label, data, backgroundColor: colors || 'rgba(99,102,241,0.6)', borderRadius: 6 }] },
        options: { responsive: true, plugins: { legend: { display: false } },
            scales: { y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                      x: { grid: { display: false }, ticks: { color: '#64748b' } } } }
    });
}

async function loadCharts() {
    // Scatter
    const scatterRes = await fetch('/api/charts/scatter');
    const scatter = await scatterRes.json();
    destroyChart('scatterChart');
    const sctx = document.getElementById('scatterChart').getContext('2d');
    const scatterData = scatter.x.map((x, i) => ({ x, y: scatter.y[i] }));
    chartInstances['scatterChart'] = new Chart(sctx, {
        type: 'scatter',
        data: { datasets: [{ label: 'Students', data: scatterData,
            backgroundColor: scatter.colors.map(c => `hsla(${(c - 40) * 2}, 70%, 60%, 0.5)`),
            pointRadius: 3 }] },
        options: { responsive: true, plugins: { legend: { display: false } },
            scales: { x: { title: { display: true, text: 'Study Hours', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                      y: { title: { display: true, text: 'Final Score', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } } } }
    });

    // Distribution
    const distRes = await fetch('/api/charts/distribution');
    const dist = await distRes.json();
    const colors = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6'];
    ['StudyHours', 'Attendance', 'SleepHours', 'PreviousScore', 'FinalScore'].forEach((col, i) => {
        const id = `dist${col}`;
        destroyChart(id);
        const ctx = document.getElementById(id)?.getContext('2d');
        if (!ctx) return;
        const d = dist[col];
        const labels = d.edges.slice(0, -1).map((e, j) => `${e}-${d.edges[j + 1]}`);
        chartInstances[id] = new Chart(ctx, {
            type: 'bar',
            data: { labels, datasets: [{ data: d.counts, backgroundColor: colors[i] + '99', borderRadius: 4 }] },
            options: { responsive: true, plugins: { legend: { display: false } },
                scales: { y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                          x: { display: false } } }
        });
    });

    // Actual vs Predicted
    const avpRes = await fetch('/api/charts/actual_vs_predicted');
    const avp = await avpRes.json();
    destroyChart('avpChart');
    const actx = document.getElementById('avpChart').getContext('2d');
    const avpData = avp.actual.map((a, i) => ({ x: a, y: avp.predicted[i] }));
    chartInstances['avpChart'] = new Chart(actx, {
        type: 'scatter',
        data: { datasets: [
            { label: 'Predictions', data: avpData, backgroundColor: 'rgba(99,102,241,0.4)', pointRadius: 2.5 },
            { label: 'Perfect', data: [{ x: 35, y: 35 }, { x: 105, y: 105 }], type: 'line',
              borderColor: '#ef4444', borderDash: [6, 4], pointRadius: 0, borderWidth: 2 }
        ] },
        options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: { x: { title: { display: true, text: 'Actual', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
                      y: { title: { display: true, text: 'Predicted', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } } } }
    });

    // Feature Importance
    const fiRes = await fetch('/api/charts/feature_importance');
    const fi = await fiRes.json();
    createBarChart('featureChart', fi.features, fi.values, 'Coefficient Value',
        ['#6366f1', '#10b981', '#f59e0b', '#ec4899']);
}

// ── Performance ──
async function loadPerformance() {
    if (!statsData) {
        const res = await fetch('/api/stats');
        statsData = await res.json();
    }
    document.getElementById('perf-r2').textContent = statsData.metrics.r2_pct + '%';
    document.getElementById('perf-mae').textContent = statsData.metrics.mae;
    document.getElementById('perf-mse').textContent = statsData.metrics.mse;
    document.getElementById('perf-rmse').textContent = statsData.metrics.rmse;

    const tbody = document.getElementById('coeffBody');
    const feats = statsData.coefficients.features;
    tbody.innerHTML = Object.entries(feats).map(([f, v]) =>
        `<tr><td class="feat">${f}</td><td class="coeff-val">${v}</td>
         <td style="color:${v > 1 ? '#10b981' : '#3b82f6'}">${v > 1 ? 'High Impact' : 'Moderate Impact'}</td></tr>`
    ).join('') + `<tr><td class="feat">Intercept</td><td class="coeff-val">${statsData.coefficients.intercept}</td><td style="color:var(--muted)">Base Score</td></tr>`;
}

// ── Prediction ──
async function makePrediction() {
    const data = {
        study_hours: document.getElementById('inp-study').value,
        attendance: document.getElementById('inp-attend').value,
        sleep_hours: document.getElementById('inp-sleep').value,
        previous_score: document.getElementById('inp-prev').value
    };

    const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    const result = await res.json();

    if (result.success) {
        document.getElementById('resultArea').classList.remove('result-hidden');
        document.getElementById('predScore').textContent = result.prediction;
        document.getElementById('predScore').style.color = result.color;
        document.getElementById('predLevel').textContent = result.level;
        document.getElementById('predLevel').style.color = result.color;

        const contribs = result.contributions;
        const maxC = Math.max(...Object.values(contribs).map(Math.abs));
        const contribDiv = document.getElementById('contributions');
        contribDiv.innerHTML = Object.entries(contribs).map(([k, v]) => `
            <div class="contrib-bar">
                <div class="bar-label"><span>${k}</span><span style="color:var(--accent)">${v}</span></div>
                <div class="bar-track"><div class="bar-fill" style="width:${Math.abs(v) / maxC * 100}%;background:${v >= 0 ? '#10b981' : '#ef4444'}"></div></div>
            </div>
        `).join('');
    }
}

// Slider value display
document.querySelectorAll('input[type=range]').forEach(input => {
    const valSpan = input.parentElement.querySelector('.slider-val');
    if (valSpan) { input.addEventListener('input', () => valSpan.textContent = input.value); }
});

// Search
document.getElementById('searchInput')?.addEventListener('input', () => loadDataset(1));

// ── Init ──
loadOverview();
