// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  event.currentTarget.classList.add('active');
  const target = document.getElementById(`tab-${tabId}`);
  if (target) target.classList.add('active');

  if (tabId === 'playground') {
    updateBoundaryPlot();
  }
}

// Preset vector loader
function loadPreset(type) {
  const uInput = document.getElementById('vec-u');
  const vInput = document.getElementById('vec-v');

  if (type === 'collinear') {
    uInput.value = "1.0, 2.0, 3.0";
    vInput.value = "3.0, 6.0, 9.0";
  } else if (type === 'orthogonal') {
    uInput.value = "1.0, 0.0, 0.0";
    vInput.value = "0.0, 1.0, 0.0";
  } else if (type === 'outlier') {
    uInput.value = "1.0, 2.0, 1.5, 2.2";
    vInput.value = "1.1, 2.1, 1.6, 25.0";
  }
  calculateDistance();
}

// Live vector calculation
async function calculateDistance() {
  const uStr = document.getElementById('vec-u').value;
  const vStr = document.getElementById('vec-v').value;

  const u = uStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
  const v = vStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));

  if (u.length === 0 || v.length === 0 || u.length !== v.length) {
    alert("Please enter valid comma-separated numbers of identical dimension!");
    return;
  }

  try {
    const res = await fetch('/api/compute_distance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ u, v })
    });
    const data = await res.json();
    if (data.success) {
      const r = data.results;
      document.getElementById('res-euc').innerText = r.euclidean.value;
      document.getElementById('res-euc-formula').innerText = `Diff sq sum: ${r.euclidean.sum_squared_diff} | Dim: ${r.dim}`;

      document.getElementById('res-man').innerText = r.manhattan.value;
      document.getElementById('res-man-formula').innerText = `Abs diff sum: ${r.manhattan.sum_abs_diff} | Dim: ${r.dim}`;

      document.getElementById('res-cos').innerText = `Dist: ${r.cosine.distance}`;
      document.getElementById('res-cos-formula').innerText = `Similarity: ${r.cosine.similarity} | Angle θ: ${r.cosine.angle_degrees}° | Dot: ${r.cosine.dot_product}`;
    }
  } catch (err) {
    console.error(err);
  }
}

// 2D Decision Boundary Canvas Renderer
async function updateBoundaryPlot() {
  const dataset = document.getElementById('play-dataset').value;
  const metric = document.getElementById('play-metric').value;
  const k = document.getElementById('play-k').value;

  const canvas = document.getElementById('boundaryCanvas');
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  // Clear canvas
  ctx.fillStyle = '#060911';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#9ca3af';
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Computing decision boundary topology...', width / 2, height / 2);

  try {
    const res = await fetch(`/api/boundary_data?dataset=${dataset}&metric=${metric}&k=${k}`);
    const data = await res.json();

    if (!data.success) return;

    const { bounds, grid_res, grid_preds, points, target_names } = data;
    const { x_min, x_max, y_min, y_max } = bounds;

    // Coordinate conversion helpers
    const scaleX = (x) => ((x - x_min) / (x_max - x_min)) * width;
    const scaleY = (y) => height - ((y - y_min) / (y_max - y_min)) * height;

    // Color palettes for classes
    const classBgColors = [
      'rgba(59, 130, 246, 0.35)',
      'rgba(239, 68, 68, 0.35)',
      'rgba(16, 185, 129, 0.35)',
      'rgba(245, 158, 11, 0.35)'
    ];
    const classPointColors = ['#60a5fa', '#f87171', '#34d399', '#fbbf24'];

    ctx.clearRect(0, 0, width, height);

    // Draw grid cells (Decision Regions)
    const cellW = width / grid_res;
    const cellH = height / grid_res;

    for (let r = 0; r < grid_res; r++) {
      for (let c = 0; c < grid_res; c++) {
        const predClass = grid_preds[r * grid_res + c];
        ctx.fillStyle = classBgColors[predClass % classBgColors.length];
        // Note: r is y-step, c is x-step
        const px = c * cellW;
        const py = height - (r + 1) * cellH;
        ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
      }
    }

    // Draw Scatter points
    for (const pt of points) {
      const cx = scaleX(pt.x);
      const cy = scaleY(pt.y);

      ctx.beginPath();
      ctx.arc(cx, cy, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = classPointColors[pt.label % classPointColors.length];
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#000';
      ctx.stroke();
    }

    // Update legend
    const legendDiv = document.getElementById('canvas-legend');
    legendDiv.innerHTML = target_names.map((name, i) => `
      <span style="display:flex; align-items:center; gap:6px;">
        <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:${classPointColors[i % classPointColors.length]};"></span>
        <span style="color:#d1d5db;">${name}</span>
      </span>
    `).join('');

  } catch (err) {
    console.error(err);
  }
}

// Semantic text search
async function executeSearch() {
  const query = document.getElementById('search-query').value;
  const metric = document.getElementById('search-metric').value;
  const box = document.getElementById('search-results-box');

  box.innerHTML = '<p style="color:var(--accent-cyan);">Searching document vectors...</p>';

  try {
    const res = await fetch('/api/similarity_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, metric, top_k: 5 })
    });
    const data = await res.json();
    if (data.success) {
      box.innerHTML = data.results.map(r => `
        <div style="background:rgba(0,0,0,0.3); border:1px solid var(--border-color); border-radius:8px; padding:10px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <strong style="color:var(--accent-cyan);">Rank #${r.rank} &bull; Category: ${r.metadata.category || 'General'}</strong>
            <span style="font-family:monospace; color:#fbbf24;">Score: ${(r.similarity * 100).toFixed(1)}%</span>
          </div>
          <div style="font-size:0.82rem; color:var(--text-secondary);">Distance: ${r.distance.toFixed(4)}</div>
          <div style="background:rgba(255,255,255,0.08); height:6px; border-radius:3px; margin-top:6px; overflow:hidden;">
            <div style="background:linear-gradient(90deg, #3b82f6, #10b981); height:100%; width:${Math.min(100, Math.max(0, r.similarity * 100))}%;"></div>
          </div>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error(err);
    box.innerHTML = '<p style="color:#f87171;">Error executing search.</p>';
  }
}

// Initial calculation on load
window.addEventListener('DOMContentLoaded', () => {
  calculateDistance();
});
