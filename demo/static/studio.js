// TrustFed-SDN Interactive Studio Controller (Option A + Option B)

document.addEventListener("DOMContentLoaded", () => {
  const html = document.documentElement;
  const brandLogos = document.querySelectorAll("[data-brand-logo]");
  const applyLogos = (theme) => {
    const src = theme === "light" ? "images/logo-gradient.png" : "images/logo.png";
    brandLogos.forEach((img) => { img.src = src; });
  };
  applyLogos(html.getAttribute("data-theme") || "dark");

  document.getElementById("themeToggleBtn")?.addEventListener("click", () => {
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    applyLogos(next);
  });

  /* View Switcher (Option A <-> Option B <-> Option C) */
  const switchBtns = document.querySelectorAll(".switch-btn[data-view]");
  const viewA = document.getElementById("viewOptionA");
  const viewB = document.getElementById("viewOptionB");
  const viewC = document.getElementById("viewOptionC");
  function activateView(target) {
    switchBtns.forEach((b) => {
      const on = b.dataset.view === target;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", String(on));
    });
    viewA?.classList.toggle("active", target === "a");
    viewB?.classList.toggle("active", target === "b");
    viewC?.classList.toggle("active", target === "c");
    if (target === "a") resizeCanvas();
    if (target === "c") refreshVmStatus();
  }
  switchBtns.forEach((btn) => {
    btn.addEventListener("click", () => activateView(btn.dataset.view));
  });

  /* Subtabs (OVS Flows / Event Log) */
  document.querySelectorAll(".mini-tab[data-subtab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.subtab;
      document.querySelectorAll(".mini-tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.getElementById("subtabFlows").classList.toggle("active", target === "flows");
      document.getElementById("subtabLogs").classList.toggle("active", target === "logs");
    });
  });

  /* Live Benign vs Attacker Traffic Canvas Chart */
  const canvas = document.getElementById("trafficCanvas");
  const ctx = canvas ? canvas.getContext("2d") : null;
  const historyLen = 36;
  const benignSeries = new Array(historyLen).fill(0);
  const attackSeries = new Array(historyLen).fill(0);
  let targetBenignPps = 0;
  let targetAttackPps = 0;

  function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      canvas.width = Math.floor(rect.width * (window.devicePixelRatio || 1));
      canvas.height = Math.floor(rect.height * (window.devicePixelRatio || 1));
    }
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function drawChart() {
    if (!ctx || !canvas.width || !canvas.height) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const isLight = html.getAttribute("data-theme") === "light";
    ctx.strokeStyle = isLight ? "#E2E8F0" : "#222938";
    ctx.lineWidth = 1 * dpr;
    for (let i = 1; i <= 3; i += 1) {
      const y = (h * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const drawLine = (series, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2 * dpr;
      ctx.beginPath();
      series.forEach((val, idx) => {
        const x = (idx / (historyLen - 1)) * (w - 16 * dpr) + 8 * dpr;
        const y = h - 10 * dpr - (Math.min(100, val) / 100) * (h - 22 * dpr);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };
    drawLine(benignSeries, isLight ? "#059669" : "#10B981");
    drawLine(attackSeries, isLight ? "#BE123C" : "#F43F5E");
  }

  setInterval(() => {
    const benJitter = targetBenignPps > 0 ? (Math.random() - 0.5) * 4 : 0;
    const atkJitter = targetAttackPps > 0 ? (Math.random() - 0.5) * 10 : 0;
    benignSeries.push(Math.max(0, targetBenignPps + benJitter));
    benignSeries.shift();
    attackSeries.push(Math.max(0, targetAttackPps + atkJitter));
    attackSeries.shift();
    drawChart();
  }, 450);

  /* Animated SVG Packet Particles */
  const packetLayer = document.getElementById("packetLayer");
  const NODES = {
    fl: [410, 32],
    c1: [145, 108], c2: [410, 108], c3: [675, 108],
    s0: [410, 158], s0_l: [336, 162], s0_r: [484, 162],
    s1: [145, 203], s2: [410, 203], s3: [675, 203],
    h1_ben: [72, 290], h1_vic: [145, 290], h1_atk: [218, 290],
    h2_ben: [337, 290], h2_vic: [410, 290], h2_atk: [483, 290],
    h3_ben: [602, 290], h3_vic: [675, 290], h3_atk: [748, 290],
  };

  function spawnPacketBurst(waypoints, color, count, durMs) {
    if (!packetLayer) return;
    for (let i = 0; i < count; i += 1) {
      setTimeout(() => {
        const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        dot.setAttribute("r", "4.3");
        dot.setAttribute("fill", color);
        dot.setAttribute("filter", color === "#10B981" ? "url(#glowGreen)" : "url(#glowRed)");
        packetLayer.appendChild(dot);

        const segs = waypoints.length - 1;
        const tStart = performance.now();
        const anim = (now) => {
          const t = Math.min(1, (now - tStart) / durMs);
          const scaled = t * segs;
          const idx = Math.min(segs - 1, Math.floor(scaled));
          const localT = scaled - idx;
          const [x1, y1] = NODES[waypoints[idx]];
          const [x2, y2] = NODES[waypoints[idx + 1]];
          dot.setAttribute("cx", String(x1 + (x2 - x1) * localT));
          dot.setAttribute("cy", String(y1 + (y2 - y1) * localT));
          if (t < 1) requestAnimationFrame(anim);
          else dot.remove();
        };
        requestAnimationFrame(anim);
      }, i * 160);
    }
  }

  let ambientTimer = null;
  function startAmbientTraffic(phase) {
    if (ambientTimer) clearInterval(ambientTimer);
    ambientTimer = setInterval(() => {
      if (phase >= 3) {
        spawnPacketBurst(["h1_ben", "s1", "h1_vic"], "#10B981", 1, 700);
        spawnPacketBurst(["h2_ben", "s2", "h2_vic"], "#10B981", 1, 700);
        spawnPacketBurst(["h3_ben", "s3", "h3_vic"], "#10B981", 1, 700);
      }
      if (phase === 3 || phase === 4) {
        spawnPacketBurst(["h1_atk", "s1", "h1_vic"], "#F43F5E", 2, 520);
        spawnPacketBurst(["h2_atk", "s2", "h2_vic"], "#F43F5E", 2, 520);
        spawnPacketBurst(["h3_atk", "s3", "h3_vic"], "#F43F5E", 2, 520);
      }
      if (phase >= 5) {
        spawnPacketBurst(["h1_atk", "s1"], "#F43F5E", 1, 420);
        spawnPacketBurst(["h2_atk", "s2"], "#F43F5E", 1, 420);
        spawnPacketBurst(["h3_atk", "s3"], "#F43F5E", 1, 420);
      }
    }, 850);
  }

  /* Topology State Helper */
  const allNodeIds = Object.keys(NODES).filter((k) => document.getElementById(`node_${k}`));
  function setTopologyOnline(online) {
    allNodeIds.forEach((id) => {
      const el = document.getElementById(`node_${id}`);
      if (!el) return;
      el.classList.toggle("node-online", online);
      el.classList.toggle("node-offline", !online);
    });
  }

  function appendLogs(lines) {
    const feed = document.getElementById("eventLogFeed");
    if (!feed || !Array.isArray(lines)) return;
    lines.forEach((txt) => {
      const div = document.createElement("div");
      div.className = "log-line";
      div.textContent = txt;
      feed.prepend(div);
    });
  }

  function renderFlowsTable(flows) {
    const tbody = document.getElementById("ovsFlowsBody");
    if (!tbody || !Array.isArray(flows)) return;
    tbody.innerHTML = flows
      .map((f) => {
        const isDrop = String(f.action).includes("DROP") || String(f.action).includes("FLOOD") || String(f.action).includes("FLAGGED");
        const cls = isDrop ? "badge-bad" : "badge-ok";
        return `<tr><td>${f.switch}</td><td>${f.priority}</td><td>${f.match}</td><td class="${cls}">${f.action}</td><td>${f.packets}</td></tr>`;
      })
      .join("");
  }

  function renderFeatureBars(containerId, items) {
    const box = document.getElementById(containerId);
    if (!box) return;
    if (!items || !items.length) {
      box.innerHTML = `<div class="muted-cell">No feature attributions available.</div>`;
      return;
    }
    const maxImp = Math.max(...items.map((x) => x.importance_pct), 1);
    box.innerHTML = items
      .map((it) => {
        const w = Math.max(4, Math.round((it.importance_pct / maxImp) * 100));
        return `<div class="bar-row">
          <div class="bar-labels"><span>${it.feature} (${it.mean_value})</span><span>${it.importance_pct}%</span></div>
          <div class="bar-bg"><div class="bar-fg" style="width:${w}%"></div></div>
        </div>`;
      })
      .join("");
  }

  /* Sequential Step Controller */
  let completedStep = 0;
  let busy = false;

  function syncStepCards() {
    document.getElementById("stepProgressText").textContent = `${completedStep} / 5 Done`;
    document.querySelectorAll(".step-card[data-step]").forEach((card) => {
      const step = Number(card.dataset.step);
      const badge = document.getElementById(`status${step}`);
      const btn = card.querySelector(".step-btn");
      const lbl = btn.querySelector(".btn-label");
      card.classList.remove("is-ready", "is-locked", "is-done", "is-processing");

      if (step <= completedStep) {
        card.classList.add("is-done");
        badge.className = "status-badge status-done";
        badge.textContent = "Done";
        lbl.textContent = `Step 0${step} Done`;
        btn.disabled = true;
      } else if (step === completedStep + 1) {
        card.classList.add("is-ready");
        badge.className = "status-badge status-ready";
        badge.textContent = "Ready";
        lbl.textContent = `Run Step 0${step}`;
        btn.disabled = busy;
      } else {
        card.classList.add("is-locked");
        badge.className = "status-badge status-locked";
        badge.textContent = "Locked";
        lbl.textContent = `Run Step 0${step}`;
        btn.disabled = true;
      }
    });
  }

  function applyStepVisuals(step, res) {
    renderFlowsTable(res.flows);
    appendLogs(res.log);

    if (step === 1) {
      setTopologyOnline(true);
      document.getElementById("collateralBadge").textContent = "Network Online (9 Hosts)";
      spawnPacketBurst(["fl", "c1", "s1"], "#38BDF8", 2, 750);
      spawnPacketBurst(["fl", "c2", "s2"], "#38BDF8", 2, 750);
      spawnPacketBurst(["fl", "c3", "s3"], "#38BDF8", 2, 750);
    } else if (step === 2) {
      document.getElementById("s0BlockMark")?.setAttribute("opacity", "1");
      document.getElementById("s0BlockMarkR")?.setAttribute("opacity", "1");
      document.getElementById("collateralBadge").textContent = "L1 Isolation: 100% (s0)";
      spawnPacketBurst(["h1_ben", "s1", "h1_vic"], "#10B981", 2, 700);
      spawnPacketBurst(["h2_ben", "s2", "h2_vic"], "#10B981", 2, 700);
      spawnPacketBurst(["h3_ben", "s3", "h3_vic"], "#10B981", 2, 700);
      spawnPacketBurst(["h1_ben", "s1", "s0_l"], "#F43F5E", 3, 680);
      spawnPacketBurst(["h2_ben", "s2", "s0"], "#F43F5E", 3, 680);
      spawnPacketBurst(["h3_ben", "s3", "s0_r"], "#F43F5E", 3, 680);
    } else if (step === 3) {
      targetBenignPps = 28;
      targetAttackPps = 88;
      startAmbientTraffic(3);
      document.getElementById("collateralBadge").textContent = "Attacks Active on s1, s2, s3";
      document.getElementById("flowCountLabel").textContent = `${res.benign_flows + res.attack_flows} flows`;
      const items = (res.sample_features || []).map((f) => ({
        feature: f.feature,
        mean_value: f.attack,
        importance_pct: f.benign > 0 ? Math.min(100, Math.round((f.attack / Math.max(1, f.benign)) * 20)) : 50,
      }));
      renderFeatureBars("featureBarsA", items);
    } else if (step === 4) {
      startAmbientTraffic(4);
      const ai = res.ai;
      const pct = Math.round(ai.max_attack_prob * 100);
      const pill = document.getElementById("verdictBannerA");
      pill.className = "verdict-pill verdict-atk";
      pill.textContent = `ATTACK DETECTED · Flag ${ai.primary_src_ip}`;
      document.getElementById("probValueA").textContent = ai.max_attack_prob.toFixed(3);
      const fill = document.getElementById("probFillA");
      fill.style.width = `${pct}%`;
      fill.style.background = "var(--accent-2)";
      renderFeatureBars("featureBarsA", ai.top_features);
      document.getElementById("flowCountLabel").textContent = `${ai.total_flows} scored`;

      const cr = res.crypto;
      document.getElementById("cryptoSummaryA").innerHTML = `
        <div class="c-row"><span>Poisoning Screen</span><strong class="badge-ok">Rejected ${res.poison.rejected.join(", ") || "malicious"}</strong></div>
        <div class="c-row"><span>Merkle + Ed25519</span><strong class="badge-ok">${cr.verified ? "VERIFIED" : "FAILED"} (${cr.merkle_root_hex.slice(0, 14)}...)</strong></div>
        <div class="c-row"><span>Wire Ratio</span><strong>${cr.wire_ratio}x (48-bit Q20)</strong></div>
      `;
      spawnPacketBurst(["h1_ben", "s1", "c1", "fl"], "#38BDF8", 2, 850);
      spawnPacketBurst(["h2_ben", "s2", "c2", "fl"], "#38BDF8", 2, 850);
      spawnPacketBurst(["h3_ben", "s3", "c3", "fl"], "#38BDF8", 2, 850);
    } else if (step === 5) {
      targetAttackPps = 0;
      targetBenignPps = 28;
      startAmbientTraffic(5);
      ["s1Shield", "s2Shield", "s3Shield"].forEach((id) => {
        document.getElementById(id)?.setAttribute("opacity", "1");
      });
      document.getElementById("ovsLatencyA").textContent = `${res.install_latency_ms} ms`;
      document.getElementById("benignLossA").textContent = "0.0% SAFE";
      document.getElementById("collateralBadge").textContent = "s1–s3 Blocked · 0% Collateral";
      spawnPacketBurst(["c1", "s1"], "#10B981", 2, 550);
      spawnPacketBurst(["c2", "s2"], "#10B981", 2, 550);
      spawnPacketBurst(["c3", "s3"], "#10B981", 2, 550);
    }
  }

  document.querySelectorAll(".step-btn[data-run-step]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const step = Number(btn.dataset.runStep);
      if (busy || step !== completedStep + 1) return;
      busy = true;
      const card = btn.closest(".step-card");
      const badge = document.getElementById(`status${step}`);
      card.classList.remove("is-ready");
      card.classList.add("is-processing");
      badge.className = "status-badge status-processing";
      badge.textContent = "Processing...";
      btn.querySelector(".btn-label").textContent = "Processing...";
      document.querySelectorAll(".step-btn").forEach((b) => { b.disabled = true; });

      try {
        const minSpin = new Promise((r) => setTimeout(r, 850));
        const apiCall = fetch("/api/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ step }),
        }).then((r) => r.json());
        const [data] = await Promise.all([apiCall, minSpin]);
        if (!data.ok) throw new Error(data.error || "Step failed");
        completedStep = data.completed_step;
        applyStepVisuals(step, data.result);
      } catch (err) {
        appendLogs([`[Error] ${err.message}`]);
      } finally {
        busy = false;
        syncStepCards();
      }
    });
  });

  document.getElementById("resetPipelineBtn")?.addEventListener("click", async () => {
    if (busy) return;
    await fetch("/api/reset", { method: "POST" });
    if (ambientTimer) clearInterval(ambientTimer);
    completedStep = 0;
    targetBenignPps = 0;
    targetAttackPps = 0;
    benignSeries.fill(0);
    attackSeries.fill(0);
    drawChart();
    setTopologyOnline(false);
    ["s0BlockMark", "s0BlockMarkR", "s1Shield", "s2Shield", "s3Shield"].forEach((id) => {
      document.getElementById(id)?.setAttribute("opacity", "0");
    });
    document.getElementById("collateralBadge").textContent = "Waiting Step 01";
    document.getElementById("verdictBannerA").className = "verdict-pill verdict-idle";
    document.getElementById("verdictBannerA").textContent = "Awaiting Traffic (Step 04)";
    document.getElementById("probValueA").textContent = "0.000";
    document.getElementById("probFillA").style.width = "0%";
    document.getElementById("ovsLatencyA").textContent = "—";
    document.getElementById("benignLossA").textContent = "—";
    document.getElementById("flowCountLabel").textContent = "0 flows";
    document.getElementById("featureBarsA").innerHTML = `<div class="muted-cell">Features extracted from real PCAPs will appear at Step 03–04.</div>`;
    document.getElementById("cryptoSummaryA").innerHTML = `
      <div class="c-row"><span>Poisoning Screen</span><strong>Pending Step 04</strong></div>
      <div class="c-row"><span>Merkle + Ed25519</span><strong>Pending Step 04</strong></div>
      <div class="c-row"><span>Wire Ratio</span><strong>—</strong></div>
    `;
    document.getElementById("ovsFlowsBody").innerHTML = `<tr><td colspan="5" class="muted-cell">Run Step 01 to initialize switches s0–s3.</td></tr>`;
    document.getElementById("eventLogFeed").innerHTML = `<div class="log-line">[Studio] Pipeline reset. Click Step 01 to start.</div>`;
    syncStepCards();
  });

  /* ============ OPTION B: AI INSPECTOR & LABS ============ */
  const threshSliderB = document.getElementById("threshSliderB");
  const threshValB = document.getElementById("threshValB");
  let lastSampleNameB = null;

  threshSliderB?.addEventListener("input", () => {
    threshValB.textContent = Number(threshSliderB.value).toFixed(2);
  });
  threshSliderB?.addEventListener("change", () => {
    if (lastSampleNameB) inspectSampleB(lastSampleNameB);
  });

  function renderOptionBResult(res) {
    document.getElementById("sourceTagB").textContent = `${res.source_name} (${res.packets_parsed} pkts)`;
    const banner = document.getElementById("verdictBannerB");
    if (res.verdict === "ATTACK") {
      banner.className = "verdict-pill verdict-atk";
      banner.textContent = `ATTACK DETECTED (${res.attack_flows}/${res.total_flows} flows) — INSTALL OVS DROP`;
    } else {
      banner.className = "verdict-pill verdict-safe";
      banner.textContent = `BENIGN TRAFFIC (${res.total_flows} flows) — PERMITTED`;
    }
    document.getElementById("bFlowsCount").textContent = res.total_flows;
    document.getElementById("bAtkCount").textContent = res.attack_flows;
    document.getElementById("bMaxProb").textContent = res.max_attack_prob.toFixed(3);
    document.getElementById("bElapsed").textContent = `${res.elapsed_ms} ms`;
    document.getElementById("bOvsRule").textContent = res.ovs_rule_preview;
    renderFeatureBars("featureBarsB", res.top_features);

    const tbody = document.getElementById("bFlowsTable");
    tbody.innerHTML = (res.sample_flows || [])
      .map((f) => {
        const cls = f.decision === "DROP" ? "badge-bad" : "badge-ok";
        return `<tr><td>${f.src_ip}</td><td>${f.dst_ip}</td><td>${f.pkts}</td><td>${f.attack_prob.toFixed(3)}</td><td class="${cls}">${f.decision}</td></tr>`;
      })
      .join("") || `<tr><td colspan="5" class="muted-cell">CSV scored directly from feature columns.</td></tr>`;
  }

  async function inspectSampleB(name) {
    lastSampleNameB = name;
    document.querySelectorAll(".sample-btn").forEach((b) => b.classList.toggle("active", b.dataset.pcap === name));
    const res = await fetch("/api/inspect-sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pcap_name: name, threshold: Number(threshSliderB.value) }),
    }).then((r) => r.json());
    if (res.ok) renderOptionBResult(res.result);
  }

  const fileUploadB = document.getElementById("fileUploadB");
  const dropZoneB = document.getElementById("dropZoneB");
  async function handleFileUpload(file) {
    if (!file) return;
    if (file.size > 15 * 1024 * 1024) {
      const banner = document.getElementById("verdictBannerB");
      if (banner) {
        banner.className = "verdict-pill verdict-atk";
        banner.textContent = "File exceeds 15 MB interactive safety cap";
      }
      return;
    }
    const buf = await file.arrayBuffer();
    const bytes = new Uint8Array(buf);
    let bin = "";
    for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
    const resp = await fetch("/api/inspect-upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        content_b64: btoa(bin),
        threshold: Number(threshSliderB.value),
      }),
    }).then((r) => r.json());
    if (resp.ok) renderOptionBResult(resp.result);
    else alert(resp.error || "Upload inspection failed");
  }
  fileUploadB?.addEventListener("change", (e) => handleFileUpload(e.target.files[0]));
  dropZoneB?.addEventListener("dragover", (e) => e.preventDefault());
  dropZoneB?.addEventListener("drop", (e) => {
    e.preventDefault();
    handleFileUpload(e.dataTransfer.files[0]);
  });

  document.getElementById("runPoisonBtn")?.addEventListener("click", async () => {
    const attack_type = document.getElementById("poisonAttackType").value;
    const malicious_pct = Number(document.getElementById("poisonPct").value);
    const data = await fetch("/api/lab/poison", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attack_type, malicious_pct }),
    }).then((r) => r.json());
    if (!data.ok) return;
    const r = data.result;
    document.getElementById("poisonOutputB").innerHTML = `
      <div class="c-row" style="margin-bottom:6px">
        <span>Cosine Align: Ours vs FedAvg</span>
        <strong class="badge-ok">${r.cosine_alignment_ours} vs ${r.cosine_alignment_fedavg}</strong>
      </div>
      <table class="mini-table">
        <thead><tr><th>Tenant</th><th>Role</th><th>L2 Norm</th><th>Cosine</th><th>Action</th></tr></thead>
        <tbody>${r.tenants.map((t) => `<tr>
          <td>${t.tenant_id}</td><td>${t.role}</td><td>${t.l2_norm}</td><td>${t.cosine}</td>
          <td class="${t.decision === "REJECTED" ? "badge-bad" : "badge-ok"}">${t.decision}</td>
        </tr>`).join("")}</tbody>
      </table>
    `;
  });

  document.getElementById("runTamperBtn")?.addEventListener("click", async () => {
    const tamper_mode = document.getElementById("tamperModeSelect").value;
    const data = await fetch("/api/lab/tamper", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tamper_mode }),
    }).then((r) => r.json());
    if (!data.ok) return;
    const r = data.result;
    document.getElementById("tamperOutputB").innerHTML = `
      <div class="c-row"><span>Verification Verdict</span><strong class="${r.verified ? "badge-ok" : "badge-bad"}">${r.verified ? "VERIFIED PASS" : "TAMPER REJECTED"}</strong></div>
      <div class="c-row"><span>Reason</span><strong>${r.reason}</strong></div>
      <div class="c-row"><span>Merkle Root</span><strong>${r.merkle_root_hex}</strong></div>
      <div class="c-row"><span>Wire Ratio / Time</span><strong>${r.wire_ratio}x · ${r.verify_time_ms} ms</strong></div>
    `;
  });

  /* Hydrate Initial State */
  fetch("/api/init")
    .then((r) => r.json())
    .then((state) => {
      document.getElementById("engineBadge").textContent =
        `GRU ${state.model_info.params.toLocaleString()} params · Live F1 ${state.model_info.live_test_f1_pct}%`;
      const list = document.getElementById("sampleListB");
      if (list) {
        list.innerHTML = (state.pcap_samples || [])
          .map((s) => `<button type="button" class="sample-btn" data-pcap="${s.name}">
            <span>${s.label}</span>
            <span class="sample-tag ${s.expected === "benign" ? "tag-ben" : "tag-atk"}">${s.expected.toUpperCase()}</span>
          </button>`)
          .join("");
        list.querySelectorAll(".sample-btn").forEach((btn) => {
          btn.addEventListener("click", () => inspectSampleB(btn.dataset.pcap));
        });
        if (state.pcap_samples && state.pcap_samples.length) {
          const syn = state.pcap_samples.find((s) => s.name.includes("syn_flood")) || state.pcap_samples[0];
          inspectSampleB(syn.name);
        }
      }
      completedStep = state.completed_step || 0;
      syncStepCards();
      refreshVmStatus();
    });

  /* ============ OPTION C: LIVE UBUNTU VM + CRASH DEFENSE ============ */
  let vmCompletedStep = 0;
  let vmBusy = false;
  let selectedVmPcapName = null;
  let vmCredsHydrated = false;

  function getVmCredentials() {
    return {
      vm_user: document.getElementById("vmUserInput")?.value.trim() || "ubuntu",
      vm_pass: document.getElementById("vmPassInput")?.value || "ubuntu",
      vmx_path: document.getElementById("vmVmxInput")?.value.trim() || "",
    };
  }

  document.getElementById("vmPassToggleBtn")?.addEventListener("click", (e) => {
    const passEl = document.getElementById("vmPassInput");
    if (!passEl) return;
    const isHidden = passEl.type === "password";
    passEl.type = isHidden ? "text" : "password";
    e.currentTarget.textContent = isHidden ? "Hide" : "Show";
  });

  function escapeHtml(txt) {
    return String(txt)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function syncVmStepCards() {
    document.querySelectorAll(".step-card[data-vm-step]").forEach((card) => {
      const step = Number(card.dataset.vmStep);
      const badge = document.getElementById(`vmStatus${step}`);
      const btn = card.querySelector(".vm-step-btn");
      const lbl = btn?.querySelector(".btn-label");
      card.classList.remove("is-ready", "is-locked", "is-done", "is-processing");
      if (!badge || !btn || !lbl) return;

      if (step <= vmCompletedStep) {
        card.classList.add("is-done");
        badge.className = "status-badge status-done";
        badge.textContent = "Done";
        lbl.textContent = `VM 0${step} Done`;
        btn.disabled = true;
      } else if (step === vmCompletedStep + 1) {
        card.classList.add("is-ready");
        badge.className = "status-badge status-ready";
        badge.textContent = "Ready";
        lbl.textContent = `Run VM Step 0${step}`;
        btn.disabled = vmBusy;
      } else {
        card.classList.add("is-locked");
        badge.className = "status-badge status-locked";
        badge.textContent = "Locked";
        lbl.textContent = `Run VM Step 0${step}`;
        btn.disabled = true;
      }
    });
  }

  function renderVmSafety(s) {
    if (!s) return;
    const stateBadge = document.getElementById("vmStateBadge");
    if (stateBadge) {
      stateBadge.className = s.vm_running ? "status-badge status-done" : "status-badge status-locked";
      stateBadge.textContent = s.vm_running ? "RUNNING" : "STOPPED";
    }
    const ramEl = document.getElementById("vmHostRam");
    const cpuEl = document.getElementById("vmHostCpu");
    const noteEl = document.getElementById("vmGuardNote");
    if (ramEl) ramEl.textContent = `${s.host_ram_available_gb} / ${s.host_ram_total_gb} GB`;
    if (cpuEl) cpuEl.textContent = `${s.host_cpu_pct}%`;
    if (noteEl) noteEl.textContent = s.guard_reason || "Hardware guard active";
    if (!vmCredsHydrated) {
      const userEl = document.getElementById("vmUserInput");
      const vmxEl = document.getElementById("vmVmxInput");
      const savedUser = localStorage.getItem("trustfed_vm_user");
      const savedVmx = localStorage.getItem("trustfed_vm_vmx");
      if (userEl) userEl.value = savedUser || s.vm_user || "ubuntu";
      if (vmxEl) vmxEl.value = s.vmx_path || savedVmx || "";
      vmCredsHydrated = true;
    }
    const authBadge = document.getElementById("vmAuthBadge");
    if (authBadge) {
      if (!s.vmrun_available || !s.vmx_available) {
        authBadge.className = "status-badge status-processing";
        authBadge.textContent = "Needs Setup";
      } else {
        authBadge.className = "status-badge status-done";
        authBadge.textContent = `User: ${s.vm_user || "ubuntu"}`;
      }
    }
  }

  function renderVmFiles(files) {
    const list = document.getElementById("vmFileList");
    const countEl = document.getElementById("vmFileCount");
    if (!list) return;
    const arr = Array.isArray(files) ? files : [];
    if (countEl) countEl.textContent = `${arr.length} files`;
    list.innerHTML = arr
      .map(
        (f) => `<button type="button" class="sample-btn vm-file-btn" data-vm-path="${escapeHtml(f.path)}">
          <span>${escapeHtml(f.label)}</span>
          <span class="sample-tag ${f.kind === "pcap" ? "tag-atk" : "tag-ben"}">${escapeHtml(f.kind.toUpperCase())} · ${f.size_kb} KB</span>
        </button>`
      )
      .join("");
    list.querySelectorAll(".vm-file-btn").forEach((btn) => {
      btn.addEventListener("click", () => inspectVmFile(btn.dataset.vmPath, btn));
    });
  }

  async function refreshVmStatus() {
    try {
      const s = await fetch("/api/vm/status").then((r) => r.json());
      renderVmSafety(s);
      renderVmFiles(s.files || []);
      if (typeof s.vm_completed_step === "number") {
        vmCompletedStep = s.vm_completed_step;
        syncVmStepCards();
      }
    } catch (_) {
      /* Ignore offline transient */
    }
  }

  async function inspectVmFile(relPath, clickedBtn) {
    document.querySelectorAll(".vm-file-btn").forEach((b) => b.classList.toggle("active", b === clickedBtn));
    const titleEl = document.getElementById("vmSelectedFileName");
    const outBox = document.getElementById("vmFileAnalysisOut");
    const openBBtn = document.getElementById("vmOpenInBBtn");
    if (titleEl) titleEl.textContent = relPath.split("/").pop();
    if (outBox) outBox.innerHTML = `<div class="muted-cell">Analyzing ${escapeHtml(relPath)}...</div>`;

    const data = await fetch("/api/vm/inspect-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: relPath }),
    }).then((r) => r.json());

    if (!data.ok) {
      if (outBox) outBox.innerHTML = `<div class="badge-bad">${escapeHtml(data.error || "Inspect failed")}</div>`;
      return;
    }

    const res = data.result;
    if (res.kind === "pcap" || res.kind === "csv") {
      const ai = res.ai;
      selectedVmPcapName = res.kind === "pcap" ? res.name : null;
      if (openBBtn) openBBtn.hidden = !selectedVmPcapName;
      const isAtk = ai.verdict === "ATTACK";
      const rows = (ai.sample_flows || [])
        .slice(0, 8)
        .map((f) => {
          const cls = f.decision === "DROP" ? "badge-bad" : "badge-ok";
          return `<tr><td>${escapeHtml(f.src_ip)}</td><td>${escapeHtml(f.dst_ip)}</td><td>${f.attack_prob.toFixed(3)}</td><td class="${cls}">${f.decision}</td></tr>`;
        })
        .join("");
      outBox.innerHTML = `
        <div class="verdict-pill ${isAtk ? "verdict-atk" : "verdict-safe"}" style="margin-bottom:6px">
          ${isAtk ? "ATTACK DETECTED" : "BENIGN"} (${ai.attack_flows}/${ai.total_flows} flows · Prob ${ai.max_attack_prob.toFixed(3)})
        </div>
        <div class="rule-preview" style="margin-bottom:6px">${escapeHtml(ai.ovs_rule_preview)}</div>
        <table class="mini-table">
          <thead><tr><th>Src IP</th><th>Dst IP</th><th>Prob</th><th>OVS</th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4" class="muted-cell">Scored CSV rows</td></tr>'}</tbody>
        </table>
      `;
    } else {
      selectedVmPcapName = null;
      if (openBBtn) openBBtn.hidden = true;
      outBox.innerHTML = `
        <div class="mono-dim" style="margin-bottom:5px">${res.line_count} lines · ${res.size_kb} KB${res.truncated ? " (truncated preview)" : ""}</div>
        <pre class="file-pre">${escapeHtml(res.content || "")}</pre>
      `;
    }
  }

  document.getElementById("vmOpenInBBtn")?.addEventListener("click", () => {
    if (!selectedVmPcapName) return;
    activateView("b");
    inspectSampleB(selectedVmPcapName);
  });

  document.querySelectorAll(".vm-step-btn[data-run-vm-step]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const step = Number(btn.dataset.runVmStep);
      if (vmBusy || step !== vmCompletedStep + 1) return;
      vmBusy = true;
      const card = btn.closest(".step-card");
      const badge = document.getElementById(`vmStatus${step}`);
      card?.classList.remove("is-ready");
      card?.classList.add("is-processing");
      if (badge) {
        badge.className = "status-badge status-processing";
        badge.textContent = "Running...";
      }
      btn.querySelector(".btn-label").textContent = "Running VM...";
      document.querySelectorAll(".vm-step-btn").forEach((b) => { b.disabled = true; });

      const termEl = document.getElementById("vmTerminalOut");
      const autoFallback = Boolean(document.getElementById("vmAutoFallback")?.checked);

      try {
        const creds = getVmCredentials();
        const data = await fetch("/api/vm/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ step, safe_mode: false, auto_fallback: autoFallback, ...creds }),
        }).then((r) => r.json());
        if (!data.ok) throw new Error(data.error || "VM step failed");
        const r = data.result;
        vmCompletedStep = data.vm_completed_step;
        document.getElementById("vmCmdBar").textContent = `$ ${r.command}`;
        document.getElementById("vmExecTime").textContent = `${r.elapsed_ms} ms`;
        const modeEl = document.getElementById("vmExecModeBadge");
        if (modeEl) {
          const isLive = r.mode === "LIVE_VMWARE_GUEST";
          modeEl.className = isLive ? "status-badge status-done" : "status-badge status-ready";
          modeEl.textContent = isLive ? "LIVE VMWARE GUEST" : "SAFE GUARD REPLAY";
        }
        if (termEl) {
          const block = `\n=== [VM Step 0${step}] ${r.title} (${r.mode}) ===\n$ ${r.command}\n${r.output}\n`;
          termEl.textContent = (termEl.textContent.startsWith("[Crash Guard Ready]") ? "" : termEl.textContent) + block;
          termEl.scrollTop = termEl.scrollHeight;
        }
        renderVmSafety(r.safety);
        if (data.files) renderVmFiles(data.files);
      } catch (err) {
        if (termEl) {
          termEl.textContent += `\n[Error] ${err.message}\n`;
          termEl.scrollTop = termEl.scrollHeight;
        }
      } finally {
        vmBusy = false;
        syncVmStepCards();
      }
    });
  });

  document.getElementById("vmEmergencyStopBtn")?.addEventListener("click", async () => {
    if (vmBusy) return;
    const termEl = document.getElementById("vmTerminalOut");
    const creds = getVmCredentials();
    const res = await fetch("/api/vm/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(creds),
    }).then((r) => r.json());
    vmCompletedStep = 0;
    syncVmStepCards();
    renderVmSafety(res.safety);
    if (termEl) {
      termEl.textContent += `\n[Crash Defense · Stop VM] ${res.message}\n`;
      termEl.scrollTop = termEl.scrollHeight;
    }
  });

  document.getElementById("vmResetBtn")?.addEventListener("click", async () => {
    if (vmBusy) return;
    const res = await fetch("/api/vm/reset", { method: "POST" }).then((r) => r.json());
    vmCompletedStep = 0;
    syncVmStepCards();
    renderVmSafety(res.safety);
  });

  document.getElementById("vmClearTermBtn")?.addEventListener("click", () => {
    const termEl = document.getElementById("vmTerminalOut");
    if (termEl) termEl.textContent = "[Terminal Cleared] Ready for VM commands.";
  });

  document.getElementById("vmCopyTermBtn")?.addEventListener("click", () => {
    const termEl = document.getElementById("vmTerminalOut");
    if (termEl && navigator.clipboard) navigator.clipboard.writeText(termEl.textContent);
  });

  document.getElementById("vmAutoSetupBtn")?.addEventListener("click", async (e) => {
    if (vmBusy) return;
    vmBusy = true;
    const btn = e.currentTarget;
    const origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Checking...";
    const termEl = document.getElementById("vmTerminalOut");
    const authBadge = document.getElementById("vmAuthBadge");
    const modeEl = document.getElementById("vmExecModeBadge");

    try {
      const creds = getVmCredentials();
      const data = await fetch("/api/vm/setup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...creds, safe_mode: false }),
      }).then((r) => r.json());
      if (!data.ok) throw new Error(data.error || "VM auto-setup failed");

      const r = data.result;
      document.getElementById("vmCmdBar").textContent = `$ ${r.command}`;
      document.getElementById("vmExecTime").textContent = `${r.elapsed_ms} ms`;
      if (modeEl) {
        const isOk = r.status === "SETUP_COMPLETE";
        modeEl.className = isOk ? "status-badge status-done" : "status-badge status-processing";
        modeEl.textContent = r.status.replaceAll("_", " ");
      }
      if (authBadge) {
        const ok = r.status === "SETUP_COMPLETE";
        authBadge.className = ok ? "status-badge status-done" : "status-badge status-processing";
        authBadge.textContent = ok ? "SETUP COMPLETE" : r.status.replaceAll("_", " ");
      }
      if (r.safety?.vmx_path) {
        const vmxEl = document.getElementById("vmVmxInput");
        if (vmxEl) vmxEl.value = r.safety.vmx_path;
        localStorage.setItem("trustfed_vm_vmx", r.safety.vmx_path);
      }
      if (r.status === "SETUP_COMPLETE" && creds.vm_user) {
        localStorage.setItem("trustfed_vm_user", creds.vm_user);
      }
      renderVmSafety(r.safety);
      if (termEl) {
        const block = `\n=== [Auto-Setup VM · ${r.status}] ===\n$ ${r.command}\n${r.output}\n`;
        termEl.textContent =
          (termEl.textContent.startsWith("[Crash Guard Ready]") ? "" : termEl.textContent) + block;
        termEl.scrollTop = termEl.scrollHeight;
      }
    } catch (err) {
      if (termEl) {
        termEl.textContent += `\n[Auto-Setup Error] ${err.message}\n`;
        termEl.scrollTop = termEl.scrollHeight;
      }
    } finally {
      vmBusy = false;
      btn.disabled = false;
      btn.textContent = origText;
      syncVmStepCards();
    }
  });
});
