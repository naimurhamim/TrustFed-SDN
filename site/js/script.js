// ============================================================================
// TrustFed-SDN — Interactive Research Monograph Controller (Book Edition v1.0)
// Handles 10-chapter pagination, scroll-all mode, keyboard/hash navigation,
// dynamic data/key_numbers.json hydration, figure lightbox, theme & copy UX.
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  const html = document.documentElement;
  const body = document.body;

  /* ---------- 1. Theme Switcher (Dark Cyber <-> Light Academic Paper) ---------- */
  const themeBtn = document.getElementById("themeToggleBtn");
  const brandLogos = document.querySelectorAll("[data-brand-logo]");

  const applyLogos = (theme) => {
    const src = theme === "light" ? "images/logo-gradient.png" : "images/logo.png";
    brandLogos.forEach((img) => {
      img.src = src;
    });
  };

  const savedTheme = localStorage.getItem("theme") || "dark";
  html.setAttribute("data-theme", savedTheme);
  applyLogos(savedTheme);

  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
      applyLogos(next);
    });
  }

  /* ---------- 2. Book Chapters Registry ---------- */
  const chapters = Array.from(document.querySelectorAll(".chapter-page[data-chapter]"));
  const tocItems = Array.from(document.querySelectorAll(".toc-item[data-goto]"));
  const pagePips = Array.from(document.querySelectorAll(".page-pip[data-goto]"));

  const prevBtnTop = document.getElementById("prevMiniBtn");
  const nextBtnTop = document.getElementById("nextMiniBtn");
  const prevBtnBot = document.getElementById("prevPageBtn");
  const nextBtnBot = document.getElementById("nextPageBtn");
  const prevTitleEl = document.getElementById("prevPageTitle");
  const nextTitleEl = document.getElementById("nextPageTitle");

  const chapterReadout = document.getElementById("chapterReadout");
  const folioLeft = document.getElementById("folioRunningLeft");
  const progressFill = document.getElementById("readProgressFill");
  const modeToggleBtn = document.getElementById("modeToggleBtn");
  const modeLabelEl = document.getElementById("modeToggleLabel");

  const legacyAliases = {
    abstract: 2,
    methodology: 3,
    workflow: 3,
    data: 4,
    evaluation: 6,
    testbed: 7,
    progress: 8,
    references: 9,
    team: 10,
  };

  let currentPage = 1;
  let isScrollMode = localStorage.getItem("bookMode") === "scroll";

  const chapterMeta = (num) => {
    const el = chapters[num - 1];
    if (!el) return { num: 1, title: "Front Cover" };
    return {
      num,
      title: el.dataset.title || `Chapter ${String(num).padStart(2, "0")}`,
    };
  };

  const updateProgressBar = () => {
    if (!progressFill) return;
    if (isScrollMode) {
      const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
      const pct = maxScroll > 0 ? Math.min(100, Math.max(4, (window.scrollY / maxScroll) * 100)) : 100;
      progressFill.style.width = `${pct}%`;
    } else {
      progressFill.style.width = `${(currentPage / chapters.length) * 100}%`;
    }
  };

  const renderPage = (targetPage, opts = {}) => {
    const prevPage = currentPage;
    const clamped = Math.max(1, Math.min(chapters.length, targetPage));
    const dir = opts.dir || (clamped >= prevPage ? "next" : "prev");
    currentPage = clamped;

    chapters.forEach((el, idx) => {
      const pNum = idx + 1;
      const active = pNum === currentPage;
      el.classList.remove("active", "anim-next", "anim-prev");
      if (active) {
        el.classList.add("active");
        if (!isScrollMode && opts.animate !== false) {
          void el.offsetWidth; // trigger reflow for smooth page flip
          el.classList.add(dir === "prev" ? "anim-prev" : "anim-next");
        }
      }
    });

    tocItems.forEach((btn) => {
      btn.classList.toggle("active", Number(btn.dataset.goto) === currentPage);
    });
    pagePips.forEach((pip) => {
      pip.classList.toggle("active", Number(pip.dataset.goto) === currentPage);
    });

    const cur = chapterMeta(currentPage);
    const padCur = String(currentPage).padStart(2, "0");

    if (chapterReadout) {
      chapterReadout.textContent = isScrollMode
        ? "All 10 Chapters · Continuous Monograph View"
        : `Chapter ${padCur} / ${chapters.length} · ${cur.title}`;
    }
    if (folioLeft) {
      folioLeft.textContent = `TRUSTFED-SDN MONOGRAPH // CH. ${padCur} · ${cur.title.toUpperCase()}`;
    }

    // Update top & bottom Prev/Next controls
    const hasPrev = currentPage > 1;
    const hasNext = currentPage < chapters.length;
    if (prevBtnTop) prevBtnTop.disabled = !hasPrev;
    if (nextBtnTop) nextBtnTop.disabled = false;
    if (prevBtnBot) prevBtnBot.disabled = !hasPrev;

    if (prevTitleEl) {
      if (hasPrev) {
        const pm = chapterMeta(currentPage - 1);
        prevTitleEl.textContent = `Ch. ${String(pm.num).padStart(2, "0")} · ${pm.title}`;
      } else {
        prevTitleEl.textContent = "Front Cover";
      }
    }

    if (nextTitleEl) {
      if (hasNext) {
        const nm = chapterMeta(currentPage + 1);
        nextTitleEl.textContent = `Ch. ${String(nm.num).padStart(2, "0")} · ${nm.title}`;
      } else {
        nextTitleEl.textContent = "Ch. 01 · Return to Cover";
      }
    }

    if (!opts.skipHash && !isScrollMode) {
      history.replaceState(null, "", `#ch${padCur}`);
    }

    updateProgressBar();

    if (opts.scroll !== false) {
      if (isScrollMode) {
        const targetEl = chapters[currentPage - 1];
        if (targetEl) targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
      } else {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    }
  };

  const applyReadingMode = (scrollOn) => {
    isScrollMode = scrollOn;
    body.classList.toggle("scroll-mode", isScrollMode);
    localStorage.setItem("bookMode", isScrollMode ? "scroll" : "book");
    if (modeToggleBtn) modeToggleBtn.classList.toggle("active", isScrollMode);
    if (modeLabelEl) {
      modeLabelEl.textContent = isScrollMode ? "Book Pages" : "Scroll All";
    }
    renderPage(currentPage, { animate: false, scroll: false, skipHash: true });
  };

  if (modeToggleBtn) {
    modeToggleBtn.addEventListener("click", () => {
      applyReadingMode(!isScrollMode);
    });
  }

  /* Bind all [data-goto] controls (TOC, Pips, CTA links) */
  document.addEventListener("click", (ev) => {
    const targetBtn = ev.target.closest("[data-goto]");
    if (!targetBtn) return;
    ev.preventDefault();
    const num = Number(targetBtn.dataset.goto);
    if (num >= 1 && num <= chapters.length) {
      renderPage(num);
      closeSidebar();
    }
  });

  if (prevBtnTop) prevBtnTop.addEventListener("click", () => renderPage(currentPage - 1, { dir: "prev" }));
  if (nextBtnTop) {
    nextBtnTop.addEventListener("click", () =>
      renderPage(currentPage < chapters.length ? currentPage + 1 : 1, { dir: "next" })
    );
  }
  if (prevBtnBot) prevBtnBot.addEventListener("click", () => renderPage(currentPage - 1, { dir: "prev" }));
  if (nextBtnBot) {
    nextBtnBot.addEventListener("click", () =>
      renderPage(currentPage < chapters.length ? currentPage + 1 : 1, { dir: "next" })
    );
  }

  /* ---------- 3. Mobile Book Spine Drawer ---------- */
  const sidebar = document.getElementById("bookSidebar");
  const backdrop = document.getElementById("sidebarBackdrop");
  const tocOpenBtn = document.getElementById("tocMobileBtn");
  const tocCloseBtn = document.getElementById("sidebarCloseBtn");

  function openSidebar() {
    if (sidebar) sidebar.classList.add("open");
    if (backdrop) backdrop.classList.add("open");
  }
  function closeSidebar() {
    if (sidebar) sidebar.classList.remove("open");
    if (backdrop) backdrop.classList.remove("open");
  }

  if (tocOpenBtn) tocOpenBtn.addEventListener("click", openSidebar);
  if (tocCloseBtn) tocCloseBtn.addEventListener("click", closeSidebar);
  if (backdrop) backdrop.addEventListener("click", closeSidebar);

  /* ---------- 4. Keyboard & Hash Navigation ---------- */
  const resolveHash = () => {
    const raw = (window.location.hash || "").replace(/^#/, "").toLowerCase();
    if (!raw) return 1;
    const match = raw.match(/^ch0?(\d{1,2})$/);
    if (match) {
      const n = Number(match[1]);
      if (n >= 1 && n <= chapters.length) return n;
    }
    if (legacyAliases[raw]) return legacyAliases[raw];
    return 1;
  };

  window.addEventListener("hashchange", () => {
    renderPage(resolveHash(), { skipHash: true });
  });

  window.addEventListener("scroll", () => {
    if (isScrollMode) updateProgressBar();
  }, { passive: true });

  /* ---------- 5. Full-Screen Figure Lightbox Modal ---------- */
  const lightbox = document.getElementById("lightboxModal");
  const lbImg = document.getElementById("lightboxImg");
  const lbTitle = document.getElementById("lightboxTitle");
  const lbSource = document.getElementById("lightboxSource");
  const lbPdf = document.getElementById("lightboxPdfLink");
  const lbClose = document.getElementById("lightboxCloseBtn");

  const closeLightbox = () => {
    if (lightbox) lightbox.classList.remove("open");
  };

  document.querySelectorAll(".figure-viewport").forEach((vp) => {
    vp.addEventListener("click", () => {
      const img = vp.querySelector("img");
      if (!img || !lightbox) return;
      if (lbImg) {
        lbImg.src = img.src;
        lbImg.alt = img.alt || "Figure zoom view";
      }
      if (lbTitle) lbTitle.textContent = vp.dataset.title || img.alt || "Figure";
      if (lbSource) lbSource.textContent = vp.dataset.source ? `Source: ${vp.dataset.source}` : "";
      if (lbPdf) {
        if (vp.dataset.pdf) {
          lbPdf.href = vp.dataset.pdf;
          lbPdf.style.display = "inline-flex";
        } else {
          lbPdf.style.display = "none";
        }
      }
      lightbox.classList.add("open");
    });
  });

  if (lbClose) lbClose.addEventListener("click", closeLightbox);
  if (lightbox) {
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) closeLightbox();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeLightbox();
      closeSidebar();
      return;
    }
    if (isScrollMode || (lightbox && lightbox.classList.contains("open"))) return;
    if (["INPUT", "TEXTAREA"].includes((document.activeElement || {}).tagName)) return;
    if (e.key === "ArrowRight" && currentPage < chapters.length) {
      renderPage(currentPage + 1, { dir: "next" });
    } else if (e.key === "ArrowLeft" && currentPage > 1) {
      renderPage(currentPage - 1, { dir: "prev" });
    }
  });

  /* ---------- 6. Copy-to-Clipboard Buttons ---------- */
  document.querySelectorAll(".copy-btn[data-copy-target]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetId = btn.dataset.copyTarget;
      const codeEl = document.getElementById(targetId);
      if (!codeEl) return;
      const orig = btn.textContent;
      try {
        await navigator.clipboard.writeText(codeEl.textContent.trim());
        btn.textContent = "Copied!";
        setTimeout(() => {
          btn.textContent = orig;
        }, 1600);
      } catch {
        btn.textContent = "Copy manually";
        setTimeout(() => {
          btn.textContent = orig;
        }, 1800);
      }
    });
  });

  /* ---------- 7. Dynamic Hydration from data/key_numbers.json ---------- */
  fetch("data/key_numbers.json")
    .then((r) => (r.ok ? r.json() : null))
    .then((kn) => {
      if (!kn) return;
      const setText = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined && val !== null) el.textContent = val;
      };
      if (kn.fedavg_insdn_natural_n3_f1_pct) {
        setText(
          "kpiFedAvg",
          `${kn.fedavg_insdn_natural_n3_f1_pct.value.toFixed(2)} ± ${kn.fedavg_insdn_natural_n3_f1_pct.std.toFixed(2)}%`
        );
      }
      if (kn.centralized_gru_insdn_binary_f1_pct) {
        setText(
          "kpiCentralizedNote",
          `Centralized GRU: ${kn.centralized_gru_insdn_binary_f1_pct.value.toFixed(2)}% · InSDN Natural N=3`
        );
      }
      if (kn.secagg_tamper_detection_pct) {
        setText(
          "kpiTamper",
          `${kn.secagg_tamper_detection_pct.value.toFixed(1)}% (${kn.secagg_tamper_detection_pct.trials}/${kn.secagg_tamper_detection_pct.trials})`
        );
      }
      if (kn.secagg_wire_overhead_ratio_n3_n10) {
        const w = kn.secagg_wire_overhead_ratio_n3_n10;
        setText("kpiWire", `${w.n3.toFixed(2)}× – ${w.n20.toFixed(2)}×`);
      }
      if (kn.adaptive_dp_insdn_mia_auc_pct_eps4) {
        const m = kn.adaptive_dp_insdn_mia_auc_pct_eps4;
        setText("kpiMia", `${m.full_partition.toFixed(2)}% / ${m.rare_cohort_finance.toFixed(2)}%`);
      }
      if (kn.e6_insdn_r4_scaling20_f1_pct && kn.e6_insdn_r1_scaling20_f1_pct) {
        setText(
          "kpiRobustGain",
          `${kn.e6_insdn_r4_scaling20_f1_pct.value.toFixed(2)}% vs ${kn.e6_insdn_r1_scaling20_f1_pct.value.toFixed(2)}%`
        );
      }
      if (kn.live_testbed) {
        const lt = kn.live_testbed;
        setText("kpiOvs", `${lt.vm_ovs_drop_install_ms.toFixed(2)} ms`);
        setText(
          "kpiOvsNote",
          `${lt.vm_attack_block_pct.toFixed(0)}% Attack Block · ${lt.vm_false_block_pct.toFixed(0)}% Benign Block · ${lt.calibrated_live_fbr_pct.toFixed(2)}% Live FBR`
        );
      }
    })
    .catch(() => {
      // Static pre-rendered fallback numbers already match key_numbers.json verbatim
    });

  /* ---------- Initialize Book Landing State ---------- */
  currentPage = resolveHash();
  applyReadingMode(isScrollMode);
});
