// ============================================================
// TrustFed-SDN — Hero Canvas: "Floating Brand Logos"
//
// Concept:
// - Dark Theme: Glowing TrustFed-SDN logos with fiery cyber aura
//   and ambient glowing particles drifting smoothly in space.
// - Light Theme: Elegant ash-colored logos in various sizes (small & big),
//   rendered without glow and with subtle low opacity (watermark effect).
//
// Drop-in replacement for <canvas id="netCanvas"> in the hero section.
// ============================================================

(function () {
  function initLogoParticles() {
    const canvas = document.getElementById("netCanvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let width = 0;
    let height = 0;
    let dpr = 1;
    let mouse = { targetX: 0, targetY: 0, currentX: 0, currentY: 0 };
    let logos = [];
    let ambientParticles = [];
    let animFrameId = null;

    // ---------- Preload Logo Assets ----------
    const darkLogo = new Image();
    darkLogo.src = "images/logo-glow.png";

    const lightLogo = new Image();
    lightLogo.src = "images/logo-ash.png";

    // Fallback image if ash logo hasn't loaded yet
    const baseLogo = new Image();
    baseLogo.src = "images/logo.png";

    // ---------- Theme & Colors ----------
    function isLightTheme() {
      return document.documentElement.getAttribute("data-theme") === "light";
    }

    function getThemeColors() {
      const styles = getComputedStyle(document.documentElement);
      const c1 = styles.getPropertyValue("--accent-1").trim() || "#E8123C"; // Crimson
      const c2 = styles.getPropertyValue("--accent-2").trim() || "#FF7A1A"; // Orange
      return { c1: hexToRgb(c1), c2: hexToRgb(c2) };
    }

    function hexToRgb(hex) {
      const clean = hex.replace("#", "");
      const num = parseInt(clean.length === 3 ? clean.split("").map(c => c + c).join("") : clean, 16);
      return {
        r: (num >> 16) & 255,
        g: (num >> 8) & 255,
        b: num & 255
      };
    }

    // ---------- Sizing ----------
    function resize() {
      dpr = window.devicePixelRatio || 1;
      width = canvas.width = (canvas.offsetWidth || window.innerWidth) * dpr;
      height = canvas.height = (canvas.offsetHeight || window.innerHeight) * dpr;
    }

    // ---------- Floating Logo Factory ----------
    // Creates varied small to large floating logos
    function createLogoItem(baseX, baseY, sizeCategory, depth) {
      let baseSize; // In CSS pixels
      let darkAlpha;
      let lightAlpha;

      switch (sizeCategory) {
        case "large":
          baseSize = 85 + Math.random() * 25; // 85px - 110px
          darkAlpha = 0.75 + Math.random() * 0.15;
          lightAlpha = 0.16 + Math.random() * 0.06;
          break;
        case "medium":
          baseSize = 48 + Math.random() * 22; // 48px - 70px
          darkAlpha = 0.60 + Math.random() * 0.20;
          lightAlpha = 0.14 + Math.random() * 0.05;
          break;
        case "small":
          baseSize = 26 + Math.random() * 14; // 26px - 40px
          darkAlpha = 0.45 + Math.random() * 0.20;
          lightAlpha = 0.11 + Math.random() * 0.04;
          break;
        case "micro":
        default:
          baseSize = 14 + Math.random() * 8; // 14px - 22px
          darkAlpha = 0.35 + Math.random() * 0.18;
          lightAlpha = 0.09 + Math.random() * 0.03;
          break;
      }

      return {
        x: baseX,
        y: baseY,
        baseSize: baseSize,
        size: baseSize * dpr,
        vx: (Math.random() - 0.5) * 0.22 * dpr,
        vy: (Math.random() - 0.5) * 0.18 * dpr,
        floatSeed: Math.random() * 100,
        floatSpeed: 0.01 + Math.random() * 0.008,
        swaySpeed: 0.006 + Math.random() * 0.006,
        swayAmp: 0.06 + Math.random() * 0.05, // Subtle +/- 3-4 degree tilt
        depth: depth,
        darkAlpha: darkAlpha,
        lightAlpha: lightAlpha,
        pulseOffset: Math.random() * Math.PI * 2
      };
    }

    // ---------- Seed Constellation ----------
    function seed() {
      logos = [];
      ambientParticles = [];

      const isMobile = width < 768 * dpr;

      if (isMobile) {
        // Mobile: 6 well-placed floating logos
        logos.push(createLogoItem(width * 0.16, height * 0.26, "medium", 0.7));
        logos.push(createLogoItem(width * 0.84, height * 0.32, "large", 0.9));
        logos.push(createLogoItem(width * 0.22, height * 0.72, "small", 0.6));
        logos.push(createLogoItem(width * 0.80, height * 0.76, "medium", 0.8));
        logos.push(createLogoItem(width * 0.48, height * 0.14, "small", 0.5));
        logos.push(createLogoItem(width * 0.52, height * 0.88, "micro", 0.4));
      } else {
        // Desktop: Rich arrangement of small, medium, and large logos framing the hero
        // 1. Prominent Large Flank Logos
        logos.push(createLogoItem(width * 0.12, height * 0.44, "large", 1.0));
        logos.push(createLogoItem(width * 0.88, height * 0.50, "large", 1.1));

        // 2. Medium Floating Logos
        logos.push(createLogoItem(width * 0.24, height * 0.22, "medium", 0.85));
        logos.push(createLogoItem(width * 0.78, height * 0.24, "medium", 0.85));
        logos.push(createLogoItem(width * 0.20, height * 0.74, "medium", 0.75));
        logos.push(createLogoItem(width * 0.82, height * 0.78, "medium", 0.80));

        // 3. Small Floating Logos
        logos.push(createLogoItem(width * 0.08, height * 0.70, "small", 0.65));
        logos.push(createLogoItem(width * 0.92, height * 0.28, "small", 0.65));
        logos.push(createLogoItem(width * 0.38, height * 0.16, "small", 0.60));
        logos.push(createLogoItem(width * 0.64, height * 0.18, "small", 0.60));

        // 4. Micro Background Watermark Logos
        logos.push(createLogoItem(width * 0.32, height * 0.84, "micro", 0.45));
        logos.push(createLogoItem(width * 0.68, height * 0.86, "micro", 0.45));
        logos.push(createLogoItem(width * 0.50, height * 0.10, "micro", 0.40));
      }

      // Ambient floating micro-sparks / dust
      const count = Math.max(30, Math.floor((width * height) / 40000));
      for (let i = 0; i < count; i++) {
        ambientParticles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.3 * dpr,
          vy: -(0.2 + Math.random() * 0.4) * dpr, // Soft upward drift
          radius: (1.0 + Math.random() * 1.5) * dpr,
          alpha: 0.15 + Math.random() * 0.45,
          twinkle: Math.random() * Math.PI * 2,
          twinkleSpeed: 0.015 + Math.random() * 0.025,
          colorType: Math.random() > 0.4 ? "orange" : "crimson"
        });
      }
    }

    // ---------- Mouse Parallax Listener ----------
    function onMouseMove(e) {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      mouse.targetX = ((e.clientX - rect.left) / rect.width - 0.5) * 28;
      mouse.targetY = ((e.clientY - rect.top) / rect.height - 0.5) * 28;
    }

    function onMouseLeave() {
      mouse.targetX = 0;
      mouse.targetY = 0;
    }

    window.addEventListener("mousemove", onMouseMove, { passive: true });
    document.addEventListener("mouseleave", onMouseLeave);

    // ---------- Main Render Loop ----------
    function render(timestamp) {
      const light = isLightTheme();
      const colors = getThemeColors();

      ctx.clearRect(0, 0, width, height);

      // Smooth mouse parallax easing
      mouse.currentX += (mouse.targetX - mouse.currentX) * 0.05;
      mouse.currentY += (mouse.targetY - mouse.currentY) * 0.05;

      // 1. Ambient Floating Particles / Micro-dust
      ambientParticles.forEach((p) => {
        p.y += p.vy;
        p.x += p.vx;
        p.twinkle += p.twinkleSpeed;

        if (p.y < -15 * dpr) {
          p.y = height + 10 * dpr;
          p.x = Math.random() * width;
        }
        if (p.x < -15 * dpr) p.x = width + 10 * dpr;
        if (p.x > width + 15 * dpr) p.x = -10 * dpr;

        if (light) {
          // Light theme: very faint subtle ash micro-dots, NO glow
          const ashAlpha = p.alpha * 0.28;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius * 0.8, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(100, 116, 139, ${ashAlpha})`;
          ctx.fill();
        } else {
          // Dark theme: glowing ember particles
          const currentAlpha = p.alpha * (0.6 + 0.4 * Math.sin(p.twinkle));
          const rgb = p.colorType === "crimson" ? colors.c1 : colors.c2;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${currentAlpha})`;
          ctx.shadowColor = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${currentAlpha * 0.8})`;
          ctx.shadowBlur = 6 * dpr;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      });

      // 2. Select appropriate logo image for current theme
      let activeLogoImg;
      if (light) {
        activeLogoImg = lightLogo.complete && lightLogo.naturalWidth > 0 ? lightLogo : baseLogo;
      } else {
        activeLogoImg = darkLogo.complete && darkLogo.naturalWidth > 0 ? darkLogo : baseLogo;
      }

      // 3. Faint Constellation Web Lines (Dark mode only)
      if (!light) {
        const maxDist = 170 * dpr;
        for (let i = 0; i < logos.length; i++) {
          for (let j = i + 1; j < logos.length; j++) {
            const l1 = logos[i];
            const l2 = logos[j];
            const dx = l1.x - l2.x;
            const dy = l1.y - l2.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < maxDist) {
              const lineAlpha = (1 - dist / maxDist) * 0.12;
              ctx.beginPath();
              ctx.moveTo(l1.x, l1.y);
              ctx.lineTo(l2.x, l2.y);
              ctx.strokeStyle = `rgba(${colors.c2.r}, ${colors.c2.g}, ${colors.c2.b}, ${lineAlpha})`;
              ctx.lineWidth = 1 * dpr;
              ctx.stroke();
            }
          }
        }
      }

      // 4. Render Floating Logos
      logos.forEach((logo) => {
        // Subtle drift movement
        logo.x += logo.vx;
        logo.y += logo.vy;

        // Bounce gently inside canvas bounds
        if (logo.x < width * 0.05) logo.vx = Math.abs(logo.vx);
        if (logo.x > width * 0.95) logo.vx = -Math.abs(logo.vx);
        if (logo.y < height * 0.08) logo.vy = Math.abs(logo.vy);
        if (logo.y > height * 0.92) logo.vy = -Math.abs(logo.vy);

        // Sinusoidal floating levitation & subtle breathing sway
        logo.floatSeed += logo.floatSpeed;
        const floatOffsetY = Math.sin(logo.floatSeed) * (7 * logo.depth * dpr);
        const floatOffsetX = Math.cos(logo.floatSeed * 0.75) * (4 * logo.depth * dpr);
        const sway = Math.sin(timestamp * logo.swaySpeed * 0.05) * logo.swayAmp;

        // Mouse Parallax
        const parallaxX = mouse.currentX * logo.depth * dpr;
        const parallaxY = mouse.currentY * logo.depth * dpr;

        const posX = logo.x + floatOffsetX + parallaxX;
        const posY = logo.y + floatOffsetY + parallaxY;

        ctx.save();
        ctx.translate(posX, posY);
        ctx.rotate(sway);

        const currentSize = logo.size;
        const half = currentSize / 2;

        if (light) {
          // ================= LIGHT THEME =================
          // Ash color, small & big logos, NO glow, LOW opacity
          ctx.globalAlpha = logo.lightAlpha;
          ctx.shadowBlur = 0; // Strictly NO glow

          if (activeLogoImg.complete && activeLogoImg.naturalWidth > 0) {
            ctx.drawImage(activeLogoImg, -half, -half, currentSize, currentSize);
          }
        } else {
          // ================= DARK THEME =================
          // Glowing logo with fiery aura
          const pulse = Math.sin(timestamp * 0.0025 + logo.pulseOffset) * 0.15 + 0.85;
          const darkAlpha = logo.darkAlpha * pulse;

          ctx.globalAlpha = darkAlpha;

          // Soft ambient radial glow halo behind the logo
          const glowRadius = half * 1.35;
          const radialGlow = ctx.createRadialGradient(0, 0, half * 0.2, 0, 0, glowRadius);
          radialGlow.addColorStop(0, `rgba(${colors.c2.r}, ${colors.c2.g}, ${colors.c2.b}, ${0.18 * darkAlpha})`);
          radialGlow.addColorStop(0.5, `rgba(${colors.c1.r}, ${colors.c1.g}, ${colors.c1.b}, ${0.10 * darkAlpha})`);
          radialGlow.addColorStop(1, "transparent");

          ctx.beginPath();
          ctx.arc(0, 0, glowRadius, 0, Math.PI * 2);
          ctx.fillStyle = radialGlow;
          ctx.fill();

          // Draw the glowing logo itself with soft glow shadow
          ctx.shadowColor = `rgba(${colors.c2.r}, ${colors.c2.g}, ${colors.c2.b}, ${0.65 * darkAlpha})`;
          ctx.shadowBlur = 14 * dpr;

          if (activeLogoImg.complete && activeLogoImg.naturalWidth > 0) {
            ctx.drawImage(activeLogoImg, -half, -half, currentSize, currentSize);
          }

          ctx.shadowBlur = 0;
        }

        ctx.restore();
      });

      if (!prefersReducedMotion) {
        animFrameId = requestAnimationFrame(render);
      }
    }

    function start() {
      resize();
      seed();
      if (prefersReducedMotion) {
        render(0);
      } else {
        if (animFrameId) cancelAnimationFrame(animFrameId);
        animFrameId = requestAnimationFrame(render);
      }
    }

    window.addEventListener("resize", () => {
      resize();
      seed();
    });

    start();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLogoParticles);
  } else {
    initLogoParticles();
  }
})();
