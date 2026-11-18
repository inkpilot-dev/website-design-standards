(function () {
  var year = document.querySelectorAll(".year");
  year.forEach(function (el) { el.textContent = new Date().getFullYear(); });

  var today = new Date().getDay();
  var row = document.querySelector('.days li[data-day="' + today + '"]');
  if (row) row.classList.add("today");

  /* Gentle parallax on large photos */
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var layers = document.querySelectorAll(".parallax");
  if (layers.length && !reduce) {
    var ticking = false;
    var update = function () {
      var vh = window.innerHeight;
      layers.forEach(function (img) {
        var r = img.parentElement.getBoundingClientRect();
        if (r.bottom < 0 || r.top > vh) return;
        var progress = (r.top + r.height / 2 - vh / 2) / (vh + r.height);
        img.style.transform = "translate3d(0," + (-8 + progress * -14).toFixed(2) + "%,0)";
      });
      ticking = false;
    };
    update();
    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener("resize", update);
  }

  /* Scroll reveals */
  var revealEls = document.querySelectorAll(".reveal, .stagger");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.12 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("in"); });
  }

  /* Header shadow once the page scrolls */
  var header = document.querySelector(".site-header");
  if (header) {
    var onScroll = function () { header.classList.toggle("scrolled", window.scrollY > 8); };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
  }

  /* Highlight the nav link for the section in view */
  var navLinks = document.querySelectorAll('.nav a[href^="#"]');
  if (navLinks.length && "IntersectionObserver" in window) {
    var byId = {};
    navLinks.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = byId[entry.target.id];
        if (!link) return;
        if (entry.isIntersecting) {
          navLinks.forEach(function (a) { a.classList.remove("active"); });
          link.classList.add("active");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(byId).forEach(function (id) {
      var sec = document.getElementById(id);
      if (sec) spy.observe(sec);
    });
  }

  /* Mobile menu */
  var toggle = document.querySelector(".menu-toggle");
  if (toggle && header) {
    var label = toggle.querySelector(".sr-only");
    var setOpen = function (open) {
      header.classList.toggle("menu-open", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      label.textContent = open ? "Close menu" : "Open menu";
    };
    toggle.addEventListener("click", function () {
      setOpen(!header.classList.contains("menu-open"));
    });
    document.querySelectorAll(".mobile-nav a").forEach(function (a) {
      a.addEventListener("click", function () { setOpen(false); });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") setOpen(false);
    });
  }

  /* Share button: native share sheet on phones, copy link elsewhere */
  var shareBtn = document.getElementById("share-btn");
  var shareStatus = document.getElementById("share-status");
  if (shareBtn) {
    shareBtn.addEventListener("click", function () {
      var data = { title: document.title, url: window.location.href };
      if (navigator.share) {
        navigator.share(data).catch(function () {});
      } else if (navigator.clipboard) {
        navigator.clipboard.writeText(data.url).then(function () {
          shareStatus.textContent = "Link copied. Paste it anywhere to share.";
        }, function () {
          shareStatus.textContent = "Copy this page's address from your browser to share it.";
        });
      }
    });
  }

  var form = document.getElementById("catering-form");
  if (!form) return;
  var status = document.getElementById("form-status");

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var missing = Array.prototype.filter.call(form.querySelectorAll("[required]"), function (f) {
      return !f.value.trim();
    });
    if (missing.length) {
      status.textContent = "Please fill in your name, phone, event date, guests and location.";
      missing[0].focus();
      return;
    }
    var d = new FormData(form);
    var lines = [
      "Catering request from the website",
      "Name: " + d.get("name"),
      "Phone: " + d.get("phone"),
      d.get("email") ? "Email: " + d.get("email") : "",
      "Event: " + d.get("type"),
      "Date: " + d.get("date"),
      "Guests: " + d.get("guests"),
      "Location: " + d.get("where"),
      d.get("notes") ? "Notes: " + d.get("notes") : ""
    ].filter(Boolean).join("\n");

    window.location.href = "sms:+19545573972?&body=" + encodeURIComponent(lines);
    status.textContent = "Your request is ready to send in your texting app. If nothing opened, call or text us at (954) 557-3972.";
  });
})();
