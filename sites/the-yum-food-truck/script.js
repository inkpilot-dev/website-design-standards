(function () {
  var year = document.querySelectorAll(".year");
  year.forEach(function (el) { el.textContent = new Date().getFullYear(); });

  var today = new Date().getDay();
  var row = document.querySelector('#week li[data-day="' + today + '"]');
  if (row) row.classList.add("today");

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
