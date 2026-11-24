// A picture marked `zoom` opens full size over the page when clicked. A click
// anywhere, or Escape, closes it. Without this script the link opens the
// picture on its own.
(function () {
  function close() {
    const view = document.querySelector(".zoom-view");
    if (view) view.remove();
  }

  function open(link) {
    close();
    const view = document.createElement("div");
    view.className = "zoom-view";
    const picture = document.createElement("img");
    picture.src = link.href;
    const small = link.querySelector("img");
    picture.alt = small ? small.alt : "";
    view.appendChild(picture);
    view.addEventListener("click", close);
    document.body.appendChild(view);
  }

  function setUp() {
    document.querySelectorAll("a.zoom").forEach((link) => {
      if (link.dataset.ready) return;
      link.dataset.ready = "1";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        open(link);
      });
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") close();
  });
  if (typeof document$ !== "undefined") document$.subscribe(setUp);
  else document.addEventListener("DOMContentLoaded", setUp);
})();
