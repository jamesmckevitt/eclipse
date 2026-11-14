// What the reader has, kept for the whole site. Once picked on the
// home page, a compact picker sits in the header on every page, and the menu
// entries and page sections that do not fit the route go faint.
(function () {
  const KEY = "eclipse-route";
  const ROUTES = {
    spectra: {
      label: "Synthetic spectrum", name: "a synthetic spectrum",
      icon: '<svg viewBox="0 0 24 24"><path d="M9.96 11.31C10.82 8.1 11.5 6 13 6s2.18 2.1 3.04 5.31C17 14.92 18.1 19 22 19v-2c-2.2 0-3-2.46-4.03-6.2C17.08 7.46 16.15 4 13 4s-4.08 3.46-4.97 6.8C7.03 14.54 6.2 17 4 17V2H2v20h20v-2H4v-1c3.9 0 5-4.08 5.96-7.69"/></svg>',
    },
    mhd: {
      label: "MHD simulation", name: "an MHD simulation",
      icon: '<svg viewBox="0 0 24 24"><path d="M21 16.5c0 .38-.21.71-.53.88l-7.9 4.44c-.16.12-.36.18-.57.18s-.41-.06-.57-.18l-7.9-4.44A.99.99 0 0 1 3 16.5v-9c0-.38.21-.71.53-.88l7.9-4.44c.16-.12.36-.18.57-.18s.41.06.57.18l7.9 4.44c.32.17.53.5.53.88zM12 4.15 6.04 7.5 12 10.85l5.96-3.35zM5 15.91l6 3.38v-6.71L5 9.21zm14 0v-6.7l-6 3.37v6.71z"/></svg>',
    },
    dem: {
      label: "DEM or VDEM", name: "a DEM or VDEM",
      icon: '<svg viewBox="0 0 24 24"><path d="M3 3h2v10h4V7h4v4h4v4h4v6H3z"/></svg>',
    },
    one: {
      label: "Single intensity", name: "a single intensity",
      icon: '<svg viewBox="0 0 24 24"><path d="M12 8a4 4 0 0 0-4 4 4 4 0 0 0 4 4 4 4 0 0 0 4-4 4 4 0 0 0-4-4"/></svg>',
    },
  };

  // The routes each page serves; a page not listed serves every route.
  const PAGES = {
    "emission/mhd/": "mhd",
    "observe/rasters/": "mhd spectra",
    "emission/dem/": "dem",
    "emission/other-codes/": "spectra",
    "emission/single-intensity/": "one",
    "observe/full-ccd-frames/": "spectra mhd dem",
    "analyse/frame/": "spectra mhd dem",
    "files/atmosphere/": "mhd",
    "files/synthesis/": "mhd dem spectra",
    "files/frame/": "spectra mhd dem",
    "files/band-lines/": "spectra mhd dem",
    "files/emission-measure/": "dem",
  };

  function stored() {
    try {
      const route = localStorage.getItem(KEY);
      return ROUTES[route] ? route : null;
    } catch (error) {
      return null;
    }
  }

  // Store the route for the whole site and tell the page; null clears it.
  function setRoute(route) {
    try {
      if (route) localStorage.setItem(KEY, route);
      else localStorage.removeItem(KEY);
    } catch (error) {
      // Without storage the route lasts only for this page.
    }
    window.eclipseRoute = route;
    window.dispatchEvent(new CustomEvent("eclipse-route", { detail: route }));
    apply();
  }
  window.eclipseSetRoute = setRoute;

  const routesOf = (path) => {
    const key = Object.keys(PAGES).find((page) => path.endsWith("/" + page) || path === "/" + page);
    return key ? PAGES[key].split(" ") : null;
  };

  // The picker in the header, made once per page.
  function picker() {
    let box = document.querySelector(".route-picker");
    if (box) return box;
    const title = document.querySelector(".md-header__title");
    if (!title) return null;
    box = document.createElement("div");
    box.className = "route-picker";
    box.innerHTML = '<span class="route-label">You have</span>'
      + ["one", "spectra", "mhd", "dem"].map((id) => [id, ROUTES[id]]).map(([id, r]) =>
        `<button type="button" class="route-option" data-route="${id}" title="${r.label}">`
        + `<span class="route-icon">${r.icon}</span><span class="route-text">${r.label}</span></button>`).join("")
      + '<button type="button" class="route-clear" title="Show everything">&times;</button>';
    title.after(box);
    box.addEventListener("click", (event) => {
      const option = event.target.closest(".route-option");
      if (option) setRoute(option.dataset.route === current() ? null : option.dataset.route);
      if (event.target.closest(".route-clear")) setRoute(null);
    });
    return box;
  }

  const current = () => (window.eclipseRoute !== undefined ? window.eclipseRoute : stored());

  // A heading tagged with the routes it serves stands for its whole section:
  // everything after it up to the next heading of the same or a higher level.
  function section(heading) {
    const level = Number(heading.tagName[1]);
    const parts = [heading];
    for (let node = heading.nextElementSibling; node; node = node.nextElementSibling) {
      if (/^H[1-6]$/.test(node.tagName) && Number(node.tagName[1]) <= level) break;
      parts.push(node);
    }
    return parts;
  }

  function apply() {
    const route = current();
    const box = picker();
    if (box) {
      box.hidden = !route;
      box.querySelectorAll(".route-option").forEach((b) => b.classList.toggle("is-active", b.dataset.route === route));
    }

    // The menu: pages that do not serve the route go faint.
    document.querySelectorAll(".md-nav--primary a.md-nav__link[href]").forEach((a) => {
      const serves = routesOf(new URL(a.href, location.href).pathname);
      a.classList.toggle("route-off", !!route && !!serves && !serves.includes(route));
    });

    // The page: a note at the top if the page does not serve the route.
    const content = document.querySelector(".md-content__inner");
    if (!content) return;
    const serves = routesOf(location.pathname);
    let note = content.querySelector(".route-note");
    const off = !!route && !!serves && !serves.includes(route);
    if (off && !note) {
      note = document.createElement("p");
      note.className = "route-note";
      const h1 = content.querySelector("h1");
      (h1 || content.firstChild).after(note);
    }
    if (note) {
      note.hidden = !off;
      if (off) {
        note.innerHTML = `You do not need this page with ${ROUTES[route].name}. <a href="#">Show everything</a>`;
        note.querySelector("a").addEventListener("click", (event) => { event.preventDefault(); setRoute(null); });
      }
    }
    content.classList.toggle("route-page-off", off);

    // The sections: those tagged for other routes go faint, with their
    // entries in the table of contents.
    content.querySelectorAll("[data-routes]").forEach((heading) => {
      if (!/^H[2-6]$/.test(heading.tagName)) return;
      const offSection = !!route && !heading.dataset.routes.split(" ").includes(route);
      section(heading).forEach((node) => {
        node.classList.toggle("route-off", offSection);
        // The theme gives the contents links full addresses, so match on
        // the part after the "#".
        if (node.id) {
          document.querySelectorAll(".md-nav--secondary a[href]").forEach((a) => {
            if (new URL(a.href, location.href).hash === "#" + node.id) a.classList.toggle("route-off", offSection);
          });
        }
      });
    });
  }

  function setUp() {
    if (window.eclipseRoute === undefined) window.eclipseRoute = stored();
    picker();
    apply();
  }

  if (typeof document$ !== "undefined") document$.subscribe(setUp);
  else document.addEventListener("DOMContentLoaded", setUp);
})();
