// The diagram on the home page. A "What do you have?" button picks a route,
// and the boxes, files and lines on it are lit step by step. Each later step
// with only one box to pick is filled in. A route whose step 2 leads to
// nothing in step 3 is complete after step 2.
(function () {
  function setUp() {
    const guide = document.querySelector(".guide[data-v=\"3\"]");
    if (!guide || guide.dataset.ready) return;
    guide.dataset.ready = "1";

    const boxes = Array.from(guide.querySelectorAll(".card"));
    const lines = Array.from(guide.querySelectorAll(".links path"));
    const labels = Array.from(guide.querySelectorAll(".label[data-tier]"));
    const chosen = [null, null, null];
    // What the reader has, from the "What do you have?" buttons. Without one,
    // the box picked in step 1 gives the route it names.
    let have = null;
    const haveBox = {};
    const boxRoute = {};
    boxes.forEach((b) => {
      if (b.dataset.route) {
        haveBox[b.dataset.route] = b.dataset.id;
        boxRoute[b.dataset.id] = b.dataset.route;
      }
    });
    const route = () => have || boxRoute[chosen[0]] || null;
    const haveButtons = Array.from(guide.querySelectorAll(".have-button"));
    const global = guide.hasAttribute("data-global-route");

    const box = (id) => boxes.find((b) => b.dataset.id === id);
    const tierOf = (id) => Number(box(id).dataset.tier) - 1;
    const files = Array.from(guide.querySelectorAll(".file"));
    const file = (id) => files.find((f) => f.dataset.id === id);

    // A file between two steps is on the path once the boxes on both sides
    // of it are picked, and lit while it leads from the box picked above to
    // a box that can still be picked below.
    function rawFileState(f) {
      // A file that belongs to another route is never on this one.
      if (f.dataset.have && route() && !f.dataset.have.split(" ").includes(route())) return "";
      const from = f.dataset.from.split(" ").filter(Boolean);
      const to = f.dataset.to.split(" ");
      // A file that goes into step 1, such as an atmosphere file, is on the
      // path once a box it goes into is picked, and lit until step 1 is picked.
      if (!from.length) {
        if (to.includes(chosen[0])) return "chosen";
        return chosen[0] ? "" : "lit";
      }
      const tier = tierOf(from[0]);
      if (!from.includes(chosen[tier])) return "";
      if (chosen[tier] && reach(chosen[tier]).length === 0) return "chosen";
      if (to.includes(chosen[tier + 1])) return "chosen";
      return to.some((id) => offered(tier + 1).includes(id)) ? "lit" : "";
    }

    // When more than one file could carry the path, as a single snapshot or
    // a series can for a raster, the first is shown as the path and the
    // others as ways it could go instead.
    function fileState(f) {
      const state = rawFileState(f);
      if (state !== "chosen" || !f.dataset.from) return state;
      const row = (g) => (g.dataset.from ? tierOf(g.dataset.from.split(" ")[0]) : -1);
      const first = files.find((g) => row(g) === row(f) && rawFileState(g) === "chosen");
      return first === f ? "chosen" : "lit";
    }

    function boxState(id) {
      const tier = tierOf(id);
      if (chosen[tier] === id) return "chosen";
      if (tier === 0) return chosen[0] ? "" : "lit";
      return chosen[tier - 1] && offered(tier).includes(id) ? "lit" : "";
    }

    const stateOf = (id) => (file(id) ? fileState(file(id)) : boxState(id));
    const reach = (id) => {
      if (!id) return [];
      return box(id).dataset.to.split(" ").filter(Boolean);
    };
    // The path ends at step 2 when the box picked there leads nowhere.
    const endsEarly = () => !!chosen[1] && reach(chosen[1]).length === 0;
    const complete = () => !!chosen[2] || endsEarly();
    const guided = () => guide.classList.contains("guided");

    boxes.forEach((b) => {
      if (b.dataset.why) {
        const why = document.createElement("p");
        why.className = "why";
        why.textContent = b.dataset.why;
        b.appendChild(why);
      }
    });

    function offered(tier) {
      if (tier === 0) return boxes.filter((b) => b.dataset.tier === "1").map((b) => b.dataset.id);
      return reach(chosen[tier - 1]);
    }

    function keepValid() {
      for (let tier = 1; tier < 3; tier++) {
        if (chosen[tier] && !offered(tier).includes(chosen[tier])) chosen[tier] = null;
      }
    }

    function render() {
      keepValid();
      const on = guided();
      guide.dataset.path = on ? chosen[0] || "" : "";
      haveButtons.forEach((b) => b.classList.toggle("is-active", on && have === b.dataset.have));

      // Where the page asks for it, a link that suits only some routes goes
      // faint on the others, and on a path through a box it does not suit.
      if (guide.hasAttribute("data-fade-links")) {
        guide.querySelectorAll("a[data-routes], a[data-without]").forEach((a) => {
          const offRoute = on && (
            (a.dataset.routes && !a.dataset.routes.split(" ").includes(route()))
            || (a.dataset.without && a.dataset.without.split(" ").some((id) => chosen.includes(id))));
          a.classList.toggle("is-off-route", !!offRoute);
          if (offRoute) a.setAttribute("tabindex", "-1");
          else a.removeAttribute("tabindex");
        });
      }
      const active = complete() ? -1 : chosen.indexOf(null);

      boxes.forEach((b) => {
        const tier = Number(b.dataset.tier) - 1;
        const id = b.dataset.id;
        const isChosen = on && chosen[tier] === id;
        const isOffered = on && offered(tier).includes(id);
        b.classList.toggle("is-chosen", isChosen);
        b.classList.toggle("is-open", isOffered && !isChosen && !chosen[tier]);
        b.classList.toggle("is-other", isOffered && !isChosen && !!chosen[tier]);
        b.classList.toggle("is-blocked", on && !isOffered && tier > 0 && !!chosen[tier - 1]);
        const why = b.querySelector(".why");
        if (why && tier > 0 && chosen[tier - 1]) {
          const own = b.dataset["why" + chosen[tier - 1][0].toUpperCase() + chosen[tier - 1].slice(1)];
          why.textContent = own || b.dataset.why;
        }
        if (on) {
          b.setAttribute("role", "button");
          b.setAttribute("tabindex", isOffered ? "0" : "-1");
          b.setAttribute("aria-pressed", isChosen ? "true" : "false");
          b.setAttribute("aria-disabled", isOffered ? "false" : "true");
        } else {
          ["role", "tabindex", "aria-pressed", "aria-disabled"].forEach((a) => b.removeAttribute(a));
        }
      });

      labels.forEach((l) => {
        const tier = Number(l.dataset.tier) - 1;
        l.classList.toggle("is-active", on && tier === active);
        l.classList.toggle("is-done", on && (!!chosen[tier] || (tier === 2 && endsEarly())));
      });

      files.forEach((f) => {
        const state = on ? fileState(f) : "";
        f.classList.toggle("is-chosen", state === "chosen");
        f.classList.toggle("is-lit", state === "lit");
      });

      lines.forEach((p) => {
        const from = stateOf(p.dataset.from);
        const to = stateOf(p.dataset.to);
        // From a box, a line leads on only once the box is picked; from a
        // file, once the file is on the way.
        const leads = file(p.dataset.from) ? from !== "" : from === "chosen";
        p.classList.toggle("is-path", on && from === "chosen" && to === "chosen");
        p.classList.toggle("is-lit", on && leads && to !== "" && !(from === "chosen" && to === "chosen"));
        // A line that belongs to one route only shows on that route.
        p.classList.toggle("is-hidden", !!p.dataset.have && !p.dataset.have.split(" ").includes(route()));
      });

    }

    function bringIntoView(target, block) {
      const r = target.getBoundingClientRect();
      if (r.bottom > window.innerHeight || r.top < 0) {
        target.scrollIntoView({ behavior: "smooth", block: block });
      }
    }

    // Fill in each later step that has only one box to pick.
    function fillOnlyChoices(tier) {
      for (let next = tier + 1; next < 3; next++) {
        if (chosen[next]) continue;
        const options = offered(next);
        if (options.length !== 1) break;
        chosen[next] = options[0];
      }
    }

    // Open the guide on a route, or close it when there is none.
    function setRoute(next, quiet) {
      have = next;
      chosen.fill(null);
      if (have) {
        chosen[0] = haveBox[have];
        fillOnlyChoices(0);
      }
      guide.classList.toggle("guided", !!have);
      render();
      // On a page that shares the route with the whole site, tell the rest.
      if (global && !quiet && window.eclipseSetRoute && window.eclipseRoute !== have) window.eclipseSetRoute(have);
    }

    function choose(b) {
      const tier = Number(b.dataset.tier) - 1;
      // A greyed-out box does nothing when clicked.
      if (b.getAttribute("aria-disabled") === "true") return;
      if (tier === 0) {
        // Picking a box in step 1 takes that box's usual route; unpicking
        // it closes the guide.
        setRoute(chosen[0] === b.dataset.id ? null : boxRoute[b.dataset.id]);
        return;
      }
      chosen[tier] = chosen[tier] === b.dataset.id ? null : b.dataset.id;
      if (chosen[tier]) fillOnlyChoices(tier);
      render();
      if (complete()) return;
      if (chosen.indexOf(null) > tier) bringIntoView(guide.querySelectorAll(".tier")[chosen.indexOf(null)], "nearest");
    }

    boxes.forEach((b) => {
      b.addEventListener("click", (event) => {
        if (!guided()) return;
        // A link in a box still goes to its page.
        if (event.target.closest("a")) return;
        event.preventDefault();
        choose(b);
      });
      b.addEventListener("keydown", (event) => {
        if (!guided() || event.target !== b) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          choose(b);
        }
      });
    });

    // A "What do you have?" button opens the guide on its route; pressing it again
    // closes the guide.
    haveButtons.forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        setRoute(guided() && have === button.dataset.have ? null : button.dataset.have);
      });
    });

    render();

    // A route picked elsewhere on the site, or in the header, opens here too.
    if (global) {
      let saved = null;
      try {
        saved = localStorage.getItem("eclipse-route");
      } catch (error) {
        saved = null;
      }
      if (saved && haveBox[saved]) setRoute(saved, true);
      if (window.eclipseGuideListener) window.removeEventListener("eclipse-route", window.eclipseGuideListener);
      window.eclipseGuideListener = (event) => {
        if (document.contains(guide) && event.detail !== have) setRoute(event.detail, true);
      };
      window.addEventListener("eclipse-route", window.eclipseGuideListener);
    }
  }

  if (typeof document$ !== "undefined") document$.subscribe(setUp);
  else document.addEventListener("DOMContentLoaded", setUp);
})();
