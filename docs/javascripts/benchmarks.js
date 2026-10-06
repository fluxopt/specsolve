// The benchmark charts on about/benchmarks.md, drawn with TanStack Charts.
//
// The site uses instant navigation, so this module is loaded on every page and
// draws only where the page carries a `#benchmark-charts` block. It draws again
// on each `document$` emission and destroys the previous page's charts first,
// the way mathjax.js re-typesets. The library is imported on demand, through the
// import map in overrides/main.html, so no other page fetches it.

let hosts = [];

document$.subscribe(() => {
  hosts.forEach(h => h.destroy());
  hosts = [];
  const root = document.getElementById('benchmark-charts');
  if (root) mount(root);
});

async function mount(root) {
  const [
    { areaY }, { crosshair }, { dot }, { mountChart }, { lineY },
    { scaleLinear }, { defineChart }, { text }, { tooltip }, { scaleLog },
  ] = await Promise.all([
    import('@tanstack/charts/area.js'),
    import('@tanstack/charts/crosshair.js'),
    import('@tanstack/charts/dot.js'),
    import('@tanstack/charts/dom.js'),
    import('@tanstack/charts/line.js'),
    import('@tanstack/charts/scales/linear.js'),
    import('@tanstack/charts/scene.js'),
    import('@tanstack/charts/text.js'),
    import('@tanstack/charts/tooltip.js'),
    import('d3-scale'),
  ]);

  const SPEC = JSON.parse(root.querySelector('#bench-spec').textContent);
  const ROWS = await (await fetch(new URL(SPEC.data, location.href))).json();
  if (!root.isConnected) return;
  const { field: SERIES, refused: REFUSED, highlight: HERO, domain: LIBRARIES, floor: FLOOR } = SPEC.series;
  const X = SPEC.x.field;

  const trim = (v, d) => `${+v.toFixed(d)}`;
  const FORMAT = {
    seconds: v => v < 1 ? `${trim(v * 1000, v < 0.01 ? 1 : 0)} ms` : `${trim(v, 2)} s`,
    gigabytes: v => v < 1 ? `${trim(v * 1000, 0)} MB` : `${trim(v, 2)} GB`,
    count: v => v >= 1e6 ? `${trim(v / 1e6, 1)}M` : v >= 1e3 ? `${trim(v / 1e3, 1)}k` : `${v}`,
    ratio: v => v >= 10 ? `${v.toFixed(0)}×` : v >= 1 ? `${trim(v, 1)}×` : `${trim(v, 2)}×`,
  };

  const state = { metric: 'wall', filter: { ...SPEC.filter }, scale: 'log', hidden: new Set() };

  const panelOf = r => SPEC.facet.map(k => r[k]).join(' — ');
  const isHero = r => r[SERIES] === HERO;
  const isFloor = r => FLOOR.members.includes(r[SERIES]);
  const ink = name => SPEC.series.ink[LIBRARIES.indexOf(name)];

  /* The rows both views read: the current filter and visible libraries, each with the chosen
     metric as `value`, its band as `lo`/`hi`, and its ratio to the highlighted library at the
     same panel and size. */
  function rowsNow() {
    const m = SPEC.metrics[state.metric];
    const cell = r => `${panelOf(r)}|${r[X]}`;
    const base = new Map(ROWS.filter(r => isHero(r) && r[m.y] != null).map(r => [cell(r), r[m.y]]));
    return ROWS
      .filter(r => Object.entries(state.filter).every(([k, v]) => r[k] === v) && !state.hidden.has(r[SERIES]))
      .map(r => {
        const value = r[m.y] ?? null;
        if (value == null) return { ...r, value };
        const ours = base.get(cell(r));
        return { ...r, value, lo: m.band ? r[m.band[0]] : value, hi: m.band ? r[m.band[1]] : value,
          ratio: ours ? value / ours : null };
      });
  }

  /* Log ticks on 1–2–5 and a domain snapped to them, so a panel inside half a decade is not drawn
     on a two-decade axis. */
  function logTicks(values) {
    const lo = Math.min(...values), hi = Math.max(...values), all = [];
    for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
      for (const k of [1, 2, 5]) all.push(+(k * 10 ** e).toPrecision(12));
    const full = all.slice(all.findLastIndex(v => v <= lo), all.findIndex(v => v >= hi) + 1);
    const step = Math.ceil(full.length / 6);
    return full.filter((_, i) => i % step === 0 || i === full.length - 1);
  }

  /* One axis under the shared switch. Linear starts at zero. Log ticks on the given sizes when
     there are some, else on 1–2–5. */
  function axis(values, fmt, label, sizes = null) {
    if (state.scale === 'linear') {
      return { scale: scaleLinear().domain([0, Math.max(...values)]).nice(5), grid: true,
        axis: { label, ticks: { count: 5, format: fmt } } };
    }
    if (sizes) return { scale: scaleLog, axis: { label, ticks: { values: sizes, format: fmt } } };
    const ticks = logTicks(values);
    return { scale: scaleLog().domain([ticks[0], ticks.at(-1)]), grid: true, axis: { label, ticks: { values: ticks, format: fmt } } };
  }

  /* Where each refused size would land: the last measured point carried forward at the log–log
     slope of the last measured step, floored at 1 (linear), the rule the harness refuses by. Each
     library's run starts at that last point, so the dashed line joins its solid one. */
  function projections(rows) {
    const out = [];
    for (const own of Map.groupBy(rows, r => r[SERIES]).values()) {
      const measured = own.filter(r => r.value != null).sort((a, b) => a[X] - b[X]);
      const last = measured.at(-1), prev = measured.at(-2);
      const ahead = own.filter(r => r[REFUSED] && last && r[X] > last[X]).sort((a, b) => a[X] - b[X]);
      if (!ahead.length) continue;
      const slope = prev ? Math.log(last.value / prev.value) / Math.log(last[X] / prev[X]) : 1;
      const k = Math.max(slope, 1);
      out.push({ ...last, projected: false });
      for (const r of ahead) out.push({ ...r, value: last.value * (r[X] / last[X]) ** k, projected: true });
    }
    return out;
  }

  function grouped(fmt) {
    return {
      use: tooltip, sort: 'color-domain',
      content: points => ({
        title: `${FORMAT[SPEC.x.format](points[0].xValue)} ${SPEC.x.label}`,
        rows: [...new Map([...points].sort((a, b) => !!b.datum.projected - !!a.datum.projected)
          .map(p => [p.datum[SERIES], p])).values()]
          .map(p => ({ label: p.datum.projected ? `${p.datum[SERIES]} (projected)` : p.datum[SERIES],
            value: p.datum.projected ? `≈ ${fmt(p.datum.value)}` : fmt(p.datum.value), color: ink(p.datum[SERIES]) })),
      }),
    };
  }

  /* One panel: a line per library over size, the band under it, the projections dashed. The
     highlighted library is drawn last and thick, so it stays on top where the lines cross, and is
     named above its last measured point. The matrix floor is a thin neutral line without a band.
     Each measured point is a dot ringed in the surface colour, so
     crossing points stay apart; each projected size is a hollow ring in the library's colour. */
  function definition(rows, metric) {
    const fmt = FORMAT[metric.format];
    const measured = rows.filter(r => r.value != null);
    const ahead = projections(rows);
    const split = data => [data.filter(r => !isHero(r)), data.filter(isHero)];
    const [others, ours] = split(measured), [theirsAhead, oursAhead] = split(ahead);
    const theirs = others.filter(r => !isFloor(r)), floor = others.filter(isFloor);
    const enc = { x: X, z: SERIES, color: SERIES };
    const line = (data, strokeWidth) => lineY(data, { ...enc, y: 'value', strokeWidth });
    const points = (data, r) => dot(data, { ...enc, y: 'value', r, stroke: 'var(--surface)', strokeWidth: 2 });
    const hollow = [...Map.groupBy(ahead.filter(r => r.projected), r => r[SERIES])].map(([name, data]) =>
      dot(data, { x: X, y: 'value', r: isHero(data[0]) ? 4.5 : 3.5, fill: 'var(--surface)',
        stroke: SPEC.series.paint[LIBRARIES.indexOf(name)], strokeWidth: isHero(data[0]) ? 2.5 : 1.5 }));
    const label = ours.length ? [text([ours.at(-1)], { x: X, y: 'value', text: () => HERO, anchor: 'end', dx: -8, dy: -10,
      fill: ink(HERO), fontSize: 12, fontWeight: 650 })] : [];
    const dashed = (data, strokeWidth) => lineY(data, { ...enc, y: 'value', strokeWidth, strokeDasharray: '5 5' });
    const area = (data, fillOpacity) => areaY(data, { ...enc, y1: 'lo', y2: 'hi', fillOpacity });
    const sizes = [...new Set([...measured, ...ahead].map(r => r[X]))].sort((a, b) => a - b);
    return defineChart({
      marks: [
        ...(metric.band ? [area(theirs, 0.35), area(ours, 0.22)] : []),
        dashed(theirsAhead, 2), dashed(oursAhead, 3),
        line(floor, 1.5), line(theirs, 2), line(ours, 3.5),
        ...hollow,
        points(floor, 2.5), points(theirs, 3.5), points(ours, 5),
        ...label,
        crosshair({ y: false }),
      ],
      scales: {
        x: axis(sizes, FORMAT[SPEC.x.format], SPEC.x.label, sizes),
        y: axis(measured.flatMap(r => [r.lo, r.hi]), fmt),
      },
      color: { domain: LIBRARIES, range: SPEC.series.paint },
      clip: true,
      focus: 'group-x', maxFocusDistance: Number.POSITIVE_INFINITY,
      tooltip: grouped(fmt),
    });
  }

  /* Diverging bins on the ratio to the highlighted library: green where it is faster, blue where
     the other library is, grey within ten per cent. Every cell also prints its ratio, which carries
     the reading where blue and green merge (tritanopia). */
  const BINS = [[1 / 4, 'b4'], [1 / 2, 'b3'], [1 / 1.25, 'b2'], [1 / 1.1, 'b1'], [1.1, 'mid'], [1.25, 'g1'], [2, 'g2'], [4, 'g3'], [Infinity, 'g4']];
  const shade = v => {
    const b = BINS.find(([edge]) => v < edge)[1];
    return `background:var(--d-${b});${b.endsWith('4') ? 'color:var(--d-ink4)' : ''}`;
  };

  function table(rows, metric) {
    const fmt = FORMAT[metric.format];
    const others = LIBRARIES.filter(n => n !== HERO && !FLOOR.members.includes(n) && rows.some(r => r[SERIES] === n));
    const floorShown = rows.some(isFloor);
    const body = [];
    for (const [title, panel] of Map.groupBy(rows, panelOf)) {
      const guess = new Map(projections(panel).filter(r => r.projected).map(r => [`${r[SERIES]}|${r[X]}`, r.value]));
      const cell = (r, kind = '') => {
        if (!r) return '<td class="none">—</td>';
        if (r.value == null) {
          if (!r[REFUSED]) return '<td class="none">—</td>';
          const g = guess.get(`${r[SERIES]}|${r[X]}`);
          const why = `refused by the ${r[REFUSED].slice(1)} budget`;
          return `<td class="none" title="${g ? `${why} · projected ≈ ${fmt(g)}` : why}">${r[REFUSED]}</td>`;
        }
        if (kind === 'own') return `<td class="own">${fmt(r.value)}</td>`;
        if (r.ratio == null) return `<td class="none">${fmt(r.value)}</td>`;
        const look = kind === 'floor' ? 'class="floor"' : `style="${shade(r.ratio)}"`;
        return `<td ${look} title="${r[SERIES]}: ${fmt(r.value)}">${FORMAT.ratio(r.ratio)}</td>`;
      };
      const sizes = [...new Set(panel.filter(isHero).map(r => r[X]))].sort((a, b) => a - b);
      sizes.forEach((size, i) => {
        const at = new Map(panel.filter(r => r[X] === size).map(r => [r[SERIES], r]));
        body.push(`<tr class="${i ? '' : 'first'}"><td class="panel">${i ? '' : title}</td>` +
          `<td>${FORMAT[SPEC.x.format](size)}</td>${cell(at.get(HERO), 'own')}${others.map(n => cell(at.get(n))).join('')}` +
          `${floorShown ? cell(FLOOR.members.map(n => at.get(n)).find(Boolean), 'floor') : ''}</tr>`);
      });
    }
    root.querySelector('#bench-heat').innerHTML =
      `<thead><tr><th>panel</th><th>${SPEC.x.label}</th><th>${HERO}</th>${others.map(n => `<th>${n}</th>`).join('')}` +
      `${floorShown ? `<th class="floor">${FLOOR.label}</th>` : ''}</tr></thead>` +
      `<tbody>${body.join('')}</tbody>`;
  }

  root.querySelector('#bench-scale').innerHTML = `<span>other library faster</span>` +
    ['b4', 'b3', 'b2', 'b1', 'mid', 'g1', 'g2', 'g3', 'g4'].map(b => `<i style="background:var(--d-${b})"></i>`).join('') +
    `<span>${HERO} faster</span><span>4× · 2× · 1.25× · 1.1× · even · 1.1× · 1.25× · 2× · 4×</span>`;

  function draw() {
    hosts.forEach(h => h.destroy());
    hosts = [];
    const metric = SPEC.metrics[state.metric];
    const rows = rowsNow();
    const grid = root.querySelector('#bench-facets');
    grid.replaceChildren();
    for (const [title, panel] of Map.groupBy(rows, panelOf)) {
      if (!panel.some(r => r.value != null)) continue;
      const fig = document.createElement('div');
      fig.className = 'bench-panel';
      fig.innerHTML = `<div class="bench-caption">${title}</div><div></div>`;
      grid.appendChild(fig);
      hosts.push(mountChart(fig.lastElementChild, { definition: definition(panel, metric), height: 240,
        ariaLabel: `${title}: ${metric.label} against model size, one line per ${SERIES}` }));
    }
    table(rows, metric);
    const present = new Set(ROWS.map(r => r[SERIES]));
    const floorHidden = FLOOR.members.every(n => state.hidden.has(n));
    root.querySelector('#bench-legend').innerHTML = LIBRARIES.filter(n => present.has(n) && !FLOOR.members.includes(n)).map(n =>
      `<button data-series="${n}" aria-pressed="${!state.hidden.has(n)}"${n === HERO ? ` disabled title="Every other line and cell is read against ${HERO}"` : ''}>` +
      `<i class="swatch${n === HERO ? ' hero' : ''}" style="background:${SPEC.series.paint[LIBRARIES.indexOf(n)]}"></i>${n}</button>`).join('') +
      `<button data-floor aria-pressed="${!floorHidden}"><i class="swatch" style="background:var(--floor)"></i>${FLOOR.label}</button>`;
  }

  root.addEventListener('click', ev => {
    const lib = ev.target.closest('#bench-legend button');
    if (lib && !lib.disabled) {
      const names = 'floor' in lib.dataset ? FLOOR.members : [lib.dataset.series];
      const show = names.every(n => state.hidden.has(n));
      names.forEach(n => (show ? state.hidden.delete(n) : state.hidden.add(n)));
      return draw();
    }
    const b = ev.target.closest('.bench-seg button');
    if (!b) return;
    const key = b.parentElement.dataset.key;
    if (key === 'metric' || key === 'scale') state[key] = b.dataset.value;
    else state.filter[key] = b.dataset.value;
    b.parentElement.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
    draw();
  });

  draw();
}
