// The benchmark charts on about/benchmarks.md, drawn with TanStack Charts.
//
// The site uses instant navigation, so this module is loaded on every page and
// draws only where the page carries a `#benchmark-charts` block. It draws again
// on each `document$` emission and destroys the previous page's charts first,
// the way mathjax.js re-typesets. The library is imported on demand, through the
// import map in overrides/main.html, so no other page fetches it.

let hosts = [];

/* Diverging bins on the ratio to the highlighted library: `o` (ours, purple, the highlighted
   library's own hue) where it is faster, `t` (theirs, orange) where the other library is, grey
   within ten per cent. Every cell also prints its ratio. */
const BINS = [[1 / 4, 't4'], [1 / 2, 't3'], [1 / 1.25, 't2'], [1 / 1.1, 't1'], [1.1, 'mid'], [1.25, 'o1'], [2, 'o2'], [4, 'o3'], [Infinity, 'o4']];
const shade = v => {
  const b = BINS.find(([edge]) => v < edge)[1];
  return `background:var(--d-${b});${b.endsWith('4') ? 'color:var(--d-ink4)' : ''}`;
};

/* A table bench.report writes as HTML carries each cell's ratio in `data-ratio`, and is shaded
   here with the same bins, so the bins have one home. A row of the matrix floor or of specsolve's
   own other lane stays neutral, as in the charts' own table: neither is a rival. Both lists come
   from the page's chart spec. */
function shadeWritten() {
  const spec = document.getElementById('bench-spec');
  const series = spec ? JSON.parse(spec.textContent).series : null;
  const neutral = series ? [...series.floor.members, ...series.internal] : [];
  document.querySelectorAll('table.bench-heat tr').forEach(tr => {
    const plain = neutral.includes(tr.cells[0]?.textContent.trim());
    tr.querySelectorAll('td[data-ratio]').forEach(td => {
      if (plain) td.className = 'neutral';
      else td.setAttribute('style', shade(+td.dataset.ratio));
    });
  });
}

document$.subscribe(() => {
  hosts.forEach(h => h.destroy());
  hosts = [];
  shadeWritten();
  const root = document.getElementById('benchmark-charts');
  if (root) mount(root);
});

async function mount(root) {
  const [
    { areaY }, { crosshair }, { dot }, { mountChart }, { lineY }, { link },
    { scaleLinear }, { defineChart }, { tooltip }, { scaleLog },
  ] = await Promise.all([
    import('@tanstack/charts/area.js'),
    import('@tanstack/charts/crosshair.js'),
    import('@tanstack/charts/dot.js'),
    import('@tanstack/charts/dom.js'),
    import('@tanstack/charts/line.js'),
    import('@tanstack/charts/link.js'),
    import('@tanstack/charts/scales/linear.js'),
    import('@tanstack/charts/scene.js'),
    import('@tanstack/charts/tooltip.js'),
    import('d3-scale'),
  ]);

  const SPEC = JSON.parse(root.querySelector('#bench-spec').textContent);
  const ROWS = await (await fetch(new URL(SPEC.data, location.href))).json();
  if (!root.isConnected) return;
  const { field: SERIES, refused: REFUSED, highlight: HERO, domain: LIBRARIES, floor: FLOOR, internal: INTERNAL } = SPEC.series;
  const X = SPEC.x.field;

  const trim = (v, d) => `${+v.toFixed(d)}`;
  const FORMAT = {
    seconds: v => v < 1 ? `${trim(v * 1000, v < 0.01 ? 1 : 0)} ms` : `${trim(v, 2)} s`,
    gigabytes: v => v < 1 ? `${trim(v * 1000, 0)} MB` : `${trim(v, 2)} GB`,
    count: v => v >= 1e6 ? `${trim(v / 1e6, 1)}M` : v >= 1e3 ? `${trim(v / 1e3, 1)}k` : `${v}`,
    ratio: v => v >= 10 ? `${v.toFixed(0)}×` : v >= 1 ? `${trim(v, 1)}×` : `${trim(v, 2)}×`,
  };

  /* The size range is chosen in decades, since every ladder steps tenfold: `dispatch` at 10k and
     `transport` at 9.8k are one choice. Below the smallest is zero, which cuts nothing and keeps a
     linear axis on its origin. */
  const decade = v => Math.round(Math.log10(v));
  const STOPS = [-Infinity, ...new Set(ROWS.map(r => decade(r[X])).sort((a, b) => a - b))];
  const state = { metric: 'wall', view: 'absolute', filter: { ...SPEC.filter }, scale: 'log', hidden: new Set(),
    range: { from: STOPS[0], to: STOPS.at(-1) } };
  const inRange = r => decade(r[X]) >= state.range.from && decade(r[X]) <= state.range.to;
  const chosen = r => Object.entries(state.filter).every(([k, v]) => r[k] === v);

  const panelOf = r => SPEC.facet.map(k => r[k]).join(' — ');
  const isHero = r => r[SERIES] === HERO;
  const isFloor = r => FLOOR.members.includes(r[SERIES]);
  const ink = name => SPEC.series.ink[LIBRARIES.indexOf(name)];

  /* The rows the charts and the table read: the current filter and visible libraries, each with
     the chosen metric as `value`, its middle half as `lo`/`hi`, the highlighted library's value at
     the same filter, panel and size as `ours`, and the ratio of the two. */
  function rowsNow() {
    const m = SPEC.metrics[state.metric];
    const cell = r => `${panelOf(r)}|${r[X]}`;
    const base = new Map(ROWS.filter(r => chosen(r) && isHero(r) && r[m.y] != null).map(r => [cell(r), r[m.y]]));
    return ROWS
      .filter(r => chosen(r) && !state.hidden.has(r[SERIES]))
      .map(r => {
        const value = r[m.y] ?? null, ours = base.get(cell(r)) ?? null;
        if (value == null) return { ...r, value, ours };
        return { ...r, value, ours, lo: m.band ? r[m.band[0]] : value, hi: m.band ? r[m.band[1]] : value,
          ratio: ours ? value / ours : null };
      });
  }

  /* The points as the chosen view draws them: as measured, or divided by the highlighted library's
     value at the same panel and size, with the measured value kept as `raw`. A point with nothing to
     divide by is left out of the relative view. */
  function viewed(points) {
    if (state.view === 'absolute') return points;
    return points.filter(p => p.ours).map(p => ({ ...p, raw: p.value, value: p.value / p.ours,
      lo: p.lo / p.ours, hi: p.hi / p.ours }));
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

  /* The y axis under the shared switch. Linear starts at zero, log ticks on 1–2–5. */
  function axis(values, fmt, label) {
    if (state.scale === 'linear') {
      return { scale: scaleLinear().domain([0, Math.max(...values)]).nice(5), grid: true,
        axis: { label, ticks: { count: 5, format: fmt } } };
    }
    const ticks = logTicks(values);
    return { scale: scaleLog().domain([ticks[0], ticks.at(-1)]), grid: true, axis: { label, ticks: { values: ticks, format: fmt } } };
  }

  /* The x axis over the sizes in view, with room past the outermost so no dot sits on the frame,
     and a single size in the middle. Linear starts at zero when `fromZero`, else below the smallest
     size by at most half of it, so it stays off zero; log ticks on the sizes themselves. */
  function xAxis(sizes, fromZero) {
    const label = SPEC.x.label, format = FORMAT[SPEC.x.format];
    const lo = sizes[0], hi = sizes.at(-1), one = sizes.length === 1;
    if (state.scale === 'linear') {
      const pad = one ? hi : (hi - (fromZero ? 0 : lo)) * 0.06;
      return { scale: scaleLinear().domain([fromZero || one ? 0 : lo - Math.min(pad, lo / 2), hi + pad]), grid: true,
        axis: { label, ticks: { count: 5, format } } };
    }
    const pad = one ? 3 : 1.25;
    return { scale: scaleLog().domain([lo / pad, hi * pad]), axis: { label, ticks: { values: sizes, format } } };
  }

  /* Where each refused size could land, as a range from `lo` to `hi`: between two continuations
     of the library's last measured step, one straight at the step's cost per variable and one
     curved at the step's growth rate. Both run through the step's two points and part with
     distance, so the range is narrow where they agree and wide where the step says little.
     `value` is the range's geometric middle. Each library's run starts at its last measured point
     with no width, so the funnel opens from the solid line. A library with one measured size has
     no step, and nothing is projected. */
  function projections(rows) {
    const out = [];
    for (const own of Map.groupBy(rows, r => r[SERIES]).values()) {
      const [prev, last] = own.filter(r => r.value != null).sort((a, b) => a[X] - b[X]).slice(-2);
      const ahead = own.filter(r => r[REFUSED] && last && r[X] > last[X]).sort((a, b) => a[X] - b[X]);
      if (!ahead.length || !prev) continue;
      const perVariable = (last.value - prev.value) / (last[X] - prev[X]);
      const growth = Math.log(last.value / prev.value) / Math.log(last[X] / prev[X]);
      out.push({ ...last, lo: last.value, hi: last.value, projected: false });
      for (const r of ahead) {
        const [lo, hi] = [last.value + perVariable * (r[X] - last[X]), last.value * (r[X] / last[X]) ** growth]
          .sort((a, b) => a - b);
        out.push({ ...r, value: Math.sqrt(lo * hi), lo, hi, projected: true });
      }
    }
    return out;
  }

  /* A tooltip's number columns. Each row lays out on its own, value right-aligned in tabular
     digits, so a column lines up only if every entry has one shape: one unit for the column (the
     smallest value's), fixed decimals, and figure spaces, a digit wide, padding the left. Each
     entry is one value or a projection's range. */
  const UNITS = { seconds: [['ms', 1e-3], ['s', 1]], gigabytes: [['MB', 1e-3], ['GB', 1]] };
  const padded = texts => texts.map(t => t.padStart(Math.max(...texts.map(x => x.length)), '\u2007'));
  function amounts(spans, format) {
    const low = Math.min(...spans.flat());
    const [unit, scale] = UNITS[format][low < 1 ? 0 : 1];
    const digits = low / scale < 1 ? 2 : low / scale < 10 ? 1 : 0;
    return padded(spans.map(vs => `${vs.map(v => (v / scale).toFixed(digits)).join('–')} ${unit}`));
  }

  /* One row per library at the hovered size, slowest first as the lines stack in the panel: its
     value, then its ratio to the highlighted library, whose row is emphasised. A projection stands
     in for a library only where the library has no measurement at that size, and gives its range. */
  function grouped(format) {
    return {
      use: tooltip,
      content: points => {
        const at = [...new Map([...points].sort((a, b) => !!b.datum.projected - !!a.datum.projected)
          .map(p => [p.datum[SERIES], p.datum])).values()].sort((a, b) => b.value - a.value);
        const ours = at.find(isHero), raw = d => d.raw ?? d.value;
        const span = d => (d.projected ? [d.lo, d.hi].map(v => v * raw(d) / d.value) : [raw(d)]);
        const values = amounts(at.map(span), format);
        const ratios = padded(at.map(d => (ours ? `${span(d).map(v => (v / raw(ours)).toFixed(1)).join('–')}×` : '–')));
        return {
          title: `${FORMAT[SPEC.x.format](points[0].xValue)} ${SPEC.x.label}`,
          rows: at.map((d, i) => ({
            label: d.projected ? `${d[SERIES]} (projected)` : d[SERIES],
            value: `${values[i]} · ${ratios[i]}`,
            color: ink(d[SERIES]), active: d === ours,
          })),
        };
      },
    };
  }

  /* One panel: a line per library over size, a whisker over the middle half at each point, and a
     shaded funnel over each projection's range. The highlighted library is drawn last and thick, so it stays on top where
     the lines cross, and is named right of its last point. The matrix floor is a thin neutral line
     without whiskers.
     Each measured point is a dot ringed in the surface colour, so
     crossing points stay apart. */
  /* The highlighted library's name, right of its last point. A control rather than a text mark:
     the marks share one clip, which would cut the name off in the right margin it sits in. */
  function heroLabel(ours) {
    const end = ours.toSorted((a, b) => a[X] - b[X]).at(-1);
    if (!end) return [];
    return [{ id: 'hero-label', resolve: ({ scales }) => ({ nodes: [{
      kind: 'label', key: 'hero-label', x: scales.x.map(end[X]) + 9, y: scales.y.map(end.value), text: HERO,
      anchor: 'start', baseline: 'middle', fontSize: 12, fontWeight: 650, style: { fill: ink(HERO) },
    }] }) }];
  }

  /* The relative view's y window per axis type, the same in every panel so panels compare at a
     glance. Wide enough for every library but pyomo in the published run. */
  const WINDOW = { linear: [0, 5], log: [0.1, 50] };
  const WINDOW_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50];

  function windowAxis() {
    const [lo, hi] = WINDOW[state.scale];
    if (state.scale === 'linear') {
      return { scale: scaleLinear().domain([lo, hi]), grid: true, axis: { ticks: { count: 5, format: FORMAT.ratio } } };
    }
    return { scale: scaleLog().domain([lo, hi]), grid: true, axis: { ticks: { values: WINDOW_TICKS, format: FORMAT.ratio } } };
  }

  /* Each library with measured points outside the window, named at the edge they cross: its ratio,
     or the range of its ratios, in its colour above its last such point. Along each edge, labels go
     left to right, each into the first row where it clears the one before, so sizes a linear axis
     draws close together do not overprint. */
  function outside(points, [lo, hi]) {
    const edges = [['↑', p => p.value > hi, hi, 1], ['↓', p => p.value < lo, lo, -1]];
    const span = ps => {
      const v = ps.map(p => p.value), a = FORMAT.ratio(Math.min(...v)), b = FORMAT.ratio(Math.max(...v));
      return a === b ? b : `${a.slice(0, -1)}–${b}`;
    };
    const marked = points.filter(p => !p.projected && (p.value > hi || p.value < lo));
    if (!marked.length) return [];
    return [{ id: 'outside', resolve: ({ scales }) => ({ nodes: edges.flatMap(([arrow, past, edge, inward]) => {
      const ends = [];
      return [...Map.groupBy(marked.filter(past), p => p[SERIES])]
        .map(([name, ps]) => ({ name, x: scales.x.map(Math.max(...ps.map(p => p[X]))), text: `${arrow}${span(ps)}` }))
        .sort((a, b) => a.x - b.x)
        .map(({ name, x, text }) => {
          const half = text.length * 3;
          let row = ends.findIndex(end => end < x - half);
          if (row < 0) row = ends.push(0) - 1;
          ends[row] = x + half + 4;
          return { kind: 'label', key: `outside-${name}-${arrow}`, x, y: scales.y.map(edge) + inward * (8 + row * 12),
            text, anchor: 'middle', baseline: 'middle', fontSize: 10, fontWeight: 650, style: { fill: ink(name) } };
        });
    }) }) }];
  }

  /* A projection is taken from the whole panel and then cut to the size range, so its slope does
     not move with the range. */
  function definition(rows, metric) {
    const fmt = FORMAT[metric.format];
    const measured = viewed(rows.filter(r => r.value != null && inRange(r)));
    const ahead = viewed(projections(rows).filter(inRange));
    const split = data => [data.filter(r => !isHero(r)), data.filter(isHero)];
    const [others, ours] = split(measured), [theirsAhead, oursAhead] = split(ahead);
    const theirs = others.filter(r => !isFloor(r)), floor = others.filter(isFloor);
    const enc = { x: X, z: SERIES, color: SERIES };
    const line = (data, strokeWidth) => lineY(data, { ...enc, y: 'value', strokeWidth });
    const points = (data, r) => dot(data, { ...enc, y: 'value', r, stroke: 'var(--surface)', strokeWidth: 2 });
    const funnel = (data, fillOpacity) => [areaY(data, { ...enc, y1: 'lo', y2: 'hi', fillOpacity }),
      lineY(data, { ...enc, y: 'lo', strokeWidth: 1 }), lineY(data, { ...enc, y: 'hi', strokeWidth: 1 })];
    const whiskers = (data, strokeWidth) => link(data.filter(r => r.hi > r.lo),
      { x1: X, x2: X, y1: 'lo', y2: 'hi', color: SERIES, strokeWidth, lineCap: 'round' });
    const sizes = [...new Set([...measured, ...ahead].map(r => r[X]))].sort((a, b) => a - b);
    const relative = state.view === 'relative';
    return defineChart({
      marks: [
        ...(metric.band ? [whiskers(theirs, 1.5), whiskers(ours, 2)] : []),
        ...funnel(theirsAhead, 0.25), ...funnel(oursAhead, 0.35),
        line(floor, 1.5), line(theirs, 2), line(ours, 3.5),
        points(floor, 2.5), points(theirs, 3.5), points(ours, 5),
        crosshair({ y: false }),
      ],
      scales: {
        x: xAxis(sizes, state.range.from === STOPS[0]),
        y: relative ? windowAxis() : axis(measured.flatMap(r => [r.lo, r.hi]), fmt),
      },
      color: { domain: LIBRARIES, range: SPEC.series.paint },
      clip: true, margin: { right: 76 },
      controls: [...heroLabel([...ours, ...oursAhead]), ...(relative ? outside(measured, WINDOW[state.scale]) : [])],
      focus: 'group-x', maxFocusDistance: Number.POSITIVE_INFINITY,
      tooltip: grouped(metric.format),
    });
  }

  function table(rows, metric) {
    const fmt = FORMAT[metric.format];
    const others = LIBRARIES.filter(n => n !== HERO && !FLOOR.members.includes(n) && rows.some(r => r[SERIES] === n));
    const floorShown = rows.some(isFloor);
    const body = [];
    for (const [title, panel] of Map.groupBy(rows, panelOf)) {
      const guess = new Map(projections(panel).filter(r => r.projected).map(r => [`${r[SERIES]}|${r[X]}`, r]));
      const cell = (r, kind = '') => {
        if (!r) return '<td class="none">—</td>';
        if (r.value == null) {
          if (!r[REFUSED]) return '<td class="none">—</td>';
          const g = guess.get(`${r[SERIES]}|${r[X]}`);
          const why = `refused by the ${r[REFUSED].slice(1)} budget`;
          return `<td class="none" title="${g ? `${why} · projected ${fmt(g.lo)} to ${fmt(g.hi)}` : why}">${r[REFUSED]}</td>`;
        }
        if (kind === 'own') return `<td class="own">${fmt(r.value)}</td>`;
        if (r.ratio == null) return `<td class="none">${fmt(r.value)}</td>`;
        const look = kind === 'neutral' ? 'class="neutral"' : `style="${shade(r.ratio)}"`;
        return `<td ${look} title="${r[SERIES]}: ${fmt(r.value)}">${FORMAT.ratio(r.ratio)}</td>`;
      };
      const sizes = [...new Set(panel.filter(r => isHero(r) && inRange(r)).map(r => r[X]))].sort((a, b) => a - b);
      sizes.forEach((size, i) => {
        const at = new Map(panel.filter(r => r[X] === size).map(r => [r[SERIES], r]));
        body.push(`<tr class="${i ? '' : 'first'}"><td class="panel">${i ? '' : title}</td>` +
          `<td>${FORMAT[SPEC.x.format](size)}</td>${cell(at.get(HERO), 'own')}${others.map(n => cell(at.get(n), INTERNAL.includes(n) ? 'neutral' : '')).join('')}` +
          `${floorShown ? cell(FLOOR.members.map(n => at.get(n)).find(Boolean), 'neutral') : ''}</tr>`);
      });
    }
    root.querySelector('#bench-heat').innerHTML =
      `<thead><tr><th>panel</th><th>${SPEC.x.label}</th><th>${HERO}</th>${others.map(n => `<th${INTERNAL.includes(n) ? ' class="neutral"' : ''}>${n}</th>`).join('')}` +
      `${floorShown ? `<th class="neutral">${FLOOR.label}</th>` : ''}</tr></thead>` +
      `<tbody>${body.join('')}</tbody>`;
  }

  root.querySelector('#bench-scale').innerHTML = `<span>other library faster</span>` +
    BINS.map(([, b]) => `<i style="background:var(--d-${b})"></i>`).join('') +
    `<span>${HERO} faster</span><span>4× · 2× · 1.25× · 1.1× · even · 1.1× · 1.25× · 2× · 4×</span>`;

  function draw() {
    hosts.forEach(h => h.destroy());
    hosts = [];
    const metric = SPEC.metrics[state.metric];
    const rows = rowsNow();
    const grid = root.querySelector('#bench-facets');
    grid.replaceChildren();
    for (const [title, panel] of Map.groupBy(rows, panelOf)) {
      if (!panel.some(r => r.value != null && inRange(r))) continue;
      const fig = document.createElement('div');
      fig.className = 'bench-panel';
      fig.innerHTML = `<div class="bench-caption">${title}</div><div></div>`;
      grid.appendChild(fig);
      hosts.push(mountChart(fig.lastElementChild, { definition: definition(panel, metric), height: 240,
        ariaLabel: `${title}: ${metric.label}${state.view === 'absolute' ? '' : ` relative to ${HERO}`} ` +
          `against model size, one line per ${SERIES}` }));
    }
    table(rows, metric);
    const present = new Set(ROWS.filter(chosen).map(r => r[SERIES]));
    const floorHidden = FLOOR.members.every(n => state.hidden.has(n));
    root.querySelector('#bench-legend').innerHTML = LIBRARIES.filter(n => present.has(n) && !FLOOR.members.includes(n)).map(n =>
      `<button data-series="${n}" aria-pressed="${!state.hidden.has(n)}"${n === HERO ? ` disabled title="Every other line and cell is read against ${HERO}"` : ''}>` +
      `<i class="swatch${n === HERO ? ' ours' : ''}" style="background:${SPEC.series.paint[LIBRARIES.indexOf(n)]}"></i>${n}</button>`).join('') +
      `<button data-floor aria-pressed="${!floorHidden}"><i class="swatch" style="background:var(--floor)"></i>${FLOOR.label}</button>`;
  }

  /* The size range as a slider on an axis of its own: zero, then a tick per decade, the chosen span
     shaded, a handle at each end, and a drag along the axis that brushes a new span; a click moves
     the nearer end. Each end snaps to a stop and stays one short of the other. */
  const slider = root.querySelector('.bench-range');
  const LAST = STOPS.length - 1;
  const at = i => `${(i / LAST) * 100}%`;
  slider.innerHTML = '<div class="bench-track"><div class="bench-band"></div>' +
    STOPS.map((d, i) => `<span class="bench-tick" style="left:${at(i)}">${FORMAT.count(10 ** d)}</span>`).join('') +
    '</div>' + ['from', 'to'].map(end => `<input type="range" data-end="${end}" min="0" max="${LAST}" step="1" ` +
      `aria-label="${end === 'from' ? 'Smallest' : 'Largest'} model shown">`).join('');
  const track = slider.querySelector('.bench-track');
  const handle = end => slider.querySelector(`input[data-end="${end}"]`);
  const span = () => [STOPS.indexOf(state.range.from), STOPS.indexOf(state.range.to)];

  function showRange() {
    const [lo, hi] = span();
    [['from', lo], ['to', hi]].forEach(([end, i]) => {
      handle(end).value = i;
      handle(end).setAttribute('aria-valuetext', `${FORMAT.count(10 ** STOPS[i])} ${SPEC.x.label}`);
    });
    Object.assign(slider.querySelector('.bench-band').style, { left: at(lo), width: `${((hi - lo) / LAST) * 100}%` });
    slider.querySelectorAll('.bench-tick').forEach((t, i) => t.classList.toggle('in', i >= lo && i <= hi));
  }

  function setRange(lo, hi) {
    const changed = lo !== span()[0] || hi !== span()[1];
    state.range = { from: STOPS[lo], to: STOPS[hi] };
    showRange();
    if (changed) draw();
  }

  slider.addEventListener('input', ev => {
    const [lo, hi] = span(), v = +ev.target.value;
    if (ev.target.dataset.end === 'from') setRange(Math.min(v, hi - 1), hi);
    else setRange(lo, Math.max(v, lo + 1));
  });

  const nearest = ev => {
    const r = track.getBoundingClientRect();
    return Math.round(Math.min(Math.max((ev.clientX - r.left) / r.width, 0), 1) * LAST);
  };
  let brush = null;
  slider.addEventListener('pointerdown', ev => {
    if (ev.target.matches('input')) return;
    brush = { start: nearest(ev), moved: false };
    slider.setPointerCapture(ev.pointerId);
  });
  slider.addEventListener('pointermove', ev => {
    if (!brush) return;
    const i = nearest(ev);
    if (i === brush.start) return;
    brush.moved = true;
    setRange(Math.min(i, brush.start), Math.max(i, brush.start));
  });
  slider.addEventListener('pointerup', () => {
    if (brush && !brush.moved) {
      const [lo, hi] = span(), i = brush.start;
      if (Math.abs(i - lo) <= Math.abs(i - hi)) setRange(Math.min(i, hi - 1), hi);
      else setRange(lo, Math.max(i, lo + 1));
    }
    brush = null;
  });

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
    if (['metric', 'view', 'scale'].includes(key)) state[key] = b.dataset.value;
    else state.filter[key] = b.dataset.value;
    b.parentElement.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
    draw();
  });

  slider.hidden = LAST < 1;
  if (LAST >= 1) showRange();
  draw();
}
