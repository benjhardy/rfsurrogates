/* rfscene.js: a small, reusable 3D view of a layered RF device (three.js r128 + OrbitControls, loaded first).
 *
 *   <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
 *   <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
 *   <script src="rfscene.js"></script>
 *
 *   if (RFScene.supported()) {
 *     const v = RFScene.create(stageElement, { colors: {...}, toolbar: element, buttonClass: 'btn' });
 *     v.setDevice(device);          // first call frames the camera; later calls morph to the new geometry
 *     v.highlight('g1');            // light up every shape, marker and label tagged 'g1' (null clears)
 *     v.setSignal({ s11: 0.3, s21: 0.9 });   // port pulses: sizes follow |S11| and |S21| (linear); null hides them
 *   }
 *
 * Device description, in the device's own units (z up, the view rotates it):
 *   {
 *     layers: [{ id, kind: 'metal' | 'ground' | 'dielectric' | 'outline' | 'marker', z0, z1,
 *                shapes: [{ rect: [x0, y0, x1, y1] } | { poly: [[x, y], ...] } |
 *                         { cyl: { c: [x, y, z], r, h, axis: 'x' | 'y' | 'z' } }   // each may carry z0, z1, tags: []
 *                        ] }],                            // layer order = stacking order for the exploded view
 *     ports:  [{ name, at: [x, y, z], size: [width, height], normal: 'x' | 'y' }],   // port face, centred at `at`
 *     labels: [{ text, at: [x, y, z], tags: [], stack }], // labels with tags only show while highlighted
 *     signal: { in: [[x, y, z], ...], out: [[x, y, z], ...], radius, stack },   // polylines: port 1 -> device -> port 2
 *     explodeGap,                                         // layer separation of the exploded view
 *     frame: [layer ids], fit,                            // what the camera frames (default: all but markers, outlines)
 *   }
 * A layer's place in the exploded stack is its index, or `stack` when given (so markers can move with the copper);
 * labels and the signal pulses move with the stack level they name. Outline layers are not used to frame the view.
 *
 * Shapes are matched by layer id and index: when the count and type stay the same, rect and cyl shapes glide to
 * the new geometry; otherwise the layer is rebuilt. Rendering runs only while something moves.
 */
(function (root) {
  'use strict';
  const COLORS = {
    bg: '#FFFFFF', ink: '#142230', muted: '#586678', copper: '#C27C3E', ground: '#8C5A30', dielectric: '#B9C9B0',
    outline: '#8898AA', marker: '#2160A8', port: '#2160A8', incident: '#2160A8', reflected: '#D9472B',
    transmitted: '#2A7F50', warn: '#C0392B',
  };

  function supported() {
    try {
      const c = document.createElement('canvas');
      return !!(root.THREE && root.THREE.OrbitControls && (c.getContext('webgl') || c.getContext('experimental-webgl')));
    } catch (e) { return false; }
  }

  function create(stage, opts) {
    opts = opts || {};
    const T = root.THREE;
    const col = Object.assign({}, COLORS, opts.colors || {});
    const reduceMQ = matchMedia('(prefers-reduced-motion: reduce)');
    const coarse = matchMedia('(pointer: coarse)').matches;

    const renderer = new T.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, coarse ? 1.5 : 2));
    renderer.outputEncoding = T.sRGBEncoding;
    const cv = renderer.domElement;
    cv.setAttribute('role', 'img');
    cv.setAttribute('aria-label', opts.label || '3D view of the device');
    cv.tabIndex = 0;
    stage.prepend(cv);

    const scene = new T.Scene();
    const camera = new T.PerspectiveCamera(34, 16 / 9, 0.01, 2000);
    const controls = new T.OrbitControls(camera, cv);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.enableZoom = coarse;              // on desktop the wheel scrolls the page; use the keys or buttons
    scene.add(new T.HemisphereLight(0xffffff, 0x8899aa, 0.85));
    const sun = new T.DirectionalLight(0xffffff, 0.75); sun.position.set(-3, 8, 5); scene.add(sun);
    const world = new T.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);   // device z -> screen up

    const v = {
      playing: !reduceMQ.matches, autoRotate: false, explode: 0, explodeTgt: 0, t: 0,
      visible: true, running: false, framed: false, dev: null, tag: null, signal: null, tints: {},
    };
    const layers = new Map();          // id -> { group, kind, z0, z1, shapes: [...] }
    let ports = [], labels = [], pulses = null;

    // ------------------------------------------------------------------ materials
    const mats = {};
    function makeMats() {
      const c = n => new T.Color(col[n]).convertSRGBToLinear();      // r128: material colours are linear
      mats.metal = new T.MeshStandardMaterial({ color: c('copper'), metalness: 0.12, roughness: 0.42 });
      mats.ground = new T.MeshStandardMaterial({ color: c('ground'), metalness: 0.12, roughness: 0.6 });
      mats.dielectric = new T.MeshStandardMaterial({ color: c('dielectric'), roughness: 0.85, transparent: true, opacity: 0.55, depthWrite: false });
      mats.marker = new T.MeshBasicMaterial({ color: c('marker'), transparent: true, opacity: 0.45, depthWrite: false });
      mats.edge = new T.LineBasicMaterial({ color: c('ink'), transparent: true, opacity: 0.22 });
      mats.outline = new T.LineBasicMaterial({ color: c('outline'), transparent: true, opacity: 0.55 });
      mats.port = new T.MeshBasicMaterial({ color: c('port'), transparent: true, opacity: 0.16, side: T.DoubleSide, depthWrite: false });
      mats.portEdge = new T.LineBasicMaterial({ color: c('port'), transparent: true, opacity: 0.8 });
      for (const k of ['incident', 'reflected', 'transmitted']) mats[k] = new T.MeshBasicMaterial({ color: c(k), transparent: true });
    }
    makeMats();
    const unitBox = new T.BoxGeometry(1, 1, 1), unitEdges = new T.EdgesGeometry(unitBox);
    const unitCyl = new T.CylinderGeometry(1, 1, 1, 32), unitCylEdges = new T.EdgesGeometry(unitCyl, 30);
    const unitSphere = new T.SphereGeometry(1, 16, 12);

    function materialFor(kind, layerId) {
      const tint = v.tints[layerId];
      if (tint && (kind === 'metal' || kind === 'ground')) {
        const m = mats[kind].clone(); m.color = new T.Color(tint).convertSRGBToLinear(); m.userData.tint = true; return m;
      }
      return mats[kind] || mats.metal;
    }

    // ------------------------------------------------------------------ shapes
    function shapeTarget(s, L) {
      const z0 = s.z0 != null ? s.z0 : L.z0, z1 = s.z1 != null ? s.z1 : L.z1;
      if (s.rect) {
        const [x0, y0, x1, y1] = s.rect;
        return { type: 'rect', p: [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2],
                 s: [Math.max(1e-4, Math.abs(x1 - x0)), Math.max(1e-4, Math.abs(y1 - y0)), Math.max(1e-4, z1 - z0)] };
      }
      if (s.cyl) {
        const { c, r, h } = s.cyl;
        return { type: 'cyl', axis: s.cyl.axis || 'z', p: c.slice(), s: [r, h, r] };
      }
      return { type: 'poly', pts: s.poly, z0, z1 };
    }

    function buildShape(s, L, layerId) {
      const tg = shapeTarget(s, L);
      let obj;
      if (L.kind === 'outline') {
        obj = new T.LineSegments(unitEdges, mats.outline);
      } else if (tg.type === 'rect' || tg.type === 'cyl') {
        obj = new T.Mesh(tg.type === 'rect' ? unitBox : unitCyl, materialFor(L.kind, layerId));
        if (L.kind !== 'marker') obj.add(new T.LineSegments(tg.type === 'rect' ? unitEdges : unitCylEdges, mats.edge));
        if (tg.type === 'cyl') {
          if (tg.axis === 'z') obj.rotation.x = Math.PI / 2;
          else if (tg.axis === 'x') obj.rotation.z = Math.PI / 2;
        }
      } else {
        const sh = new T.Shape(tg.pts.map(q => new T.Vector2(q[0], q[1])));
        const g = new T.ExtrudeGeometry(sh, { depth: Math.max(1e-4, tg.z1 - tg.z0), bevelEnabled: false });
        obj = new T.Mesh(g, materialFor(L.kind, layerId));
        obj.position.z = tg.z0;
        if (L.kind !== 'marker') obj.add(new T.LineSegments(new T.EdgesGeometry(g), mats.edge));
      }
      if (tg.type !== 'poly') { obj.position.set(...tg.p); obj.scale.set(...tg.s); }
      const item = { obj, tg, tags: s.tags || [], kind: L.kind };
      if (L.kind === 'marker') obj.visible = false;
      return item;
    }

    function disposeItem(it) {
      if (it.tg.type === 'poly') { it.obj.geometry.dispose(); it.obj.children.forEach(c => c.geometry.dispose()); }
      if (it.obj.material && it.obj.material.userData && it.obj.material.userData.tint) it.obj.material.dispose();
    }

    function setLayer(L, i) {
      let rec = layers.get(L.id);
      const same = rec && rec.kind === L.kind && rec.items.length === L.shapes.length &&
        rec.items.every((it, k) => it.tg.type !== 'poly' && it.tg.type === shapeTarget(L.shapes[k], L).type);
      if (same) {
        rec.items.forEach((it, k) => { it.tg = shapeTarget(L.shapes[k], L); it.tags = L.shapes[k].tags || []; });
      } else {
        if (rec) { rec.items.forEach(disposeItem); world.remove(rec.group); }
        rec = { group: new T.Group(), kind: L.kind, items: [] };
        rec.items = L.shapes.map(s => buildShape(s, L, L.id));
        rec.items.forEach(it => rec.group.add(it.obj));
        world.add(rec.group);
        layers.set(L.id, rec);
      }
      rec.index = L.stack != null ? L.stack : i;
      return rec;
    }

    // ------------------------------------------------------------------ labels, ports, pulses
    function textSprite(text, color) {
      const pad = 10, fs = 44, c = document.createElement('canvas'), g = c.getContext('2d');
      g.font = `600 ${fs}px ${opts.font || 'system-ui, sans-serif'}`;
      const w = Math.ceil(g.measureText(text).width) + 2 * pad, h = fs + 2 * pad;
      c.width = w; c.height = h;
      g.font = `600 ${fs}px ${opts.font || 'system-ui, sans-serif'}`;
      g.fillStyle = col.bg; g.globalAlpha = 0.82; g.fillRect(0, 0, w, h); g.globalAlpha = 1;
      g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, pad, h / 2 + 2);
      const tex = new T.CanvasTexture(c); tex.encoding = T.sRGBEncoding;
      const sp = new T.Sprite(new T.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
      sp.userData.aspect = w / h; sp.renderOrder = 10;
      return sp;
    }

    function setLabels(list, size) {
      labels.forEach(l => { world.remove(l.sp); l.sp.material.map.dispose(); l.sp.material.dispose(); });
      labels = (list || []).map(L => {
        const sp = textSprite(L.text, L.tags && L.tags.length ? col.marker : col.ink);
        sp.position.set(...L.at);
        sp.scale.set(size * sp.userData.aspect, size, 1);
        sp.visible = !(L.tags && L.tags.length);
        world.add(sp);
        return { sp, tags: L.tags || [], z: L.at[2], stack: L.stack || 0 };
      });
    }

    function setPorts(list, size) {
      ports.forEach(p => { world.remove(p.g); p.plane.geometry.dispose(); });
      ports = (list || []).map(P => {
        const g = new T.Group(), [w, h] = P.size;
        const plane = new T.Mesh(new T.PlaneGeometry(w, h), mats.port);
        const edge = new T.LineSegments(new T.EdgesGeometry(plane.geometry), mats.portEdge);
        plane.add(edge);
        if (P.normal === 'y') plane.rotation.x = Math.PI / 2; else plane.rotation.set(Math.PI / 2, Math.PI / 2, 0);
        g.add(plane); g.position.set(...P.at);
        const sp = textSprite(P.name, col.port); sp.scale.set(size * sp.userData.aspect, size, 1);
        sp.position.set(0, 0, h / 2 + size); g.add(sp);
        world.add(g);
        return { g, plane };
      });
    }

    function polyline(pts) {
      const seg = [], cum = [0];
      for (let i = 1; i < pts.length; i++) {
        const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
        seg.push(d); cum.push(cum[i - 1] + d);
      }
      const len = cum[cum.length - 1] || 1;
      return u => {
        const s = Math.max(0, Math.min(1, u)) * len;
        let i = 1; while (i < cum.length - 1 && cum[i] < s) i++;
        const a = pts[i - 1], b = pts[i], f = seg[i - 1] ? (s - cum[i - 1]) / seg[i - 1] : 0;
        return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
      };
    }

    const N_PULSE = 5;
    function setPulses(sig) {
      if (pulses) pulses.meshes.forEach(m => world.remove(m));
      pulses = null;
      if (!sig) return;
      const mk = kind => Array.from({ length: N_PULSE }, () => { const m = new T.Mesh(unitSphere, mats[kind]); m.visible = false; world.add(m); return m; });
      pulses = { inc: mk('incident'), ref: mk('reflected'), tra: mk('transmitted'), fin: polyline(sig.in), fout: polyline(sig.out), r: sig.radius || 0.3, stack: sig.stack || 0 };
      pulses.meshes = [...pulses.inc, ...pulses.ref, ...pulses.tra];
    }

    function drawPulses(t) {
      if (!pulses) return;
      const on = !!v.signal;
      const phase = t * 0.45;
      const place = (arr, f, amp, reverse) => arr.forEach((m, i) => {
        const u = (phase + i / N_PULSE) % 1;
        m.visible = on && amp > 0.02;
        if (!m.visible) return;
        m.position.set(...f(reverse ? 1 - u : u));
        m.position.z += v.explode * ((v.dev && v.dev.explodeGap) || 1) * pulses.stack;
        const r = pulses.r * (0.35 + 0.65 * amp) * Math.sin(Math.PI * u) ** 0.35;
        m.scale.setScalar(Math.max(1e-4, r));
      });
      place(pulses.inc, pulses.fin, 1, false);
      place(pulses.ref, pulses.fin, on ? v.signal.s11 : 0, true);
      place(pulses.tra, pulses.fout, on ? v.signal.s21 : 0, false);
      for (const k of ['reflected', 'transmitted']) mats[k].opacity = 0.95;
    }

    // ------------------------------------------------------------------ motion
    function approach(cur, tgt, a) { let moved = false; for (let i = 0; i < 3; i++) { const d = tgt[i] - cur[i]; if (Math.abs(d) > 1e-5) { cur[i] += d * a; moved = true; } else cur[i] = tgt[i]; } return moved; }
    function step(dt) {
      const a = reduceMQ.matches ? 1 : 1 - Math.exp(-dt * 14);
      let moving = false;
      const ex = v.explodeTgt - v.explode;
      if (Math.abs(ex) > 1e-4) { v.explode += ex * a; moving = true; } else v.explode = v.explodeTgt;
      const gap = (v.dev && v.dev.explodeGap) || 1;
      layers.forEach(rec => {
        rec.group.position.z = v.explode * gap * rec.index;
        rec.items.forEach(it => {
          if (it.tg.type === 'poly') return;
          const p = it.obj.position.toArray(), s = it.obj.scale.toArray();
          const m1 = approach(p, it.tg.p, a), m2 = approach(s, it.tg.s, a);
          if (m1 || m2) { it.obj.position.fromArray(p); it.obj.scale.fromArray(s); moving = true; }
        });
      });
      labels.forEach(l => { l.sp.position.z = l.z + v.explode * gap * l.stack; });
      return moving;
    }

    let last = 0;
    function tick(now) {
      const dt = Math.min(0.05, (now - last) / 1000 || 0); last = now;
      controls.autoRotate = v.autoRotate && v.playing;
      const orbiting = controls.update();
      const anim = v.playing && v.visible && !document.hidden && (!!v.signal || v.autoRotate);
      if (anim) v.t += dt;
      const morphing = step(dt);
      drawPulses(v.t);
      renderer.render(scene, camera);
      if (anim || orbiting || morphing) requestAnimationFrame(tick); else v.running = false;
    }
    v.request = () => { if (!v.running) { v.running = true; last = performance.now(); requestAnimationFrame(tick); } };
    controls.addEventListener('change', v.request);
    controls.addEventListener('start', () => { v.userMoved = true; });   // until then a resize re-frames the view

    // ------------------------------------------------------------------ camera
    v.frame = () => {
      const box = new T.Box3();
      const ids = v.dev && v.dev.frame;
      layers.forEach((rec, id) => rec.items.forEach(it => { if (ids ? ids.includes(id) : it.kind !== 'marker' && it.kind !== 'outline') box.expandByObject(it.obj); }));
      if (box.isEmpty()) return;
      const sph = box.getBoundingSphere(new T.Sphere());
      const vHalf = T.MathUtils.degToRad(camera.fov / 2), hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
      const d = ((v.dev && v.dev.fit) || opts.fit || 1.05) * sph.radius / Math.sin(Math.min(vHalf, hHalf));
      const az = T.MathUtils.degToRad(opts.azimuth == null ? -30 : opts.azimuth), el = T.MathUtils.degToRad(opts.elevation == null ? 38 : opts.elevation);
      controls.target.copy(sph.center);
      camera.position.set(sph.center.x + d * Math.cos(el) * Math.sin(az), sph.center.y + d * Math.sin(el), sph.center.z + d * Math.cos(el) * Math.cos(az));
      camera.near = d / 200; camera.far = d * 20; camera.updateProjectionMatrix();
      controls.minDistance = d * 0.15; controls.maxDistance = d * 4; controls.update(); v.request();
    };
    v.orbit = (dTheta, dPhi) => {
      const off = camera.position.clone().sub(controls.target), s = new T.Spherical().setFromVector3(off);
      s.theta += dTheta; s.phi = Math.max(0.05, Math.min(Math.PI - 0.05, s.phi + dPhi));
      camera.position.copy(controls.target).add(new T.Vector3().setFromSpherical(s)); controls.update(); v.request();
    };
    v.zoom = f => { const off = camera.position.clone().sub(controls.target).multiplyScalar(f); camera.position.copy(controls.target).add(off); controls.update(); v.request(); };
    v.top = () => { const d = camera.position.distanceTo(controls.target); camera.position.copy(controls.target).add(new T.Vector3(0, d, 0.001 * d)); controls.update(); v.request(); };

    cv.addEventListener('keydown', e => {
      const k = e.key;
      if (k === 'ArrowLeft') v.orbit(-0.2, 0); else if (k === 'ArrowRight') v.orbit(0.2, 0);
      else if (k === 'ArrowUp') v.orbit(0, -0.15); else if (k === 'ArrowDown') v.orbit(0, 0.15);
      else if (k === '+' || k === '=') v.zoom(0.85); else if (k === '-' || k === '_') v.zoom(1.18);
      else if (k === '0') v.frame(); else if (k === 'e') v.setExplode(v.explodeTgt ? 0 : 1);
      else if (k === ' ') v.setPlaying(!v.playing); else return;
      e.preventDefault();
    });
    if (coarse) {                       // touch: the page scrolls until the user asks to turn the model
      controls.enabled = false; cv.style.touchAction = 'pan-y';
    }

    // ------------------------------------------------------------------ public API
    v.setDevice = dev => {
      v.dev = dev;
      const ids = new Set(dev.layers.map(L => L.id));
      layers.forEach((rec, id) => { if (!ids.has(id)) { rec.items.forEach(disposeItem); world.remove(rec.group); layers.delete(id); } });
      dev.layers.forEach((L, i) => setLayer(L, i));
      const size = dev.labelSize || 1;
      const portsKey = JSON.stringify(dev.ports), labelsKey = JSON.stringify(dev.labels), sigKey = JSON.stringify(dev.signal);
      if (portsKey !== v._ports) { setPorts(dev.ports, size); v._ports = portsKey; }
      if (labelsKey !== v._labels) { setLabels(dev.labels, size); v._labels = labelsKey; }
      if (sigKey !== v._sig) { setPulses(dev.signal); v._sig = sigKey; }
      applyHighlight();
      if (!v.framed) { step(1); v.frame(); v.framed = true; }
      v.request();
    };
    function applyHighlight() {
      const t = v.tag, on = it => !!(t && it.tags.includes(t));
      layers.forEach(rec => rec.items.forEach(it => {
        if (it.kind === 'marker') it.obj.visible = on(it);
        else if (it.obj.material && it.obj.material.emissive) {
          if (on(it)) { if (!it.obj.material.userData.hl) { it.obj.material = it.obj.material.clone(); it.obj.material.userData.hl = true; } it.obj.material.emissive = new T.Color(col.marker).convertSRGBToLinear(); it.obj.material.emissiveIntensity = 0.45; }
          else if (it.obj.material.userData.hl) { it.obj.material.dispose(); it.obj.material = materialFor(it.kind, [...layers].find(([, r]) => r === rec)[0]); }
        }
      }));
      labels.forEach(l => { if (l.tags.length) l.sp.visible = !!(t && l.tags.includes(t)); });
      v.request();
    }
    v.highlight = tag => { if (tag !== v.tag) { v.tag = tag || null; applyHighlight(); } };
    v.tint = (layerId, color) => {
      if ((v.tints[layerId] || null) === (color || null)) return;
      if (color) v.tints[layerId] = color; else delete v.tints[layerId];
      const rec = layers.get(layerId);
      if (rec) rec.items.forEach(it => { if (it.obj.material && it.obj.material.userData) { if (it.obj.material.userData.tint || it.obj.material.userData.hl) it.obj.material.dispose(); it.obj.material = materialFor(it.kind, layerId); } });
      applyHighlight();
    };
    v.setSignal = sig => { v.signal = sig ? { s11: Math.min(1, Math.max(0, sig.s11)), s21: Math.min(1, Math.max(0, sig.s21)) } : null; if (!v.signal) drawPulses(v.t); v.request(); };
    v.setExplode = x => { v.explodeTgt = x; v.request(); opts.onState && opts.onState(v); };
    v.setPlaying = on => { v.playing = on; v.request(); opts.onState && opts.onState(v); };
    v.setAutoRotate = on => { v.autoRotate = on; v.request(); opts.onState && opts.onState(v); };
    v.setColors = c => {
      Object.assign(col, c || {});
      Object.values(mats).forEach(m => m.dispose()); makeMats();
      renderer.setClearColor(new T.Color(col.bg));
      layers.forEach((rec, id) => rec.items.forEach(it => {
        if (it.kind === 'outline') it.obj.material = mats.outline;
        else { it.obj.material = materialFor(it.kind, id); it.obj.children.forEach(ch => { ch.material = mats.edge; }); }
      }));
      v._ports = v._labels = v._sig = null;
      if (v.dev) v.setDevice(v.dev);
      v.request();
    };
    v.dispose = () => { renderer.dispose(); cv.remove(); };

    // optional toolbar
    if (opts.toolbar) {
      const bc = opts.buttonClass || '';
      const mk = (txt, title, fn) => { const b = document.createElement('button'); b.type = 'button'; b.className = bc; b.textContent = txt; b.title = title; b.addEventListener('click', fn); opts.toolbar.append(b); return b; };
      const bReset = mk('Reset view', 'Frame the device (key 0)', () => { v.userMoved = false; v.frame(); });
      const bTop = mk('Top', 'Look straight down', () => v.top());
      const bEx = mk('Exploded', 'Separate the layers (key E)', () => v.setExplode(v.explodeTgt ? 0 : 1));
      const bRot = mk('Rotate', 'Turn the view slowly', () => v.setAutoRotate(!v.autoRotate));
      const bPlay = mk('Pause', 'Stop or start the motion (space)', () => v.setPlaying(!v.playing));
      let bTouch = null;
      if (coarse) bTouch = mk('Turn', 'Use touch to turn the model', () => { controls.enabled = !controls.enabled; cv.style.touchAction = controls.enabled ? 'none' : 'pan-y'; bTouch.textContent = controls.enabled ? 'Done' : 'Turn'; bTouch.setAttribute('aria-pressed', String(controls.enabled)); });
      void bReset; void bTop;
      const prev = opts.onState;
      opts.onState = s => {
        bEx.setAttribute('aria-pressed', String(!!s.explodeTgt));
        bRot.setAttribute('aria-pressed', String(s.autoRotate));
        bPlay.textContent = s.playing ? 'Pause' : 'Play'; bPlay.setAttribute('aria-pressed', String(!s.playing));
        prev && prev(s);
      };
      opts.onState(v);
    }

    renderer.setClearColor(new T.Color(col.bg));
    new ResizeObserver(() => {
      const w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
      renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
      if (v.framed && !v.userMoved) v.frame(); else v.request();
    }).observe(stage);
    new IntersectionObserver(ents => { v.visible = ents[0].isIntersecting; if (v.visible) v.request(); }, { threshold: 0.02 }).observe(stage);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) v.request(); });
    reduceMQ.addEventListener && reduceMQ.addEventListener('change', () => v.setPlaying(!reduceMQ.matches));
    cv.addEventListener('webglcontextlost', e => { e.preventDefault(); stage.insertAdjacentHTML('beforeend', '<div class="fallback">The 3D view stopped. Reload the page to start it again.</div>'); });
    v.three = { scene, camera, controls, renderer, world };
    return v;
  }

  root.RFScene = { supported, create, COLORS };
})(typeof window !== 'undefined' ? window : this);
