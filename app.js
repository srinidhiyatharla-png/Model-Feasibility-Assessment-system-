(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };

  var MODELS = { transformer: 'Transformer / LLM', cnn: 'CNN Image Classifier', resnet: 'ResNet', mlp: 'Tabular MLP', rnn: 'RNN / LSTM' };
  var OPT_NAMES = { sgd: 'SGD', adam: 'Adam', adamw: 'AdamW', rmsprop: 'RMSprop' };
  var MODE_LABEL = { hard: 'Hard Failure Risk', silent: 'Silent Failure Risk', ok: 'Likely Success' };
  var HEAD = {
    ok: ['Likely to train successfully', 'The configuration sits inside the range that usually converges.'],
    hard: ['Training is likely to crash or diverge', 'Expect NaN loss, or a loss that climbs from the first epochs.'],
    silent: ['Training will run but is unlikely to learn', 'No error is raised, so the run can look healthy while accuracy stays near baseline.']
  };
  var IDS = ['lr', 'batch', 'epochs', 'ds', 'drop', 'wd'];
  var history = [];

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]; }); }
  function fmtNum(x) {
    if (x === 0) return '0';
    var a = Math.abs(x);
    if (a >= 0.001 && a < 100000) return String(+x.toPrecision(3));
    return x.toExponential(0).replace('e+', 'e');
  }
  function state(s) { return s >= 0.75 ? 'ok' : s >= 0.4 ? 'silent' : 'hard'; }

  // ---- API ----
  function api(method, url, body) {
    return fetch(url, {
      method: method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) {
      return r.json().then(function (data) { return { ok: r.ok, status: r.status, data: data }; });
    });
  }

  // ---- Result ----
  function renderResult(res) {
    var pct = Math.round(res.p * 100);
    var h = HEAD[res.mode];
    var html = '';
    html += '<div class="verdict"><div><div class="pct-label">Probability of success</div><div class="pct">' + pct + '<small>%</small></div></div>';
    html += '<div class="vtext"><span class="badge ' + res.mode + '">' + MODE_LABEL[res.mode] + '</span><h3>' + h[0] + '</h3><p>' + h[1] + '</p></div></div>';
    html += '<div class="meter" role="img" aria-label="Probability ' + pct + ' percent"><i class="' + res.mode + '" style="width:' + Math.max(pct, 1) + '%"></i></div>';
    html += '<div><p class="sect">What drives this result</p><div class="factors">';
    res.factors.forEach(function (f) {
      html += '<div class="factor"><b>' + esc(f.name) + '</b><div class="meter" aria-hidden="true"><i class="' + state(f.s) + '" style="width:' + Math.round(f.s * 100) + '%"></i></div><span class="msg">' + esc(f.msg) + '</span></div>';
    });
    html += '</div></div>';
    if (res.mode !== 'ok') {
      var s = res.sug;
      html += '<div><p class="sect">Suggested starting point</p><div class="suggest">' +
        '<span class="chip">' + OPT_NAMES[s.opt] + '</span><span class="chip">lr ' + fmtNum(s.lr) + '</span><span class="chip">batch ' + s.batch + '</span><span class="chip">' + s.epochs + ' epochs</span>' +
        '<button type="button" class="apply" id="applyBtn">Apply to form</button></div></div>';
    } else {
      html += '<p class="note">No changes needed. Validate with a short pilot run before committing the full budget.</p>';
    }
    var card = $('resultCard');
    card.innerHTML = html;
    card.hidden = false;
    var ab = $('applyBtn');
    if (ab) ab.addEventListener('click', function () {
      $('opt').value = res.sug.opt; $('lr').value = fmtNum(res.sug.lr);
      $('batch').value = res.sug.batch; $('epochs').value = res.sug.epochs;
      $('predict').focus();
    });
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    try { card.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' }); } catch (e) {}
  }

  // ---- History ----
  function renderHistory() {
    var body = $('histBody');
    body.innerHTML = '';
    $('histEmpty').hidden = history.length > 0;
    history.forEach(function (h, i) {
      var tr = document.createElement('tr');
      tr.tabIndex = 0;
      tr.setAttribute('data-i', i);
      tr.innerHTML = '<td class="t">' + esc(h.t) + '</td><td>' + esc(MODELS[h.model] || h.model) + '</td><td>' + esc(fmtNum(h.lr)) +
        '</td><td>' + esc(h.batch) + '</td><td>' + esc(h.epochs) + '</td><td>' + esc(OPT_NAMES[h.opt] || h.opt) + '</td><td class="p">' + esc(h.p) + '%</td><td><span class="badge ' + h.mode + '">' + MODE_LABEL[h.mode] + '</span></td>';
      body.appendChild(tr);
    });
  }
  function fetchHistory() {
    return api('GET', '/api/history').then(function (r) {
      if (r.ok) { history = r.data.history; renderHistory(); }
    }).catch(function () { $('e-net').textContent = 'Cannot reach the server. Start it with npm start.'; });
  }
  function loadRow(i) {
    var h = history[i]; if (!h) return;
    $('model').value = h.model; $('opt').value = h.opt;
    $('lr').value = fmtNum(h.lr); $('batch').value = h.batch; $('epochs').value = h.epochs;
    $('ds').value = h.ds != null ? h.ds : ''; $('drop').value = h.drop != null ? h.drop : ''; $('wd').value = h.wd != null ? h.wd : '';
    showView(false);
  }
  $('histBody').addEventListener('click', function (e) { var tr = e.target.closest('tr'); if (tr) loadRow(+tr.getAttribute('data-i')); });
  $('histBody').addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { var tr = e.target.closest('tr'); if (tr) { e.preventDefault(); loadRow(+tr.getAttribute('data-i')); } }
  });
  $('clearBtn').addEventListener('click', function () {
    api('DELETE', '/api/history').then(function (r) { if (r.ok) { history = r.data.history; renderHistory(); } });
  });

  var showingHistory = false;
  function showView(h) {
    showingHistory = h;
    $('formView').hidden = h; $('histView').hidden = !h;
    $('histBtn').setAttribute('aria-pressed', String(h));
    $('histBtnText').textContent = h ? 'New assessment' : 'History';
    if (h) fetchHistory();
  }
  $('histBtn').addEventListener('click', function () { showView(!showingHistory); });

  // ---- Form ----
  function setErrors(errors) {
    IDS.forEach(function (id) {
      var m = errors[id] || '';
      $('e-' + id).textContent = m;
      if (m) $(id).setAttribute('aria-invalid', 'true'); else $(id).removeAttribute('aria-invalid');
    });
  }
  IDS.forEach(function (id) {
    $(id).addEventListener('input', function () { $(id).removeAttribute('aria-invalid'); $('e-' + id).textContent = ''; });
  });

  $('cfg').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var btn = $('predict');
    $('e-net').textContent = '';
    setErrors({});
    btn.disabled = true; btn.textContent = 'Analyzing…';
    var payload = { model: $('model').value, opt: $('opt').value };
    IDS.forEach(function (id) { payload[id] = $(id).value; });

    api('POST', '/api/predict', payload).then(function (r) {
      if (r.ok) { renderResult(r.data); return; }
      if (r.data && r.data.errors) {
        setErrors(r.data.errors);
        var first = IDS.filter(function (id) { return r.data.errors[id]; })[0];
        if (first) $(first).focus();
      } else {
        $('e-net').textContent = (r.data && r.data.error) || 'Something went wrong. Try again.';
      }
    }).catch(function () {
      $('e-net').textContent = 'Cannot reach the server. Start it with npm start, then reload this page.';
    }).then(function () {
      btn.disabled = false; btn.textContent = 'Predict';
    });
  });
})();
