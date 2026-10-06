'use strict';
// Zero-dependency server: JSON API + static hosting of ../frontend.
const http = require('http');
const fs = require('fs');
const path = require('path');
const engine = require('./engine');

const PORT = process.env.PORT || 3000;
const FRONTEND = path.join(__dirname, '..', 'frontend');
const DATA_DIR = path.join(__dirname, 'data');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');
const MAX_HISTORY = 30;

const SEEDS = [
  { t: '05 Oct, 21:32', model: 'mlp', lr: 4, batch: 18, epochs: 2, opt: 'adam', p: 0, mode: 'hard' },
  { t: '05 Oct, 21:30', model: 'resnet', lr: 0.1, batch: 8, epochs: 2, opt: 'adam', p: 0, mode: 'hard' },
  { t: '05 Oct, 21:27', model: 'transformer', lr: 1e-5, batch: 32, epochs: 5, opt: 'adam', p: 0, mode: 'silent' },
  { t: '05 Oct, 21:20', model: 'cnn', lr: 0.001, batch: 32, epochs: 30, opt: 'adam', p: 89, mode: 'ok' }
];

function loadHistory() {
  try {
    const a = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
    if (Array.isArray(a)) return a;
  } catch (e) { /* first run or unreadable file */ }
  return SEEDS.slice();
}
function saveHistory(list) {
  try {
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(list, null, 2));
  } catch (e) { console.error('Could not save history:', e.message); }
}
let history = loadHistory();

function stamp() {
  const d = new Date();
  try {
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }) + ', ' +
      d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  } catch (e) { return d.toISOString(); }
}

function send(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': Buffer.byteLength(body) });
  res.end(body);
}

function readBody(req) {
  return new Promise(function (resolve, reject) {
    let data = '';
    req.on('data', function (chunk) {
      data += chunk;
      if (data.length > 100000) { reject(new Error('Body too large')); req.destroy(); }
    });
    req.on('end', function () {
      try { resolve(data ? JSON.parse(data) : {}); } catch (e) { reject(new Error('Invalid JSON')); }
    });
    req.on('error', reject);
  });
}

async function handleApi(req, res, pathname) {
  if (pathname === '/api/health' && req.method === 'GET') return send(res, 200, { ok: true });

  if (pathname === '/api/history' && req.method === 'GET') return send(res, 200, { history });
  if (pathname === '/api/history' && req.method === 'DELETE') {
    history = [];
    saveHistory(history);
    return send(res, 200, { history });
  }

  if (pathname === '/api/predict' && req.method === 'POST') {
    let body;
    try { body = await readBody(req); } catch (e) { return send(res, 400, { error: e.message }); }
    const c = engine.normalize(body);
    const errors = engine.validate(c);
    if (Object.keys(errors).length) return send(res, 400, { errors });
    const result = engine.assess(c);
    history.unshift({
      t: stamp(), model: c.model, lr: c.lr, batch: c.batch, epochs: c.epochs, opt: c.opt,
      p: Math.round(result.p * 100), mode: result.mode,
      ds: c.raw.ds === '' ? null : c.ds, drop: c.raw.drop === '' ? null : c.drop, wd: c.raw.wd === '' ? null : c.wd
    });
    history = history.slice(0, MAX_HISTORY);
    saveHistory(history);
    return send(res, 200, result);
  }

  return send(res, 404, { error: 'Not found' });
}

const TYPES = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon' };

function serveStatic(req, res, pathname) {
  let rel = decodeURIComponent(pathname);
  if (rel === '/') rel = '/index.html';
  const file = path.normalize(path.join(FRONTEND, rel));
  if (file !== FRONTEND && !file.startsWith(FRONTEND + path.sep)) { res.writeHead(403); return res.end('Forbidden'); }
  fs.readFile(file, function (err, buf) {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream' });
    res.end(buf);
  });
}

http.createServer(function (req, res) {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname.startsWith('/api/')) {
    handleApi(req, res, pathname).catch(function (e) { console.error(e); send(res, 500, { error: 'Server error' }); });
  } else if (req.method === 'GET') {
    serveStatic(req, res, pathname);
  } else {
    res.writeHead(405); res.end();
  }
}).listen(PORT, function () {
  console.log('Model Feasibility Assessment running at http://localhost:' + PORT);
});
