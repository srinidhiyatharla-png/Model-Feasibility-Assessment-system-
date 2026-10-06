'use strict';
// Rule-based feasibility engine. No ML model is involved: probabilities come from
// typical hyperparameter ranges for each model family.

const PROFILES = {
  transformer: { label: 'Transformer / LLM', lr: 3e-4, batch: 64, epochs: 10, drop: 0.1, minData: 100000, cap: 0.98,
    opt: { adamw: 1, adam: 0.92, rmsprop: 0.35, sgd: 0.25 } },
  cnn: { label: 'CNN Image Classifier', lr: 1e-3, batch: 64, epochs: 30, drop: 0.25, minData: 20000, cap: 0.98,
    opt: { adam: 1, adamw: 1, sgd: 0.88, rmsprop: 0.75 } },
  resnet: { label: 'ResNet', lr: 1e-3, batch: 64, epochs: 30, drop: 0.1, minData: 30000, cap: 0.98,
    opt: { adamw: 1, adam: 0.95, sgd: 0.95, rmsprop: 0.6 } },
  mlp: { label: 'Tabular MLP', lr: 1e-3, batch: 64, epochs: 50, drop: 0.2, minData: 5000, cap: 0.98,
    opt: { adam: 1, adamw: 1, sgd: 0.8, rmsprop: 0.85 } },
  rnn: { label: 'RNN / LSTM', lr: 1e-3, batch: 64, epochs: 25, drop: 0.25, minData: 20000, cap: 0.98,
    opt: { adam: 1, adamw: 0.95, rmsprop: 0.95, sgd: 0.4 } }
};
const OPT_NAMES = { sgd: 'SGD', adam: 'Adam', adamw: 'AdamW', rmsprop: 'RMSprop' };
// Typical stable learning rate relative to Adam
const OPT_LR_SCALE = { sgd: 30, adam: 1, adamw: 1, rmsprop: 0.5 };

function fmtNum(x) {
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 0.001 && a < 100000) return String(+x.toPrecision(3));
  return x.toExponential(0).replace('e+', 'e');
}

function parseNum(v) {
  if (v === null || v === undefined) return NaN;
  const s = String(v).trim();
  return s === '' ? NaN : Number(s);
}
function isBlank(v) { return v === null || v === undefined || String(v).trim() === ''; }

// Turns a raw request body into a typed config. `raw.x` is '1' when the optional field was provided.
function normalize(b) {
  b = b || {};
  return {
    model: String(b.model || ''), opt: String(b.opt || ''),
    lr: parseNum(b.lr), batch: parseNum(b.batch), epochs: parseNum(b.epochs),
    ds: parseNum(b.ds), drop: parseNum(b.drop), wd: parseNum(b.wd),
    raw: { ds: isBlank(b.ds) ? '' : '1', drop: isBlank(b.drop) ? '' : '1', wd: isBlank(b.wd) ? '' : '1' }
  };
}

function validate(c) {
  const e = {};
  if (!PROFILES[c.model]) e.model = 'Choose a supported model type.';
  if (!OPT_NAMES[c.opt]) e.opt = 'Choose a supported optimizer.';
  if (isNaN(c.lr)) e.lr = 'Enter a number, for example 0.001 or 1e-4.';
  else if (c.lr <= 0) e.lr = 'Learning rate must be greater than 0.';
  if (isNaN(c.batch) || c.batch < 1 || Math.floor(c.batch) !== c.batch) e.batch = 'Enter a whole number of 1 or more.';
  if (isNaN(c.epochs) || c.epochs < 1 || Math.floor(c.epochs) !== c.epochs) e.epochs = 'Enter a whole number of 1 or more.';
  if (c.raw.ds !== '' && (isNaN(c.ds) || c.ds < 1 || Math.floor(c.ds) !== c.ds)) e.ds = 'Enter a whole number of 1 or more, or leave blank.';
  if (c.raw.drop !== '' && (isNaN(c.drop) || c.drop < 0 || c.drop >= 1)) e.drop = 'Dropout must be at least 0 and below 1.';
  if (c.raw.wd !== '' && (isNaN(c.wd) || c.wd < 0)) e.wd = 'Weight decay must be 0 or more.';
  if (!e.batch && !e.ds && c.raw.ds !== '' && c.batch > c.ds) e.batch = 'Batch size cannot be larger than the dataset.';
  return e;
}

function assess(c) {
  const pr = PROFILES[c.model];
  let hard = false;
  const factors = [];

  // Learning rate
  const lrEff = pr.lr * OPT_LR_SCALE[c.opt];
  const d = Math.log10(c.lr / lrEff);
  const er = c.epochs / pr.epochs;
  const u = d + (er < 1 ? 0.5 * Math.log10(er) : 0);
  const lrS = d > 0 ? Math.exp(-d * d / 0.72) : Math.exp(-u * u / 0.5);
  let lrMsg;
  if (d > 1.2) {
    hard = true;
    lrMsg = fmtNum(c.lr) + ' is roughly ' + Math.round(Math.pow(10, d)) + '× above a stable value for ' + OPT_NAMES[c.opt] + ' on this model. Loss is likely to blow up to NaN.';
  } else if (d > 0.5) {
    lrMsg = 'High for ' + OPT_NAMES[c.opt] + '. Training may oscillate or spike early.';
  } else if (u < -0.8) {
    lrMsg = 'Very low for this setup. Weights barely move, so loss looks calm while the model learns almost nothing.';
  } else if (u < -0.4) {
    lrMsg = 'On the low side. Convergence will be slow within this epoch budget.';
  } else {
    lrMsg = 'Inside the usual range for ' + OPT_NAMES[c.opt] + ' on ' + pr.label + '.';
  }
  factors.push({ name: 'Learning rate', s: lrS, msg: lrMsg });

  // Optimizer
  const oS = pr.opt[c.opt];
  factors.push({ name: 'Optimizer', s: oS, msg: oS >= 0.8
    ? OPT_NAMES[c.opt] + ' is a solid choice for ' + pr.label + '.'
    : OPT_NAMES[c.opt] + ' is a weak fit for ' + pr.label + '. Adaptive optimizers converge more reliably here.' });

  // Epochs
  const eS = er < 1 ? Math.pow(er, 0.5) : (er > 4 ? 0.85 : 1);
  factors.push({ name: 'Epochs', s: eS, msg: er < 1
    ? c.epochs + ' epochs is short. Models like this usually need about ' + pr.epochs + '.'
    : er > 4 ? 'Far longer than usual. Watch for overfitting and use early stopping.' : 'Enough passes over the data to converge.' });

  // Batch size
  const db = Math.log10(c.batch / pr.batch);
  const bS = Math.exp(-db * db / 2.88);
  factors.push({ name: 'Batch size', s: bS, msg: bS >= 0.75
    ? 'Close to the typical ' + pr.batch + '.'
    : (db < 0 ? 'Small batches give noisy gradients and slow epochs.' : 'Very large batches tend to generalize worse unless the learning rate is scaled.') });

  // Regularization
  let dS = 0.98, dMsg = 'Not provided. Assuming a sensible default.';
  if (c.raw.drop !== '') {
    if (c.drop > 0.6) { dS = Math.max(0.05, 1 - (c.drop - 0.5) * 2.5); dMsg = 'Dropout this high removes most of the signal. The model will underfit.'; }
    else { dS = 0.85 + 0.15 * Math.exp(-Math.pow(c.drop - pr.drop, 2) / (2 * 0.0625)); dMsg = 'Reasonable regularization.'; }
  }
  let wS = 0.98, wMsg = '';
  if (c.raw.wd !== '') {
    if (c.wd > 0.1) { wS = 0.3; wMsg = 'Weight decay above 0.1 shrinks weights faster than they can learn.'; }
    else { wS = 1; wMsg = 'Weight decay is in a safe range.'; }
  }
  factors.push({ name: 'Regularization', s: dS * wS, msg: (c.raw.wd !== '' && c.wd > 0.1) ? wMsg : dMsg });

  // Dataset size
  let sS = 0.98, sMsg = 'Not provided. Assuming enough data.';
  if (c.raw.ds !== '') {
    const r = c.ds / pr.minData;
    sS = r >= 1 ? 1 : 0.55 + 0.45 * Math.sqrt(r);
    sMsg = r >= 1 ? 'Large enough for this model type.'
      : (r < 0.3 ? 'Small for ' + pr.label + '. Expect overfitting or reliance on pretrained weights.'
        : 'Slightly below the usual ' + pr.minData.toLocaleString('en-US') + ' examples.');
  }
  factors.push({ name: 'Dataset size', s: sS, msg: sMsg });

  let p = pr.cap;
  factors.forEach(function (f) { p *= f.s; });
  if (hard) p *= 0.03;
  p = Math.max(0, Math.min(0.99, p));
  const mode = hard ? 'hard' : (p >= 0.5 ? 'ok' : 'silent');

  let bestOpt = c.opt;
  if (pr.opt[c.opt] < 0.8) {
    bestOpt = Object.keys(pr.opt).reduce(function (a, b) { return pr.opt[a] >= pr.opt[b] ? a : b; });
  }
  const sug = { opt: bestOpt, lr: pr.lr * OPT_LR_SCALE[bestOpt], batch: pr.batch, epochs: Math.max(c.epochs, pr.epochs) };
  return { p, mode, factors, sug, label: pr.label };
}

module.exports = { PROFILES, OPT_NAMES, normalize, validate, assess };
