/* A small in-browser model of linearly precoded signal alignment.
   Two receive antennas, one interferer, QPSK, a random isometric precoder
   with e extra "null slots". Everything here runs live on the page. */
(function (root) {
  'use strict';

  // seeded RNG so the precoder is the same every time
  function mulberry(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  let rnd = Math.random;
  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  // CN(0,1): returns [re, im]
  function cn() { return [gauss() / Math.SQRT2, gauss() / Math.SQRT2]; }

  // Orthonormal n x l precoder Q (columns), stored as Q[col] = {re:[], im:[]}
  const cache = {};
  function precoder(l, e) {
    const key = l + ':' + e;
    if (cache[key]) return cache[key];
    const n = l + e;
    const saved = rnd;
    rnd = mulberry(1234 + 31 * l + e);
    const cols = [];
    for (let j = 0; j < l; j++) {
      let re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) { const c = cn(); re[i] = c[0]; im[i] = c[1]; }
      // modified Gram-Schmidt (twice for stability)
      for (let pass = 0; pass < 2; pass++) {
        for (let k = 0; k < j; k++) {
          const q = cols[k];
          let dr = 0, di = 0; // q^H v
          for (let i = 0; i < n; i++) {
            dr += q.re[i] * re[i] + q.im[i] * im[i];
            di += q.re[i] * im[i] - q.im[i] * re[i];
          }
          for (let i = 0; i < n; i++) {
            re[i] -= dr * q.re[i] - di * q.im[i];
            im[i] -= dr * q.im[i] + di * q.re[i];
          }
        }
      }
      let nr = 0;
      for (let i = 0; i < n; i++) nr += re[i] * re[i] + im[i] * im[i];
      nr = Math.sqrt(nr);
      for (let i = 0; i < n; i++) { re[i] /= nr; im[i] /= nr; }
      cols.push({ re, im });
    }
    rnd = saved;
    return (cache[key] = { n, l, e, cols });
  }

  // project away the precoder's column space: r = y - Q Q^H y
  function perp(P, yre, yim) {
    const { n, l, cols } = P;
    const re = Float64Array.from(yre), im = Float64Array.from(yim);
    for (let k = 0; k < l; k++) {
      const q = cols[k];
      let dr = 0, di = 0;
      for (let i = 0; i < n; i++) {
        dr += q.re[i] * yre[i] + q.im[i] * yim[i];
        di += q.re[i] * yim[i] - q.im[i] * yre[i];
      }
      for (let i = 0; i < n; i++) {
        re[i] -= dr * q.re[i] - di * q.im[i];
        im[i] -= dr * q.im[i] + di * q.re[i];
      }
    }
    return { re, im };
  }

  // 2x2 Hermitian smallest eigenpair of M_ij = r_i^H r_j
  function align2(r1, r2) {
    const n = r1.re.length;
    let A = 0, D = 0, Br = 0, Bi = 0;
    for (let i = 0; i < n; i++) {
      A += r1.re[i] ** 2 + r1.im[i] ** 2;
      D += r2.re[i] ** 2 + r2.im[i] ** 2;
      // r1^H r2
      Br += r1.re[i] * r2.re[i] + r1.im[i] * r2.im[i];
      Bi += r1.re[i] * r2.im[i] - r1.im[i] * r2.re[i];
    }
    const half = (A + D) / 2;
    const rad = Math.sqrt(((A - D) / 2) ** 2 + Br * Br + Bi * Bi);
    const lmin = half - rad, lmax = half + rad;
    let a;
    if (Br * Br + Bi * Bi < 1e-300) a = A <= D ? [[1, 0], [0, 0]] : [[0, 0], [1, 0]];
    else a = [[Br, Bi], [lmin - A, 0]];
    return { a, lmin, lmax };
  }

  const QPSK = [[1, 1], [-1, 1], [-1, -1], [1, -1]].map(p => [p[0] / Math.SQRT2, p[1] / Math.SQRT2]);

  // one block. opts: l, e, snr (dB), sir (dB, negative = strong interference)
  function block(o) {
    const P = precoder(o.l, o.e);
    const { n, l, cols } = P;
    // data: x[0] is a known pilot
    const xs = [];
    for (let k = 0; k < l; k++) xs.push(k === 0 ? 0 : (rnd() * 4) | 0);
    const x = xs.map((s, k) => (k === 0 ? [Math.SQRT1_2, Math.SQRT1_2] : QPSK[s]));
    // s = Q x
    const sre = new Float64Array(n), sim = new Float64Array(n);
    for (let k = 0; k < l; k++) {
      const q = cols[k];
      for (let i = 0; i < n; i++) {
        sre[i] += q.re[i] * x[k][0] - q.im[i] * x[k][1];
        sim[i] += q.re[i] * x[k][1] + q.im[i] * x[k][0];
      }
    }
    const ps = (l / n);                         // mean signal power per sample
    const pw = ps * Math.pow(10, -o.snr / 10);  // noise power
    const pp = ps * Math.pow(10, -o.sir / 10);  // interference power
    const pre = new Float64Array(n), pim = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const c = cn();
      pre[i] = c[0] * Math.sqrt(pp); pim[i] = c[1] * Math.sqrt(pp);
    }
    const y = [];
    for (let m = 0; m < 2; m++) {
      const h = cn(), g = cn();
      const re = new Float64Array(n), im = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const w = cn();
        re[i] = h[0] * sre[i] - h[1] * sim[i] + g[0] * pre[i] - g[1] * pim[i] + w[0] * Math.sqrt(pw);
        im[i] = h[0] * sim[i] + h[1] * sre[i] + g[0] * pim[i] + g[1] * pre[i] + w[1] * Math.sqrt(pw);
      }
      y.push({ re, im });
    }
    return { P, x, xs, y };
  }

  // x_hat = Q^H (a1 y1 + a2 y2), normalised by the pilot
  function recover(b, a) {
    const { P, y } = b;
    const { n, l, cols } = P;
    const zr = new Float64Array(n), zi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      zr[i] = a[0][0] * y[0].re[i] - a[0][1] * y[0].im[i] + a[1][0] * y[1].re[i] - a[1][1] * y[1].im[i];
      zi[i] = a[0][0] * y[0].im[i] + a[0][1] * y[0].re[i] + a[1][0] * y[1].im[i] + a[1][1] * y[1].re[i];
    }
    const out = [];
    for (let k = 0; k < l; k++) {
      const q = cols[k];
      let dr = 0, di = 0;
      for (let i = 0; i < n; i++) {
        dr += q.re[i] * zr[i] + q.im[i] * zi[i];
        di += q.re[i] * zi[i] - q.im[i] * zr[i];
      }
      out.push([dr, di]);
    }
    // divide by c = out[0] / pilot
    const pr = Math.SQRT1_2, pi = Math.SQRT1_2;
    const c = [(out[0][0] * pr + out[0][1] * pi) / (pr * pr + pi * pi), (out[0][1] * pr - out[0][0] * pi) / (pr * pr + pi * pi)];
    const cc = c[0] * c[0] + c[1] * c[1] || 1e-300;
    return out.map(v => [(v[0] * c[0] + v[1] * c[1]) / cc, (v[1] * c[0] - v[0] * c[1]) / cc]);
  }

  function decide(v) { return v[0] >= 0 ? (v[1] >= 0 ? 0 : 3) : (v[1] >= 0 ? 1 : 2); }

  // Two antennas, no precoding help: steer a null on whatever dominates the
  // antenna covariance (the minor eigenvector of the 2x2 sample covariance).
  function blind(b) {
    const y = b.y, n = b.P.n;
    let A = 0, D = 0, Br = 0, Bi = 0;
    for (let i = 0; i < n; i++) {
      A += y[0].re[i] ** 2 + y[0].im[i] ** 2;
      D += y[1].re[i] ** 2 + y[1].im[i] ** 2;
      Br += y[0].re[i] * y[1].re[i] + y[0].im[i] * y[1].im[i];
      Bi += y[0].re[i] * y[1].im[i] - y[0].im[i] * y[1].re[i];
    }
    const lmin = (A + D) / 2 - Math.sqrt(((A - D) / 2) ** 2 + Br * Br + Bi * Bi);
    return Br * Br + Bi * Bi < 1e-300 ? [[1, 0], [0, 0]] : [[Br, Bi], [lmin - A, 0]];
  }

  // returns the symbols and error rates of three receivers over many blocks:
  // aligned (precoded, two antennas), blind (two antennas, no precoding help),
  // and naive (one antenna)
  function run(o, blocks) {
    const A = [], N = [], B = [];
    let errA = 0, errN = 0, errB = 0, tot = 0;
    for (let t = 0; t < blocks; t++) {
      const b = block(o);
      const r1 = perp(b.P, b.y[0].re, b.y[0].im);
      const r2 = perp(b.P, b.y[1].re, b.y[1].im);
      const al = align2(r1, r2);
      const xa = recover(b, al.a);
      const xn = recover(b, [[1, 0], [0, 0]]);
      const xb = recover(b, blind(b));
      for (let k = 1; k < b.P.l; k++) {
        if (decide(xa[k]) !== b.xs[k]) errA++;
        if (decide(xn[k]) !== b.xs[k]) errN++;
        if (decide(xb[k]) !== b.xs[k]) errB++;
        tot++;
        A.push(xa[k]); N.push(xn[k]); B.push(xb[k]);
      }
    }
    return { aligned: A, naive: N, blind: B, serA: errA / tot, serN: errN / tot, serB: errB / tot };
  }

  // packet-timing score curve: stream of back-to-back packets under one
  // interferer; score(t) = lambda_min of the 2x2 matrix for the window at t
  function timing(o) {
    const P = precoder(o.l, o.e);
    const { n, l, cols } = P;
    const nPk = o.packets, L = nPk * n + n;
    const offset = o.offset;                      // true start inside the stream
    const stream = [new Float64Array(L + n), new Float64Array(L + n)];
    const streamI = [new Float64Array(L + n), new Float64Array(L + n)];
    const ps = l / n, pw = ps * Math.pow(10, -o.snr / 10), pp = ps * Math.pow(10, -o.sir / 10);
    const h = [cn(), cn()], g = [cn(), cn()];
    const sRe = new Float64Array(L + n), sIm = new Float64Array(L + n);
    for (let p = -1; p < nPk + 1; p++) {
      const sr = new Float64Array(n), si = new Float64Array(n);
      for (let k = 0; k < l; k++) {
        const xv = QPSK[(rnd() * 4) | 0], q = cols[k];
        for (let i = 0; i < n; i++) {
          sr[i] += q.re[i] * xv[0] - q.im[i] * xv[1];
          si[i] += q.re[i] * xv[1] + q.im[i] * xv[0];
        }
      }
      for (let i = 0; i < n; i++) {
        const idx = offset + p * n + i;
        if (idx >= 0 && idx < L + n) { sRe[idx] = sr[i]; sIm[idx] = si[i]; }
      }
    }
    for (let i = 0; i < L + n; i++) {
      const c = cn();
      const pr = c[0] * Math.sqrt(pp), pi = c[1] * Math.sqrt(pp);
      for (let m = 0; m < 2; m++) {
        const w = cn();
        stream[m][i] = h[m][0] * sRe[i] - h[m][1] * sIm[i] + g[m][0] * pr - g[m][1] * pi + w[0] * Math.sqrt(pw);
        streamI[m][i] = h[m][0] * sIm[i] + h[m][1] * sRe[i] + g[m][0] * pi + g[m][1] * pr + w[1] * Math.sqrt(pw);
      }
    }
    const score = [];
    for (let t = 0; t < o.span; t++) {
      const r = [0, 1].map(m => perp(P, stream[m].subarray(t, t + n), streamI[m].subarray(t, t + n)));
      const al = align2(r[0], r[1]);
      score.push(al.lmin);
    }
    return { score, n };
  }

  root.Sigal = { run, timing, precoder, setSeed(s) { rnd = mulberry(s); }, useRandom() { rnd = Math.random; } };
  if (typeof module !== 'undefined') module.exports = root.Sigal;
})(typeof window !== 'undefined' ? window : globalThis);
