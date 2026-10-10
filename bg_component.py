"""Animated particle backdrop, rendered as a pinned full-screen component.

Streamlit strips <script> from st.markdown, so real animation has to live in a
components iframe. The iframe is pinned behind the page with CSS and made
click-through, so the app stays fully usable.
"""

import streamlit as st
import streamlit.components.v1 as components


def particle_backdrop(accent: str = "#9A9AA3", energy: float = 0.0,
                      secondary: str = "#4C8DFF") -> None:
    """
    accent    colour the field takes (risk band colour once a score exists)
    energy    0-1; raises drift speed, link density and brightness
    """
    markup = f"""
<canvas id="bg"></canvas>
<style>html,body{{margin:0;background:transparent;overflow:hidden}}
#bg{{display:block;width:100vw;height:100vh}}</style>
<script>
const ACCENT = "{accent}", SECOND = "{secondary}", ENERGY = {energy:.3f};

const cv = document.getElementById("bg"), ctx = cv.getContext("2d");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
let W = 0, H = 0, DPR = 1, P = [], t = 0;

function hex(h) {{
  h = h.replace("#", "");
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}}
const A = hex(ACCENT), S = hex(SECOND), GREY = [154,154,163];

function size() {{
  const w = cv.clientWidth || innerWidth, h = cv.clientHeight || innerHeight;
  if (w === W && h === H) return;
  W = w; H = h;
  DPR = Math.min(devicePixelRatio || 1, 2);
  cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  build();
}}

function build() {{
  const n = W < 760 ? 46 : 92;
  P = [];
  for (let i = 0; i < n; i++) {{
    P.push({{
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - 0.5) * 0.28, vy: (Math.random() - 0.5) * 0.28,
      r: 1.1 + Math.random() * 1.9, ph: Math.random() * 6.283,
      tone: Math.random()
    }});
  }}
}}

function frame() {{
  size();
  t += 1 / 60;
  const speed = 0.45 + ENERGY * 1.25;
  const link = (W < 760 ? 88 : 128) * (1 + ENERGY * 0.3);
  const glow = 0.30 + ENERGY * 0.45;

  ctx.clearRect(0, 0, W, H);

  for (const p of P) {{
    if (!reduce) {{
      // slow wandering current, so the field never looks static
      p.x += (p.vx + Math.sin(t * 0.22 + p.ph) * 0.16) * speed;
      p.y += (p.vy + Math.cos(t * 0.19 + p.ph) * 0.16) * speed;
      if (p.x < -40) p.x = W + 40; if (p.x > W + 40) p.x = -40;
      if (p.y < -40) p.y = H + 40; if (p.y > H + 40) p.y = -40;
    }}
  }}

  // links between near neighbours
  ctx.lineWidth = 0.75;
  for (let i = 0; i < P.length; i++) {{
    for (let j = i + 1; j < P.length; j++) {{
      const dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d2 = dx*dx + dy*dy;
      if (d2 < link * link) {{
        const a = (1 - Math.sqrt(d2) / link) * 0.26 * (0.55 + glow);
        const c = P[i].tone > 0.55 ? A : S;
        ctx.strokeStyle = `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{a.toFixed(3)}})`;
        ctx.beginPath(); ctx.moveTo(P[i].x, P[i].y); ctx.lineTo(P[j].x, P[j].y); ctx.stroke();
      }}
    }}
  }}

  // dots
  for (const p of P) {{
    const c = p.tone > 0.72 ? A : (p.tone > 0.4 ? S : GREY);
    const pulse = reduce ? 1 : 0.75 + 0.25 * Math.sin(t * 1.1 + p.ph);
    ctx.fillStyle = `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{(0.42 + glow * 0.5) * pulse}})`;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.2832); ctx.fill();
  }}

  requestAnimationFrame(frame);
}}

addEventListener("resize", size);
size();
requestAnimationFrame(frame);
</script>
"""
    # st.iframe is the current API; components.html is deprecated and due for
    # removal, so fall back only on older Streamlit versions.
    if hasattr(st, "iframe"):
        st.iframe(markup, height=1)
    else:
        components.html(markup, height=1)
