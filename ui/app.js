async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

async function loadSessions() {
  try {
    const data = await fetchJSON('/api/sessions');
    const ul = document.getElementById('sessions');
    ul.innerHTML = '';
    data.sessions.forEach((sid) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#';
      a.textContent = sid;
      a.onclick = (e) => {
        e.preventDefault();
        loadSession(sid);
      };
      li.appendChild(a);
      ul.appendChild(li);
    });
  } catch (e) {
    console.error('Failed to load sessions', e);
  }
}

async function loadPapers() {
  try {
    const data = await fetchJSON('/api/papers');
    const ul = document.getElementById('papers');
    ul.innerHTML = '';
    data.papers.slice(0, 50).forEach((p) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = p.url || '#';
      a.target = '_blank';
      a.textContent = p.title || p.id;
      li.appendChild(a);
      ul.appendChild(li);
    });
  } catch (e) {
    console.error('Failed to load papers', e);
  }
}

function el(tag, cls, text) {
  const x = document.createElement(tag);
  if (cls) x.className = cls;
  if (text) x.textContent = text;
  return x;
}

async function loadSession(sid) {
  const data = await fetchJSON(`/api/sessions/${sid}`);
  document.getElementById('welcome').classList.add('hidden');
  document.getElementById('session').classList.remove('hidden');

  document.getElementById('topic').textContent = `${data.topic} (session: ${data.session_id})`;
  const warningsBox = document.getElementById('warnings');
  if (Array.isArray(data.warnings) && data.warnings.length > 0) {
    warningsBox.classList.remove('hidden');
    const list = data.warnings.map((w) => `<li>${w}</li>`).join('');
    warningsBox.innerHTML = `<strong>Warnings</strong><ul>${list}</ul>`;
  } else {
    warningsBox.classList.add('hidden');
    warningsBox.innerHTML = '';
  }
  const q = document.getElementById('questions');
  q.innerHTML = '';
  (data.questions || []).forEach((t) => q.appendChild(el('li', null, t)));

  const f = document.getElementById('findings');
  f.innerHTML = '';
  (data.findings || []).forEach((fi) => {
    const box = el('div', 'card');
    box.appendChild(el('div', 'q', fi.question));
    box.appendChild(el('div', 'a', fi.answer));
    const cites = el('div', 'cites');
    (fi.citations || []).forEach((c) => {
      const span = el('span', 'cite', `${c.paper_id}@${c.page_from ?? '?'} `);
      cites.appendChild(span);
    });
    box.appendChild(cites);
    f.appendChild(box);
  });

  const ideas = document.getElementById('ideas');
  ideas.innerHTML = '';
  (data.ideas || []).forEach((id) => {
    const box = el('div', 'card');
    box.appendChild(el('div', 'idea-title', id.title));
    box.appendChild(el('div', 'idea-line', `Motivation: ${id.motivation}`));
    box.appendChild(el('div', 'idea-line', `Method: ${id.method}`));
    box.appendChild(el('div', 'idea-line', `Eval: ${id.eval}`));
    box.appendChild(el('div', 'idea-line', `Risks: ${id.risks}`));
    ideas.appendChild(box);
  });

  const rl = document.getElementById('reading');
  rl.innerHTML = '';
  (data.reading_list || []).forEach((r) => rl.appendChild(el('li', null, `${r.paper_id} — ${r.reason}`)));
}

loadSessions();
loadPapers();
