let CURRENT = null;
let chartTopic = null;

async function fetchJSON(url, opts={}){
  const r = await fetch(url, Object.assign({headers:{"Content-Type":"application/json"}}, opts));
  return await r.json();
}

function getCheckedBiasTypes(){
  const ids = ["bt1","bt2","bt3","bt4","bt5"];
  const map = {"bt1":"wording","bt2":"selection","bt3":"framing","bt4":"evidence","bt5":"other"};
  let out = [];
  ids.forEach(id=>{ const el = document.getElementById(id); if (el && el.checked){ out.push(map[id]); }});
  return out;
}

async function loadTopics(){
  const data = await fetchJSON("/api/topics");
  const sel = document.getElementById("topic-select");
  sel.innerHTML = '<option value="">所有主题</option>';
  data.topics.forEach(t=>{
    const opt = document.createElement("option");
    opt.value = t.topic;
    opt.textContent = `${t.topic} (${t.count})`;
    sel.appendChild(opt);
  });
}

async function loadSample(){
  const topic = document.getElementById("topic-select").value;
  const data = await fetchJSON(`/api/sample${topic?`?topic=${encodeURIComponent(topic)}`:""}`);
  if(!data.ok){ alert("没有样本"); return; }
  CURRENT = data.article;
  document.getElementById("article-id").textContent = "#" + CURRENT.id;
  document.getElementById("topic-badge").textContent = CURRENT.topic || "(Unlabeled)";
  document.getElementById("meta-badge").textContent = CURRENT.time_place || "";
  document.getElementById("title").textContent = CURRENT.title || "（无标题）";
  document.getElementById("text").textContent = CURRENT.text || "";
  document.getElementById("bias_yes0").checked = true;
  document.getElementById("bias_side").value = "";
  document.getElementById("bias_strength").value = 0;
  ["bt1","bt2","bt3","bt4","bt5"].forEach(id=>{ const el=document.getElementById(id); if(el){ el.checked=false; }});
  document.getElementById("reasons").value="";
  document.getElementById("ai-tip").textContent = "";
}

async function submitVote(e){
  e.preventDefault();
  if(!CURRENT){ return; }
  const bias_yes = document.getElementById("bias_yes1").checked ? 1 : 0;
  const payload = {
    article_id: CURRENT.id,
    bias_yes: bias_yes,
    bias_side: document.getElementById("bias_side").value,
    bias_strength: parseInt(document.getElementById("bias_strength").value || "0"),
    bias_types: getCheckedBiasTypes(),
    reasons: document.getElementById("reasons").value.trim()
  };
  const res = await fetchJSON("/api/submit", {method:"POST", body: JSON.stringify(payload)});
  if(res.ok){
    await refreshStats();
    await loadSample();
  }else{
    alert("提交失败：" + (res.error||""));
  }
}

async function aiJudge(){
  const txt = document.getElementById("text").textContent || "";
  const res = await fetchJSON("/api/judge", {method:"POST", body: JSON.stringify({text: txt})});
  if(res.ok){
    const r = res.result;
    document.getElementById("ai-tip").textContent = `AI参考：${r.bias_yes? "有偏见":"无偏见"} (score=${r.score})`;
  }
}

async function refreshStats(){
  const data = await fetchJSON("/api/stats");
  const g = data.global;
  document.getElementById("global-stats").textContent = `投票数：${g.votes}，偏见判定：${g.bias_yes}（比例 ${(g.bias_rate*100).toFixed(1)}%）`;

  const labels = data.by_topic.map(x=>x.topic);
  const rates = data.by_topic.map(x=> (x.bias_rate*100).toFixed(1));
  const ctx = document.getElementById("chart-topic");
  if(chartTopic){ chartTopic.destroy(); }
  chartTopic = new Chart(ctx, {
    type: "bar",
    data: { labels: labels, datasets: [{ label: "偏见比例(%)", data: rates }] },
    options: { animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: 100 } } }
  });

  const ul = document.getElementById("bias-types");
  ul.innerHTML = "";
  const entries = Object.entries(data.bias_types).sort((a,b)=>b[1]-a[1]);
  entries.forEach(([k,v])=>{
    const li = document.createElement("li");
    li.textContent = `${k}: ${v}`;
    ul.appendChild(li);
  });
}
  // 偏见强度滑条（0-2）
  const strengthInput = document.getElementById("bias_strength");
  const strengthValue = document.getElementById("bias_strength_value");

  if (strengthInput && strengthValue) {
    // 初始化显示
    strengthValue.textContent = strengthInput.value;

    strengthInput.addEventListener("input", () => {
      strengthValue.textContent = strengthInput.value;
    });
  }

document.addEventListener("DOMContentLoaded", async ()=>{
  await loadTopics();
  await loadSample();
  await refreshStats();
  document.getElementById("btn-refresh").addEventListener("click", loadSample);
  document.getElementById("judge-form").addEventListener("submit", submitVote);
  document.getElementById("btn-ai-judge").addEventListener("click", aiJudge);
  document.getElementById("topic-select").addEventListener("change", loadSample);
  document.addEventListener("keydown", (e)=>{ if(e.key.toLowerCase()==="r"){ loadSample(); }});
});
