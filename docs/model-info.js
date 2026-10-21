/* Contenuti statici della sezione "Fisica e modello": equazioni con definizione dei coefficienti,
   schede della loss, stadi del tracciatore nodo/coppia e ipotesi sull'asimmetria nodi/coppie.
   Le voci [id, simbolo, testo] puntano a GLOSSARY: il simbolo diventa un termine con popup. */
const MODEL_INFO = (() => {
const R = String.raw;

const EQS = [
  { tag: "(1)(2)", title: "Osservabili: ingresso e bersaglio",
    tex: R`N_t[i,a]=\langle\sigma^a_i\rangle_t,\qquad E_t[i,j,a,b]=\langle\sigma^a_i\sigma^b_j\rangle_t,\qquad o_t=(N_t,E_t)`,
    note: "Valori medi sullo stato nascosto; E è simmetrica e ha diagonale nulla.",
    where: [["Nt", "N_t[i,a]", "magnetizzazione del sito i lungo l'asse a (3n numeri)"], ["Et", "E_t[i,j,a,b]", "correlazione tra i siti i e j (9 canali per coppia)"],
      ["obs", "o_t", "insieme dei due; M numeri indipendenti"], ["M", "M", "3n + 9n(n−1)/2"]] },
  { tag: "(5)(6)", title: "Embedding",
    tex: R`h^{(0)}_{i,t}=\mathrm{MLP}_n\big[N_t[i,:],\log\Delta t\big]+\mathrm{PE}(t),\qquad e^{(0)}_{ij,t}=\mathrm{MLP}_e\big(E_t[i,j,:]\big),\ e_{ii}=0`,
    note: "Nessun indice di sito è codificato: il modello è equivariante per permutazioni dei siti e indipendente da n. Solo i token di sito ricevono la codifica di t.",
    where: [["hit", "h_{i,t}", "token di sito, dimensione d"], ["eit", "e_{ij,t}", "feature di coppia, dimensione d_e"], ["mlpn", "\\mathrm{MLP}_n", "da 4 numeri a d"],
      ["mlpe", "\\mathrm{MLP}_e", "da 9 numeri a d_e"], ["pe", "\\mathrm{PE}(t)", "codifica sinusoidale della posizione nella finestra"], ["dt", "\\Delta t", "passo temporale, entra come log Δt"]] },
  { tag: "(7)–(9)", title: "Attenzione spaziale (tra siti, a t fisso)",
    tex: R`q_i=W_q\mathrm{LN}(h_i),\ k_j=W_k\mathrm{LN}(h_j),\qquad b^\eta_{ij}=w_b^{\eta\top}\mathrm{LN}(e_{ij}),\qquad \alpha^\eta_{ij}=\mathrm{softmax}_j\!\Big(\tfrac{q^\eta_i\cdot k^\eta_j}{\sqrt{d_h}}+b^\eta_{ij}\Big)`,
    note: "La media di α su blocchi, teste e tempo è la seconda stima del grafo.",
    where: [["qkv", "q_i,k_j,v_j", "proiezioni del token di sito"], ["bij", "b_{ij}", "bias che la coppia aggiunge al logit"], ["attn", "\\alpha_{ij}", "quanto il sito i legge il sito j"],
      ["alpha", "\\bar\\alpha_{ij}", "α mediata e simmetrizzata: stima del grafo"], ["nh", "\\eta,\\ n_h", "indice della testa, numero di teste"], ["ln", "\\mathrm{LN}", "LayerNorm"]] },
  { tag: "(10)–(12)", title: "Valori aumentati e residuo",
    tex: R`u_i=\sum_j\alpha_{ij}\big(v_j+W_{ev}\mathrm{LN}(e_{ij})\big),\quad h_i\leftarrow h_i+W_o[u_i^1;\dots;u_i^{n_h}],\quad h_i\leftarrow h_i+\mathrm{MLP}_{ff}(\mathrm{LN}(h_i))`,
    note: "È il canale per cui l'informazione di coppia entra nel token del sito.",
    where: [["valaug", "u_i", "valori aumentati con le coppie"], ["mlpff", "\\mathrm{MLP}_{ff}", "feed-forward d → 4d → d"], ["hit", "h_i", "token di sito aggiornato"]] },
  { tag: "(13)", title: "Aggiornamento delle coppie",
    tex: R`e_{ij}\leftarrow e_{ij}+\mathrm{MLP}_{eu}\Big(\mathrm{LN}(e_{ij});\ (W_a\mathrm{LN}(h_i))\odot(W_b\mathrm{LN}(h_j))\Big),\qquad e_{ij}\leftarrow\tfrac12(e_{ij}+e_{ji})`,
    note: "La coppia (i,j) riceve informazione solo da sé stessa e dai due siti che connette; poi viene simmetrizzata.",
    where: [["mlpeu", "\\mathrm{MLP}_{eu},W_a,W_b", "MLP di aggiornamento e proiezioni dei due siti"], ["eit", "e_{ij}", "feature di coppia"], ["hit", "h_i,h_j", "i due siti connessi"]] },
  { tag: "(14)–(16)", title: "Strato temporale",
    tex: R`h_{i,t}\leftarrow h_{i,t}+\mathrm{CausalMHA}_t\big(\mathrm{LN}(h_{i,0:T})\big),\qquad e_{ij,t}\leftarrow e_{ij,t}+\mathrm{CausalDWConv}_t\big(\mathrm{LN}(e_{ij,0:T})\big)`,
    note: "I siti hanno attenzione causale su tutto il passato (RoPE); le coppie solo una convoluzione di 4 passi per canale. Dopo L blocchi il campo recettivo di una coppia è 3L+1 passi.",
    where: [["tmha", "\\mathrm{CausalMHA}", "attenzione causale sui propri passi passati"], ["conv", "\\mathrm{CausalDWConv}", "filtro causale depthwise, kernel 4"], ["light", "\\text{campo recettivo}", "quanti passi/siti influenzano un token"], ["Lbl", "L", "numero di blocchi"]] },
  { tag: "opz.", title: "Estensioni per le coppie (pair_attn, triangle, pair_latent)",
    tex: R`e_{ij}\leftarrow e_{ij}+\mathrm{Tri}\Big(\tfrac1n\sum_k a_{ik}\odot a_{jk}\Big),\qquad e_{ij,t}\leftarrow e_{ij,t}+\mathrm{CausalMHA}_t(e_{ij,0:T}),\qquad u_{ij}(k)=e^{k\Delta t\,C_{ij}}u_{ij}(0)`,
    note: "Tre modifiche opzionali (flag nel config, spente di default dopo il feedback del professore) che danno alle coppie ciò che i nodi avevano già: memoria temporale su tutta la finestra, contesto dei terzi siti e uno stato proprio da propagare.",
    where: [["pairattn", "\\mathrm{CausalMHA}_t", "attenzione temporale causale sulle coppie"], ["triangle", "\\sum_k a_{ik}\\odot a_{jk}", "aggiornamento triangolare sui terzi siti"], ["pairlat", "u_{ij},\\ C_{ij}", "latente di coppia e suo generatore antisimmetrico"]] },
  { tag: "(17)(18)", title: "Coefficienti globali e latenti per sito",
    tex: R`\theta_t=W_\theta\,\frac1n\sum_{i=1}^n h^{(L)}_{i,t}\in\mathbb R^K,\qquad z_{i,t}=W_z\,h^{(L)}_{i,t}\in\mathbb R^m,\quad Z_t=[z_1;\dots;z_n]`,
    note: "θ è una descrizione globale e indipendente da n della dinamica; z porta la memoria dello stato nascosto.",
    where: [["theta", "\\theta_t", "K coefficienti globali"], ["z", "z_{i,t}", "latente del sito i, dimensione m"], ["K", "K", "numero di basi"], ["m", "m", "dimensione del latente per sito"]] },
  { tag: "(19)(20)", title: "Generatore stabile",
    tex: R`A_{ii}=\sum_k\theta_{t,k}A_k,\ \ A_{ij}=g_{ij}\,\mathrm{reshape}(W_A\bar e_{ij}),\ \ g_{ij}=\sigma(w_g^\top\bar e_{ij}),\qquad G_t=(A-A^\top)-SS^\top`,
    note: "Con model.use_S = false il termine −SSᵀ è assente: G è antisimmetrica e la propagazione conserva esattamente la norma del latente. Le coppie entrano nella dinamica solo attraverso i blocchi A_ij.",
    where: [["Ak", "A_k,S_k", "basi apprese, combinate da θ"], ["WA", "A_{ij}", "blocco di accoppiamento tra i siti i e j"], ["gate", "g_{ij}", "gate: stima dell'arco"], ["ebar", "\\bar e_{ij}", "feature di coppia all'anchor"],
      ["G", "G_t", "generatore, nm × nm"], ["rot", "A-A^\\top,\\ SS^\\top", "rotazione e contrazione"], ["SF", "\\|S\\|_F", "gauge di dissipazione"]] },
  { tag: "opz.", title: "Generatore di Pauli (pauli_A)",
    tex: R`A_{ij}=\sum_{s}c_s(h_i,h_j,e_{ij})\,P_s,\quad P_s\in\{\sigma^{a_1}\!\otimes\cdots\otimes\sigma^{a_q}\},\ \mathrm{wt}\le w,\quad m=2^q,\qquad G=A-A^\top`,
    note: "Un solo MLP produce tutti i blocchi, diagonali e no, come somma di una base di Pauli fissa nel latente (non appresa). Nessun θ, nessun gate; il punteggio d'arco è ‖A_ij‖_F normalizzato. G resta antisimmetrica.",
    where: [["pauliA", "P_s,\\ c_s", "base di Pauli fissa e coefficienti predetti"], ["ebar", "e_{ij}", "feature di coppia (zero sulla diagonale)"], ["G", "G", "generatore, nm × nm"], ["gate", "\\|A_{ij}\\|_F", "punteggio d'arco"]] },
  { tag: "—", title: "Lettura a posteriori dal generatore",
    tex: R`\mathrm{AUROC}\big(\|G_{ij}\|_F,\mathrm{adj}\big),\qquad \mathrm{corr}\big(\|G_{ij}\|_F,J_{ij}\big)_\text{archi},\qquad \mathrm{corr}\big(\|G_{ii}\|_F,h_i\big)`,
    note: "Sostituisce il probe. h, J e adiacenza veri entrano solo qui, dopo il training, come misura: mai come bersaglio o gradiente. L'obiettivo resta prevedere bene gli stati: G e z possono essere lontani dalle descrizioni vere.",
    where: [["gen_auroc_adj", "\\mathrm{AUROC}", "separazione archi / non archi"], ["gen_corr_J", "\\mathrm{corr}_J", "intensità contro J"], ["gen_corr_h", "\\mathrm{corr}_h", "blocco diagonale contro h"]] },
  { tag: "(21)", title: "Propagazione nel latente",
    tex: R`P_t=\exp(\Delta t\,G_t),\qquad Z_{t+k}=P_t^{\,k}Z_t,\quad k=1,\dots,H`,
    note: "Calcolata con prodotti ripetuti matrice-vettore; con G antisimmetrica ‖P‖₂ = 1.",
    where: [["Pprop", "P_t", "propagatore a un passo"], ["k", "k", "passi nel futuro"], ["H", "H", "orizzonte"], ["spectrum", "\\mathrm{Re}\\,\\lambda(G)\\le0", "garanzia di stabilità"]] },
  { tag: "(22)(23)", title: "Decoder: dal latente alle osservabili",
    tex: R`\hat N_{t+k}[i,:]=\tanh\big(D_n z_{i,t+k}\big),\qquad \hat E_{t+k}[i,j,c]=\tanh\big(z_{i,t+k}^\top B_c\,z_{j,t+k}+\beta_c\big)`,
    note: "Nodi: lineari nel latente. Coppie: bilineari nei latenti dei due siti, simmetrizzati. La feature di coppia non entra direttamente nel decoder.",
    where: [["decN", "D_n", "mappa m → 3"], ["decE", "B_c,\\beta_c", "forma bilineare e bias del canale c (9 canali)"], ["tanh", "\\tanh", "tiene l'uscita in (−1, 1)"], ["z", "z_{i,t+k}", "latente propagato"]] },
  { tag: "(26)", title: "Probe (solo diagnostica, spento nelle run nuove)",
    tex: R`\hat h_i=w_h^\top h^{(L)}_{i,T},\qquad \hat J_{ij}=w_J^\top\bar e_{ij},\qquad \hat a_{ij}=\sigma(w_a^\top\bar e_{ij})`,
    note: "Lettori lineari con stop-gradient: non modificano il corpo, misurano cosa c'è già nelle feature. Nelle run nuove (model.probe = false, λ_probe = 0) non esistono: h e J veri non sono mai usati come bersaglio.",
    where: [["probe_h", "\\hat h_i", "stima del campo"], ["probe_J", "\\hat J_{ij}", "stima dell'accoppiamento"], ["probe_a", "\\hat a_{ij}", "stima dell'arco"], ["ebar", "\\bar e_{ij}", "feature di coppia"]] },
  { tag: "(28)", title: "Matrice densità ricostruita",
    tex: R`\hat\rho_{ij}=\tfrac14\sum_{a,b\in\{0,x,y,z\}}c_{ab}\,\sigma^a\otimes\sigma^b,\quad c_{00}=1,\ c_{a0}=\hat N[i,a],\ c_{0b}=\hat N[j,b],\ c_{ab}=\hat E[i,j,a,b]`,
    note: "Qualunque stato a due qubit è completamente determinato da queste 16 componenti: per questo è usata sia nella penalità fisica sia nella fidelity.",
    where: [["rho", "\\hat\\rho_{ij}", "matrice densità a due qubit"], ["sigma", "\\sigma^a", "operatori di Pauli"], ["lammin", "\\lambda_{\\min}", "autovalore minimo: < 0 = non fisica"]] },
  { tag: "—", title: "Fidelity con lo stato vero",
    tex: R`F_2(k)=\Big\langle\Big(\mathrm{Tr}\sqrt{\sqrt{\rho_{ij}}\,\hat\rho_{ij}\sqrt{\rho_{ij}}}\Big)^2\Big\rangle_{i<j}`,
    note: "ρ_ij è calcolata dalle osservabili vere (esatte), ρ̂_ij dalla previsione proiettata sul cono semidefinito positivo. Lo stato completo non è confrontabile perché il modello non lo prevede.",
    where: [["fid2", "F_2(k)", "fidelity ridotta a due qubit al passo k"], ["rho", "\\rho_{ij},\\hat\\rho_{ij}", "marginale vero e previsto"], ["fidmean", "\\bar F_2", "media su k = 1…16"]] },
  { tag: "(29)", title: "Equazioni di Heisenberg a un sito (baseline HL)",
    tex: R`\tfrac{d}{dt}\langle X_i\rangle=2\sum_jJ_{ij}\langle Y_iZ_j\rangle,\ \ \tfrac{d}{dt}\langle Y_i\rangle=-2\sum_jJ_{ij}\langle X_iZ_j\rangle+2h_i\langle Z_i\rangle,\ \ \tfrac{d}{dt}\langle Z_i\rangle=-2h_i\langle Y_i\rangle`,
    note: "Si chiudono sui dati di peso ≤ 2 e permettono di stimare J e h dal contesto, ma non di prevedere: serve lo stato nascosto.",
    where: [["heis", "\\tfrac{d}{dt}\\langle\\cdot\\rangle", "derivata temporale della media"], ["Jij", "J_{ij}", "accoppiamento ZZ"], ["hi", "h_i", "campo trasverso"], ["hl", "\\text{HL}", "baseline che usa queste equazioni"]] },
];

const LOSS_CARDS = [
  { id: "Lpred", grad: true,
    tex: R`\mathcal L_\text{pred}=\frac{1}{|\mathcal T_\text{anc}|H}\sum_{t\in\mathcal T_\text{anc}}\sum_{k=1}^H w_k\Big(\tfrac{1}{3n}\|\hat N_{t+k}-N_{t+k}\|^2+\tfrac{1}{9n(n-1)}\|\hat E_{t+k}-E_{t+k}\|^2_{i\ne j}\Big)`,
    steps: ["Per ogni anchor t il modello propaga H passi e produce N̂ (n×3) ed Ê (n×n×9).",
      "Si confrontano con le osservabili vere dei passi t+1…t+H.",
      "Nodi: errore quadratico medio sulle 3n entrate. Coppie: errore quadratico sulle sole entrate fuori diagonale, diviso 9n(n−1).",
      "I due termini si sommano (stessa scala, valori in [−1, 1]), si moltiplicano per w_k (uniformi) e si mediano su k e anchor."],
    code: "losses.pred_loss", typical: "10⁻² a metà training" },
  { id: "Lphys", grad: true,
    tex: R`\mathcal L_\text{phys}=\operatorname{mean}_{i<j,\,t,\,k}\ \mathrm{relu}\big(-\lambda_{\min}(\hat\rho_{ij})\big)`,
    steps: ["Da N̂ ed Ê previsti si costruisce, per ogni coppia i<j, la matrice 4×4 ρ̂_ij (Eq. 28).",
      "Si calcola l'autovalore minimo con eigvalsh (float32).",
      "Si tiene solo la parte negativa, relu(−λ_min): vale 0 se ρ̂ è già semidefinita positiva.",
      "Si mediano coppie, anchor e passi k. Il termine è sommato con peso λ_phys."],
    code: "losses.phys_loss, losses.rho_pairs", typical: "≈ 10⁻⁶ (quasi sempre nulla)" },
  { id: "Lprobe", grad: false,
    tex: R`\mathcal L_\text{probe}=\mathrm{MSE}(\hat h,h)+\mathrm{MSE}\big(\hat J,J\big)_{i\ne j}+\mathrm{BCE}\big(\hat a,\mathrm{adj}\big)_{i\ne j}`,
    steps: ["Si prendono h_i ed ē_ij all'ultima posizione di contesto e si staccano dal grafo di calcolo (detach).",
      "Tre lettori lineari producono ĥ_i, Ĵ_ij e il logit di adiacenza.",
      "MSE su h (n valori per traiettoria), MSE su J su tutte le coppie i≠j (archi con J ∈ [0.5, 1.5], non-archi con J = 0), BCE sull'adiacenza.",
      "Si sommano. Poiché le feature sono staccate, il gradiente arriva solo ai tre lettori, mai al corpo né alla head."],
    code: "losses.probe_loss, model/probe.py", typical: "≈ 1 e quasi costante nelle run vecchie; disattivato (λ = 0, nessun modulo probe) nelle run nuove" },
  { id: "Ltheta", grad: true,
    tex: R`\|\theta\|^2=\operatorname{mean}_{b,t}\sum_{k=1}^K\theta_{t,k}^2`,
    steps: ["Per ogni anchor la head produce il vettore θ_t di K coefficienti (Eq. 17).",
      "Si sommano i quadrati sulle K componenti.",
      "Si mediano su batch e anchor. Con λ_θ = 10⁻⁴ è un regolarizzatore molto debole."],
    code: "losses.total_loss", typical: "O(1); contributo ≈ 10⁻⁴" },
];

/* Stadi del tracciatore. dims: {n,d,de,nh,L,m,K,T,H}; sel: {kind: "node"|"pair", i, j}. pg: param_groups | null */
function traceStages(dims, sel, pg) {
  const { n, d, de, nh, L, m, K } = dims;
  const pa = !!dims.pair_attn, tri = !!dims.triangle, pl = dims.pair_latent || 0;   // opzioni architetturali della run
  const node = sel.kind === "node", i = sel.i, j = sel.j;
  const P = (key) => (pg && pg[key] != null ? ` · <b>${pg[key].toLocaleString("it-IT")}</b> parametri` : "");
  const sl = node ? `[:, :, ${i}, :]` : `[:, :, ${i}, ${j}, :]`;
  const rf = 3 * L + 1;
  const mix = (t) => `<span class="x">${t}</span>`;
  return [
    { n: 1, t: "Ingresso", role: node ? "node" : "pair",
      rows: node
        ? [["Cosa entra", `\\(N_t[${i},:]\\in\\mathbb R^3\\) per ogni \\(t=0\\dots T\\), più \\(\\log\\Delta t\\). Sono ⟨X<sub>${i}</sub>⟩, ⟨Y<sub>${i}</sub>⟩, ⟨Z<sub>${i}</sub>⟩.`],
           ["Forma", `<span class="mono">N[B, T+1, ${n}, 3]</span>, fetta <span class="mono">${sl}</span> → 3 numeri per passo`],
           ["Peso nei dati", `i nodi sono <b>${3 * n}</b> dei ${3 * n + 9 * n * (n - 1) / 2} numeri indipendenti per passo (${(100 * 3 * n / (3 * n + 9 * n * (n - 1) / 2)).toFixed(0)}%)`]]
        : [["Cosa entra", `\\(E_t[${i},${j},:]\\in\\mathbb R^9\\): le correlazioni ⟨σ<sup>a</sup><sub>${i}</sub>σ<sup>b</sup><sub>${j}</sub>⟩ con canale \\(c=3a+b\\). La coppia (${j},${i}) contiene gli stessi numeri con (a,b) scambiati.`],
           ["Forma", `<span class="mono">E[B, T+1, ${n}, ${n}, 9]</span>, fetta <span class="mono">${sl}</span> → 9 numeri per passo; la diagonale è 0`],
           ["Peso nei dati", `le coppie sono <b>${9 * n * (n - 1) / 2}</b> dei ${3 * n + 9 * n * (n - 1) / 2} numeri indipendenti per passo (${(100 * 9 * n * (n - 1) / 2 / (3 * n + 9 * n * (n - 1) / 2)).toFixed(0)}%)`]],
      cross: [] },
    { n: 2, t: "Embedding", role: node ? "node" : "pair",
      rows: node
        ? [["Operazione", `\\(h^{(0)}_{${i},t}=\\mathrm{MLP}_n[N_t[${i},:],\\log\\Delta t]+\\mathrm{PE}(t)\\)`],
           ["Forma", `<span class="mono">h[B, T+1, ${n}, ${d}]</span>, fetta <span class="mono">${sl}</span> → d = ${d} numeri`], ["Capacità", `MLP<sub>n</sub> 4→${4 * d}→${d}${P("embed_node")}`]]
        : [["Operazione", `\\(e^{(0)}_{${i}${j},t}=\\mathrm{MLP}_e(E_t[${i},${j},:])\\), diagonale azzerata. <b>Nessuna codifica posizionale</b>.`],
           ["Forma", `<span class="mono">e[B, T+1, ${n}, ${n}, ${de}]</span>, fetta <span class="mono">${sl}</span> → d<sub>e</sub> = ${de} numeri`], ["Capacità", `MLP<sub>e</sub> 9→${4 * de}→${de}${P("embed_pair")}`]],
      cross: [] },
    { n: 3, t: `Strato spaziale · attenzione (× ${L} blocchi)`, role: "both",
      rows: node
        ? [["Cosa riceve", `Il token del sito ${i} attende i ${n} siti (sé incluso), per ciascuna delle ${nh} teste, con logit \\(q_i\\cdot k_j/\\sqrt{d_h}+b_{ij}\\).`],
           ["Dalle coppie", `Riceve da <b>tutte</b> le coppie \\((${i},j)\\): il bias \\(b_{ij}\\) e il valore \\(W_{ev}\\mathrm{LN}(e_{${i}j})\\). È il solo punto in cui le coppie scrivono nei nodi.`],
           ["Capacità", `qkv, W<sub>o</sub>, FFN${P("spatial_node")}; bias e valori${P("spatial_pair_to_node")}`]]
        : [["Cosa fa", `La coppia (${i},${j}) <b>non è modificata</b> qui: viene letta. Entra come bias \\(b_{${i}${j}}\\) nel logit di attenzione del sito ${i} (e ${j} tramite \\(e_{${j}${i}}\\)) e come valore \\(W_{ev}\\mathrm{LN}(e_{${i}${j}})\\).`],
           ["Verso i nodi", `Scrive nei token di ${i} e di ${j}, pesata da \\(\\alpha_{${i}${j}}\\) e \\(\\alpha_{${j}${i}}\\). Questi pesi, mediati, sono \\(\\bar\\alpha_{${i}${j}}\\), la stima del grafo.`]],
      cross: [node ? "↖ riceve dalle coppie (i, ·)" : "↗ scrive nei nodi i e j"] },
    { n: 4, t: `Strato spaziale · aggiornamento delle coppie`, role: "both",
      rows: node
        ? [["Cosa dà", `Il token \\(h_${i}\\) entra, insieme a quello di ogni altro sito \\(j\\), nel prodotto \\((W_a\\mathrm{LN}h_${i})\\odot(W_b\\mathrm{LN}h_j)\\) che aggiorna la coppia \\((${i},j)\\).`]]
        : [["Operazione", `\\(e_{${i}${j}}\\leftarrow e_{${i}${j}}+\\mathrm{MLP}_{eu}(\\mathrm{LN}e_{${i}${j}};\\,(W_a\\mathrm{LN}h_${i})\\odot(W_b\\mathrm{LN}h_${j}))\\), poi \\((e_{${i}${j}}+e_{${j}${i}})/2\\).`],
           ["Cosa vede", tri ? `Sé stessa, i due siti ${i} e ${j} e, con l'<b>aggiornamento triangolare</b>, le coppie \\(e_{${i}k}\\) ed \\(e_{${j}k}\\) su tutti i terzi siti k: \\(e_{${i}${j}}\\leftarrow e_{${i}${j}}+\\mathrm{Tri}(\\tfrac1n\\sum_k a_{${i}k}\\odot a_{${j}k})\\).`
             : `Solo sé stessa e i due siti ${i} e ${j}. <b>Non vede le altre coppie</b> \\(e_{${i}k}\\), \\(e_{k${j}}\\): nessun aggiornamento «triangolare». Le informazioni degli altri siti arrivano solo indirettamente, già mescolate in \\(h_${i}\\), \\(h_${j}\\).`],
           ["Capacità", `MLP<sub>eu</sub>, W<sub>a</sub>, W<sub>b</sub>${P("spatial_pair_update")}`]],
      cross: [node ? "↘ scrive nelle coppie (i, ·)" : "↖ riceve dai nodi i e j"] },
    { n: 5, t: `Strato temporale (× ${L} blocchi)`, role: node ? "node" : "pair",
      rows: node
        ? [["Operazione", `Attenzione causale multi-testa sui passi \\(t'\\le t\\) del solo sito ${i} (RoPE), poi FFN. Non confronta mai siti diversi.`],
           ["Campo recettivo", `tutta la finestra: <b>T+1</b> passi, completamente connessi`], ["Capacità", `MHA + FFN${P("temporal_node")}`]]
        : [["Operazione", pa ? `Convoluzione causale depthwise di kernel 4 <b>più attenzione causale multi-testa</b> sui passi \\(t'\\le t\\) della coppia (RoPE), come per i siti.`
             : `Convoluzione causale <b>depthwise</b> di kernel 4: ogni canale vede solo se stesso a \\(t, t-1, t-2, t-3\\). Nessuna attenzione, nessun mescolamento tra canali.`],
           ["Campo recettivo", pa ? `tutta la finestra: <b>T+1</b> passi, come i nodi` : `<b>${rf}</b> passi dopo ${L} blocchi (3L+1), contro la finestra intera dei nodi`],
           ["Capacità", `${pa ? "conv + MHA" : "un filtro per canale"}${P("temporal_pair")}`]],
      cross: [] },
    { n: 6, t: "LayerNorm finale e ancoraggio", role: node ? "node" : "pair",
      rows: node ? [["Esce", `\\(h^{(L)}_{${i},t}\\in\\mathbb R^{${d}}\\) nella posizione dell'anchor.`]] : [["Esce", `\\(\\bar e_{${i}${j}}=e^{(L)}_{${i}${j},t}\\in\\mathbb R^{${de}}\\) nella posizione dell'anchor.`]],
      cross: [] },
    { n: 7, t: "Head A · cosa legge dal corpo", role: "both",
      rows: node
        ? [["Latente", `\\(z_${i}=W_z h_${i}\\in\\mathbb R^{${m}}\\): è lo stato iniziale del sito nella propagazione.`],
           dims.pauli_A ? ["Coefficienti", `Nessun θ: il sito ${i} entra, con gli altri, nell'unico MLP che produce i coefficienti di Pauli di ogni blocco \\(A_{${i}j}\\) e di \\(A_{${i}${i}}\\) (ingresso \\([h_${i},h_j,e_{${i}j}]\\)).`]
             : ["Coefficienti", `\\(\\theta=W_\\theta\\tfrac1n\\sum_i h_i\\in\\mathbb R^{${K}}\\): il sito ${i} contribuisce per 1/${n} a una descrizione globale; θ fissa il blocco diagonale \\(A_{ii}\\) uguale per tutti i siti.`],
           ["Capacità", `W<sub>z</sub>, ${dims.pauli_A ? "MLP dei coefficienti" : "W<sub>θ</sub>, basi"}, decoder${P("head_node_and_generator")}`]]
        : dims.pauli_A ? [["Coefficienti", `\\(c=\\mathrm{MLP}([h_${i},h_${j},\\bar e_{${i}${j}}])\\): coefficienti su una base fissa di stringhe di Pauli (peso ≤ w) dello spazio latente di dimensione ${m}.`],
           ["Accoppiamento", `\\(A_{${i}${j}}=\\sum_s c_sP_s\\), base fissa non appresa. Nessun gate: il punteggio d'arco è \\(\\|A_{${i}${j}}\\|_F\\) normalizzato. <b>La feature di coppia entra nella dinamica solo da qui.</b>`],
           ["Capacità", `MLP dei coefficienti${P("head_pair")}`]]
        : [["Gate", `\\(g_{${i}${j}}=\\sigma(w_g^\\top\\bar e_{${i}${j}})\\in(0,1)\\): la stima dell'arco.`],
           ["Accoppiamento", `\\(A_{${i}${j}}=g_{${i}${j}}\\,\\mathrm{reshape}_{${m}\\times${m}}(W_A\\bar e_{${i}${j}})\\): blocco di G che lega i latenti dei siti ${i} e ${j}. <b>È l'unico punto in cui la feature di coppia entra nella dinamica.</b>`],
           ...(pl ? [["Latente di coppia", `\\(u_{${i}${j}}(0)=W_u\\bar e_{${i}${j}}\\in\\mathbb R^{${pl}}\\), ruotato da \\(C_{${i}${j}}=B-B^\\top\\) letto da \\(\\bar e_{${i}${j}}\\): \\(u(k)=\\exp(k\\Delta t\\,C)u(0)\\). Qui la feature di coppia entra <b>anche</b> nella previsione.`]] : []),
           ["Capacità", `w<sub>g</sub>, W<sub>A</sub>${pl ? ", W<sub>u</sub>, W<sub>C</sub>, D<sub>u</sub>" : ""}${P("head_pair")}`]],
      cross: [node ? "z, θ ← da nodi" : "g, A_ij ← da coppie"] },
    { n: 8, t: "Propagazione", role: "both",
      rows: [["Operazione", `\\(Z_{t+k}=\\exp(\\Delta t\\,G)^k Z_t\\), con \\(G\\) di dimensione ${n * m}×${n * m}.`],
        node ? ["Ruolo", `Il latente \\(z_${i}\\) (${m} numeri) è il blocco ${i} del vettore \\(Z\\); evolve secondo \\(A_{ii}\\) (locale) e i blocchi \\(A_{${i}j}\\) (accoppiamento con gli altri siti).`]
             : ["Ruolo", `Il blocco \\(A_{${i}${j}}\\) è il solo canale con cui \\(z_${j}\\) influenza \\(z_${i}\\) (e viceversa tramite \\(-A^\\top\\)): è qui che l'accoppiamento J deve apparire.`]],
      cross: [] },
    { n: 9, t: "Decoder", role: node ? "node" : "pair",
      rows: node
        ? [["Operazione", `\\(\\hat N_{t+k}[${i},:]=\\tanh(D_n z_{${i},t+k})\\): lineare nel latente.`], ["Cosa dipende da cosa", "la previsione del nodo dipende solo dal proprio latente."]]
        : [["Operazione", pl ? `\\(\\hat E_{t+k}[${i},${j},c]=\\tanh(z_${i}^\\top B_c z_${j}+\\beta_c+(D_uu_{${i}${j}})_c)\\), poi simmetrizzata.`
             : `\\(\\hat E_{t+k}[${i},${j},c]=\\tanh(z_${i}^\\top B_c z_${j}+\\beta_c)\\), poi simmetrizzata.`],
           ["Cosa dipende da cosa", pl ? `Dai latenti dei due siti <b>e dal latente di coppia</b> \\(u_{${i}${j}}\\), quindi direttamente da \\(\\bar e_{${i}${j}}\\).`
             : `La previsione della coppia dipende <b>solo dai latenti dei due siti</b>, non da \\(\\bar e_{${i}${j}}\\): la correlazione iniziale a t = T deve essere già codificata, in forma bilineare, in \\(z_${i}\\) e \\(z_${j}\\).`]],
      cross: [] },
    { n: 10, t: "Probe e loss", role: "both",
      rows: node
        ? [dims.probe ? ["Probe", `\\(\\hat h_${i}=w_h^\\top h_${i}\\): lettura lineare del campo. Stop-gradient.`] : ["Probe", "assente in questa run: h vero non è mai bersaglio. A posteriori si confronta ‖G<sub>ii</sub>‖ con h<sub>i</sub>."], ["Loss", `Errore su \\(\\hat N[${i},:]\\) in \\(\\mathcal L_\\text{pred}\\); il gradiente arriva a \\(h_${i}\\) attraverso \\(z_${i}\\) e θ.`]]
        : [dims.probe ? ["Probe", `\\(\\hat J_{${i}${j}}=w_J^\\top\\bar e_{${i}${j}}\\), \\(\\hat a_{${i}${j}}\\): lettura lineare di J e dell'arco. Stop-gradient.`] : ["Probe", "assente in questa run: J vero non è mai bersaglio. A posteriori si confronta ‖G<sub>ij</sub>‖ con J<sub>ij</sub> e con l'adiacenza."],
           ["Loss", `Errore su \\(\\hat E[${i},${j},:]\\) in \\(\\mathcal L_\\text{pred}\\); il gradiente arriva a \\(\\bar e_{${i}${j}}\\) <b>solo</b> attraverso il gate e \\(A_{${i}${j}}\\), e a \\(h_${i}\\), \\(h_${j}\\) attraverso i latenti.`]],
      cross: [] },
  ];
}

/* Ipotesi sull'asimmetria: ogni voce è un fatto verificabile nel codice. */
const ASYMMETRY = [
  ["Capacità temporale", "I nodi hanno attenzione causale su tutta la finestra; le coppie solo una convoluzione depthwise di 4 passi per canale (campo recettivo 3L+1), senza mescolare i canali. Eppure le coppie sono oltre il 90% dei numeri in ingresso.", "model/temporal.py", "pair_attn"],
  ["Nessuna codifica del tempo sulle coppie", "PE(t) è sommata solo ai token di sito; la coppia conosce solo l'ordine locale dato dai 4 tap della convoluzione.", "model/embed.py", "pair_attn"],
  ["Aggiornamento di coppia locale", "e_ij si aggiorna solo da sé stessa e da h_i, h_j. Non c'è un passo che aggreghi e_ik ed e_kj (informazione «triangolare»).", "model/spatial.py", "triangle"],
  ["Decoder delle coppie cieco a ē_ij", "Ê_ij(k) = tanh(z_iᵀ B_c z_j + β_c): la correlazione prevista dipende dai latenti dei siti, non dalla feature di coppia. ē_ij influisce solo su gate e blocchi A_ij di G.", "model/head_generator.py", "pair_latent"],
  ["Probe lineare e bersaglio su tutte le coppie", "Ĵ è una lettura lineare di ē_ij; se J è codificato in modo non lineare risulta sottostimato. L'errore su J include le coppie senza arco (J = 0), che dominano la varianza.", "model/probe.py, losses.py"],
  ["Il probe non aiuta il corpo", "L_probe è staccato dal corpo: se J non serve a prevedere meglio i nodi, nulla spinge le feature di coppia a contenerlo.", "losses.py"],
];
const CANDIDATES = [
  ["Attenzione temporale (o GRU) anche sulle coppie", "Dà alle coppie lo stesso campo recettivo dei nodi. Costo O(n²T²d_e): per n = 8 e T = 64 accettabile.", "pair_attn"],
  ["Aggiornamento triangolare di e_ij", "Aggregare e_ik ed e_kj sulle terne, come nei blocchi triangolari di AlphaFold, per far arrivare alle coppie la struttura del grafo.", "triangle"],
  ["Latente di coppia o decoder con ē_ij", "Propagare anche un latente per coppia con lo stesso generatore, oppure aggiungere un termine che legga ē_ij nel decoder di Ê, così la correlazione iniziale non passa solo dai latenti di sito.", "pair_latent"],
  ["Esperimento discriminante a basso costo", "Addestrare tre varianti brevi: stesso modello con dataset ×2; d_e = 64; e coppie con attenzione temporale. Se solo l'ultima abbassa l'errore relativo su Ĵ, il collo di bottiglia è architetturale."],
];

return { EQS, LOSS_CARDS, traceStages, ASYMMETRY, CANDIDATES };
})();
if (typeof window !== "undefined") window.MODEL_INFO = MODEL_INFO;
