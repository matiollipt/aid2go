/* Offline protein and ontology explorers. Cytoscape.js is bundled in this HTML. */
(() => {
  'use strict';
  const source = D.explorer;
  const proteinIndex = new Map(source.ids.map((id, i) => [id, i]));
  const analyzedIndex = new Map(D.go.map((term, i) => [term.id, i]));
  const proteinChoices = source.ids.map(id => ({id, label: id})).sort((a,b) => a.id.localeCompare(b.id));
  const goChoices = D.go.map(term => ({id:term.id, name:term.name, label:`${term.id} · ${term.name}`})).sort((a,b) => a.id.localeCompare(b.id));
  const softAspect={BP:'#eaf1fa',MF:'#fff1e6',CC:'#f0eafa',unknown:'#eff2f6'};
  const ontology = source.ontology;
  const ontologyIndex = new Map(ontology.nodes.map((term,i) => [term[0],i]));
  const incident = Array.from({length:source.ids.length}, () => []);
  source.edges.forEach((edge,i) => {incident[edge[0]].push(i);incident[edge[1]].push(i);});
  for (const edges of incident) edges.sort((a,b) => source.edges[b][2]-source.edges[a][2] || a-b);
  const parents = Array.from({length:ontology.nodes.length}, () => []);
  const children = Array.from({length:ontology.nodes.length}, () => []);
  ontology.edges.forEach((edge,i) => {parents[edge[0]].push(i);children[edge[1]].push(i);});
  const incoming = source.incoming_degree;
  let root = 0;
  let goRoot = D.go.find(term => term.id==='GO:0004672')?.id || D.go[0].id;
  let networkView;
  const pickers = new Map();
  const cyViews = new Map();
  const scores = decodeScores(source.predictions);

  function decodeScores(packed) {
    if (packed.encoding!=='base64-float32-le' || packed.shape[0]!==source.ids.length || packed.shape[1]!==D.go.length) throw Error('Prediction matrix metadata mismatch');
    const binary = atob(packed.scores_b64);
    if (binary.length!==packed.shape[0]*packed.shape[1]*4) throw Error('Prediction matrix byte count mismatch');
    const bytes = new Uint8Array(binary.length);
    for (let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
    const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0]===1;
    if (littleEndian) return new Float32Array(bytes.buffer);
    const view = new DataView(bytes.buffer), values = new Float32Array(binary.length/4);
    for (let i=0;i<values.length;i++) values[i]=view.getFloat32(i*4,true);
    return values;
  }

  function mountPicker(prefix, choices, onSelect, statusId) {
    const input=$(prefix+'-input'), menu=$(prefix+'-options'), toggle=$(prefix+'-toggle');
    let matches=[], active=-1, rendered=0;
    input.setAttribute('role','combobox');input.setAttribute('aria-autocomplete','list');
    input.setAttribute('aria-controls',menu.id);input.setAttribute('aria-expanded','false');
    function close() {menu.hidden=true;input.setAttribute('aria-expanded','false');input.removeAttribute('aria-activedescendant');}
    function choose(choice) {input.value=choice.id;close();$(statusId).textContent='';onSelect(choice.id);}
    function appendOptions() {
      const end=Math.min(matches.length,rendered+100);
      const fragment=document.createDocumentFragment();
      for (let i=rendered;i<end;i++) {
        const option=document.createElement('div');option.className='picker-option';
        option.id=menu.id+'-'+i;option.setAttribute('role','option');option.setAttribute('aria-selected',i===active?'true':'false');
        option.textContent=matches[i].label;
        option.addEventListener('mousedown',event => event.preventDefault());
        option.addEventListener('click',() => choose(matches[i]));fragment.append(option);
      }
      menu.append(fragment);rendered=end;
    }
    function show(query='') {
      matches=choices.filter(choice => choice.label.toLowerCase().includes(query.toLowerCase()));active=-1;rendered=0;menu.replaceChildren();
      if (!matches.length) {const empty=document.createElement('div');empty.className='picker-empty';empty.textContent='No matching IDs in this analyzed dataset.';menu.append(empty);}
      else appendOptions();
      menu.hidden=false;input.setAttribute('aria-expanded','true');menu.scrollTop=0;
    }
    function validate() {
      const query=input.value.trim().toLowerCase();
      const exact=choices.find(choice => choice.id.toLowerCase()===query || choice.label.toLowerCase()===query || choice.name?.toLowerCase()===query);
      if (exact) choose(exact);
      else {$(statusId).textContent='Select an ID from the analyzed dataset. The last valid selection is retained.';close();}
    }
    input.addEventListener('input',() => {show(input.value.trim());$(statusId).textContent='';});
    input.addEventListener('focus',() => show(input.value));
    input.addEventListener('change',validate);
    input.addEventListener('keydown',event => {
      if (event.key==='Escape') {close();return;}
      if (event.key==='ArrowDown' || event.key==='ArrowUp') {
        event.preventDefault();if(menu.hidden)show(input.value);
        if(!matches.length)return;
        active=Math.max(0,Math.min(matches.length-1,active+(event.key==='ArrowDown'?1:-1)));
        while(rendered<=active)appendOptions();
        menu.querySelectorAll('[role=option]').forEach((option,i) => {option.classList.toggle('active',i===active);option.setAttribute('aria-selected',i===active?'true':'false');});
        const option=$(menu.id+'-'+active);input.setAttribute('aria-activedescendant',option.id);option.scrollIntoView({block:'nearest'});
      } else if (event.key==='Enter') {event.preventDefault();if(!menu.hidden && active>=0)choose(matches[active]);else validate();}
    });
    toggle.addEventListener('click',() => {if(menu.hidden)show('');else close();});
    menu.addEventListener('scroll',() => {if(menu.scrollTop+menu.clientHeight>=menu.scrollHeight-50 && rendered<matches.length)appendOptions();});
    document.addEventListener('pointerdown',event => {if(!input.closest('.picker').contains(event.target))close();});
    const picker={set:id=>{input.value=id;close();},validate,close,show};pickers.set(prefix,picker);return picker;
  }

  function proteinNeighborhood(rootIndex,hops,minimum,limit) {
    const depths=new Map([[rootIndex,0]]), queue=[rootIndex], parentEdges=new Map();
    for (let cursor=0;cursor<queue.length;cursor++) {
      const current=queue[cursor], depth=depths.get(current);
      if(depth>=hops)continue;
      for (const edgeIndex of incident[current]) {
        const edge=source.edges[edgeIndex];if(edge[2]<minimum)continue;
        const neighbor=edge[0]===current?edge[1]:edge[0];
        if(!depths.has(neighbor)){depths.set(neighbor,depth+1);parentEdges.set(neighbor,edgeIndex);queue.push(neighbor);}
      }
    }
    const shown=queue.slice(0,limit), selected=new Set(shown);
    const induced=[];source.edges.forEach((edge,i) => {if(edge[2]>=minimum&&selected.has(edge[0])&&selected.has(edge[1]))induced.push(i);});
    const spanning=new Set(shown.slice(1).map(index => parentEdges.get(index)));
    const edgeLimit=1800;
    induced.sort((a,b) => Number(spanning.has(b))-Number(spanning.has(a)) || source.edges[b][2]-source.edges[a][2] || a-b);
    return {root:rootIndex,hops,minimum,nodes:shown,depths,edges:induced.slice(0,edgeLimit),
            eligibleNodes:queue.length,eligibleEdges:induced.length,omittedNodes:queue.length-shown.length,omittedEdges:Math.max(0,induced.length-edgeLimit)};
  }

  function cyStyle() {
    return [
      {selector:'node',style:{'width':'data(size)','height':'data(size)','background-color':'data(color)','border-width':1.5,'border-color':'#ffffff','label':'data(label)','font-family':'system-ui','font-size':12,'text-wrap':'wrap','text-max-width':120,'text-valign':'bottom','text-margin-y':5,'color':'#253850','text-background-color':'#fbfcfe','text-background-opacity':.85,'text-background-padding':2}},
      {selector:'edge',style:{'line-color':'#a7b5c5','target-arrow-color':'#8a9aae','target-arrow-shape':'triangle','arrow-scale':.6,'width':'data(width)','curve-style':'bezier','opacity':.48}},
      {selector:'node.root',style:{'border-width':3,'border-color':'#ba5b21','z-index':10,'font-weight':'bold'}},
      {selector:'node.context',style:{'border-style':'dashed','border-color':'#8996a9','border-width':2}},
      {selector:'edge.part_of',style:{'line-style':'dashed','line-color':'#2563a6','target-arrow-color':'#2563a6'}},
      {selector:'.dim',style:{'opacity':.15}},
      {selector:'.focused',style:{'opacity':1,'z-index':20}},
    ];
  }

  function ensureCy(id, kind) {
    if(cyViews.has(id))return cyViews.get(id);
    const cy=cytoscape({container:$(id),elements:[],style:cyStyle(),minZoom:.08,maxZoom:4,
                        pixelRatio:Math.min(devicePixelRatio||1,2),wheelSensitivity:.2,hideEdgesOnViewport:true});
    cy.on('mouseover','node, edge',event => inspectGraphElement(event.target,kind));
    cy.on('tap','node',event => {const node=event.target;if(kind==='protein')selectProtein(node.id());else if(node.data('analyzed'))selectGoTerm(node.id());else inspectGraphElement(node,kind);});
    cy.on('tap',event => {if(event.target===cy){cy.elements().removeClass('dim focused');}});
    cyViews.set(id,cy);return cy;
  }

  function inspectGraphElement(element,kind) {
    const cy=element.cy();cy.elements().removeClass('dim focused');
    const neighborhood=element.isNode()?element.closedNeighborhood():element.connectedNodes().union(element);
    cy.elements().not(neighborhood).addClass('dim');neighborhood.addClass('focused');
    let text;
    if(kind==='protein') {
      if(element.isNode()) {const index=proteinIndex.get(element.id());text=`${element.id()} · ${element.data('hop')} hop(s) from ${source.ids[root]} · incoming degree ${incoming[index]} · outgoing degree ${source.outgoing_degree[index]}`;}
      else text=`${element.data('source')} → ${element.data('target')} · ESM cosine ${fmt(element.data('cosine'),5)} · L2 distance ${fmt(element.data('l2'),5)}`;
      $('protein-node-inspector').textContent=text;$('association-network-title').textContent=`ESM network rooted at ${source.ids[root]} · inspect: ${element.isNode()?element.id():element.data('source')+' → '+element.data('target')}`;
    } else {
      if(element.isNode()){const term=element.data();text=`${term.id} · ${term.name} · ${term.aspect} · ${term.analyzed?'analyzed vocabulary':'full-ontology context only'}${term.ia===null?'':` · IA ${fmt(term.ia,4)} bits`}`;}
      else text=`${element.data('source')} → ${element.data('target')} · ${element.data('relation')} · child → parent`;
      $('go-node-inspector').textContent=text;
    }
  }

  function drawProteinView(id,view) {
    const cy=ensureCy(id,'protein');
    const elements=view.nodes.map(index => ({data:{id:source.ids[index],hop:view.depths.get(index),
      label:index===root||view.nodes.length<=60?source.ids[index]:'',size:12+Math.min(22,Math.sqrt(incoming[index])*2),
      color:index===root?C.orange:[C.blue,C.purple,C.gray][Math.max(0,view.depths.get(index)-1)]},classes:index===root?'root':''}));
    for(const index of view.edges){const edge=source.edges[index];elements.push({data:{id:'esm-'+index,source:source.ids[edge[0]],target:source.ids[edge[1]],cosine:edge[2],l2:edge[3],width:.7+2.3*Math.max(0,edge[2])}});}
    cy.batch(() => {cy.elements().remove();cy.add(elements);});cy.resize();
    cy.layout({name:'concentric',concentric:node=>view.hops+1-node.data('hop'),levelWidth:()=>1,
               minNodeSpacing:view.nodes.length>80?12:25,avoidOverlap:true,padding:28,animate:false,nodeDimensionsIncludeLabels:view.nodes.length<=60}).run();
    $(id).dataset.root=source.ids[root];$(id).dataset.nodes=String(view.nodes.length);$(id).dataset.hops=String(view.hops);
  }

  function updateNetworks() {
    const hops=Number($('network-hops').value),minimum=Number($('edge-threshold').value),limit=Number($('network-limit').value);
    $('association-hops').value=String(hops);$('edge-threshold-value').textContent=minimum.toFixed(2);
    networkView=proteinNeighborhood(root,hops,minimum,limit);
    drawProteinView('protein-network',networkView);drawProteinView('association-network',networkView);
    const text=`Root ${source.ids[root]} · ${hops} hop(s), incoming + outgoing traversal · ${networkView.nodes.length} / ${networkView.eligibleNodes} reachable proteins · ${networkView.edges.length} / ${networkView.eligibleEdges} induced directed edges among displayed proteins · omitted ${networkView.omittedNodes} proteins and ${networkView.omittedEdges} displayed-subgraph edges`;
    $('protein-network-count').textContent=text;$('association-network-count').textContent=text;
    $('protein-node-inspector').textContent=`${source.ids[root]} · selected root · incoming degree ${incoming[root]} · outgoing degree ${source.outgoing_degree[root]}. Hover or tap nodes/edges for exact IDs and weights. Click a protein node to re-root at one hop.`;
    $('association-network-title').textContent=`ESM network rooted at ${source.ids[root]}`;
  }

  function predictedRows(index,minimum,topN) {
    const offset=index*D.go.length;
    const terms=[];
    for(let term=0;term<D.go.length;term++){const score=scores[offset+term];if(score>=minimum)terms.push({term,score});}
    terms.sort((a,b)=>b.score-a.score||D.go[a.term].id.localeCompare(D.go[b.term].id));
    return {rows:terms.slice(0,topN),qualified:terms.length};
  }

  function renderTermList(container,rows,predicted) {
    const fragment=document.createDocumentFragment();const observed=new Set(source.annotations[root]);
    for(const entry of rows){const term=D.go[entry.term];const button=document.createElement('button');button.type='button';button.className='term-row';button.dataset.go=term.id;
      if(predicted)button.dataset.score=String(entry.score);
      const id=document.createElement('span');id.className='term-id';id.textContent=term.id;
      const name=document.createElement('span');name.className='term-name';name.textContent=term.name;
      if(predicted){const badge=document.createElement('small');badge.textContent=observed.has(entry.term)?'Also annotated':'Not observed in this artifact';name.append(badge);}
      const meta=document.createElement('span');meta.className='term-meta';meta.textContent=term.aspect+' · '+fmt(term.ia,2)+' bits';
      const score=document.createElement('span');score.className='term-score';score.textContent=predicted?entry.score.toFixed(4):'Observed';
      button.append(id,name,meta,score);button.title=`${term.id} · ${term.name}\n${term.aspect} · IA ${fmt(term.ia,4)} bits${predicted?'\nScore '+entry.score.toFixed(6):''}`;
      button.addEventListener('click',()=>{selectGoTerm(term.id);container.querySelectorAll('.term-row').forEach(row=>row.classList.toggle('active',row===button));});fragment.append(button);
    }
    if(!rows.length){const empty=document.createElement('div');empty.className='association-empty';empty.textContent=predicted?'No analyzed GO terms meet this score threshold.':'No observed analyzed GO associations for this protein.';fragment.append(empty);}
    container.replaceChildren(fragment);container.scrollTop=0;container.dataset.protein=source.ids[root];
  }

  function updateLists() {
    const annotated=source.annotations[root].map(term=>({term})).sort((a,b)=>D.go[a.term].aspect.localeCompare(D.go[b.term].aspect)||D.go[a.term].id.localeCompare(D.go[b.term].id));
    renderTermList($('annotated-terms'),annotated,false);
    $('annotated-summary').textContent=`${source.ids[root]} · ${annotated.length} observed GO associations · all analyzed terms shown; scroll to read.`;
    updatePredictions();
  }

  function updatePredictions() {
    const minimum=Number($('prediction-threshold').value),topN=Number($('prediction-top-n').value);
    if(!Number.isFinite(minimum)||minimum<0||minimum>1){$('predicted-protein-status').textContent='Minimum score must be between 0 and 1; the previous results are retained.';return;}
    $('predicted-protein-status').textContent='';const result=predictedRows(root,minimum,topN);
    renderTermList($('predicted-terms'),result.rows,true);
    $('predicted-summary').textContent=`${source.ids[root]} · ${result.rows.length} shown of ${result.qualified} terms with score ≥ ${minimum.toFixed(2)} · ranked highest first${result.qualified>result.rows.length?' · increase Top N to see more':''}.`;
  }

  function selectProtein(id) {
    const index=proteinIndex.get(id);if(index===undefined)return false;
    root=index;for(const prefix of ['network-protein','annotated-protein','predicted-protein']){pickers.get(prefix)?.set(id);$(prefix+'-status').textContent='';}
    $('network-hops').value='1';$('association-hops').value='1';$('edge-threshold').value='0';
    updateLists();updateNetworks();return true;
  }

  function goNeighborhood(id,parentHops,childHops,limit=80) {
    const focal=ontologyIndex.get(id);const depths=new Map([[focal,0]]);
    function traverse(adjacency,hops,parentDirection) {
      const queue=[focal], visited=new Map([[focal,0]]);
      for(let cursor=0;cursor<queue.length;cursor++) {const node=queue[cursor],depth=visited.get(node);if(depth>=hops)continue;
        for(const edgeIndex of adjacency[node]) {const edge=ontology.edges[edgeIndex],neighbor=edge[parentDirection?1:0];
          if(!visited.has(neighbor)){visited.set(neighbor,depth+1);queue.push(neighbor);depths.set(neighbor,Math.min(depths.get(neighbor)??Infinity,depth+1));}}
      }
    }
    traverse(parents,parentHops,true);traverse(children,childHops,false);
    // Parent context first for equal-distance ties; keep parents before deeper branches.
    const ordered=[...depths.keys()].sort((a,b)=>depths.get(a)-depths.get(b)||a-b);
    const nodes=ordered.slice(0,limit),selected=new Set(nodes);
    const edges=ontology.edges.filter(edge=>selected.has(edge[0])&&selected.has(edge[1]));
    return {focal,nodes,edges,eligible:ordered.length,omitted:ordered.length-nodes.length};
  }

  function layeredGoPositions(view) {
    const remaining=new Map(view.nodes.map(index=>[index,0]));const kids=new Map(view.nodes.map(index=>[index,[]]));
    for(const [child,parent] of view.edges){remaining.set(child,remaining.get(child)+1);kids.get(parent).push(child);}
    const queue=view.nodes.filter(index=>remaining.get(index)===0),depths=new Map(queue.map(index=>[index,0]));
    for(let cursor=0;cursor<queue.length;cursor++){const parent=queue[cursor];for(const child of kids.get(parent)){
      depths.set(child,Math.max(depths.get(child)||0,depths.get(parent)+1));remaining.set(child,remaining.get(child)-1);if(remaining.get(child)===0)queue.push(child);
    }}
    const levels=new Map();for(const index of view.nodes){const depth=depths.get(index)||0;if(!levels.has(depth))levels.set(depth,[]);levels.get(depth).push(index);}
    const positions=new Map();let y=50;
    for(const [,nodes] of [...levels].sort((a,b)=>a[0]-b[0])){
      nodes.sort((a,b)=>a===view.focal?-1:b===view.focal?1:a-b);
      const columns=Math.min(5,nodes.length),rows=Math.ceil(nodes.length/columns);
      nodes.forEach((index,i)=>positions.set(index,{x:80+(i%columns)*175+(5-columns)*87.5,y:y+Math.floor(i/columns)*125}));y+=rows*125+30;
    }
    return positions;
  }

  function selectGoTerm(id) {
    if(!analyzedIndex.has(id))return false;goRoot=id;pickers.get('go-tree')?.set(id);$('go-tree-status').textContent='';selectGo(id);updateGo();return true;
  }

  function updateGo() {
    const view=goNeighborhood(goRoot,Number($('go-parent-hops').value),Number($('go-child-hops').value));
    const positions=layeredGoPositions(view),cy=ensureCy('go-tree-view','go');
    const elements=view.nodes.map(index=>{const [id,name,aspect]=ontology.nodes[index],term=D.go[analyzedIndex.get(id)];return {
      data:{id,name,aspect,analyzed:Boolean(term),ia:term?term.ia:null,label:`${name.length>70?name.slice(0,67)+'…':name}\n${id}`,size:54,color:softAspect[aspect]||softAspect.unknown,border:id===goRoot?C.orange:C[aspect]||C.gray},
      classes:(id===goRoot?'root ':'')+(!term?'context':''),position:positions.get(index)};});
    view.edges.forEach((edge,index)=>elements.push({data:{id:'go-edge-'+index,source:ontology.nodes[edge[0]][0],target:ontology.nodes[edge[1]][0],relation:edge[2],width:1.2},classes:edge[2]}));
    cy.batch(()=>{cy.elements().remove();cy.add(elements);cy.nodes().style({'shape':'round-rectangle','width':160,'height':80,'text-valign':'center','text-margin-y':0,'font-size':13,'text-max-width':140,'border-color':'data(border)','text-background-opacity':0});});
    cy.resize();cy.layout({name:'preset',fit:false,padding:25,animate:false}).run();cy.zoom(.9);cy.center(cy.getElementById(goRoot));
    $('go-tree-view').dataset.root=goRoot;$('go-tree-view').dataset.nodes=String(view.nodes.length);
    $('go-tree-note').textContent=`${goRoot} · ${view.nodes.length} / ${view.eligible} neighborhood terms shown · ${view.edges.length} is_a/part_of relations · omitted ${view.omitted} terms at the 80-node display cap. Biological parents above children; arrows child → parent. Pan/zoom to read neighboring branches or use Fit graph for the overview. Hover for complete term names. Context nodes preserve paths outside the analyzed vocabulary; they have no displayed cohort IA.`;
    const term=D.go[analyzedIndex.get(goRoot)];$('go-node-inspector').textContent=goText(term).replaceAll('\n',' · ');
  }

  for(const prefix of ['network-protein','annotated-protein','predicted-protein'])mountPicker(prefix,proteinChoices,selectProtein,prefix+'-status');
  mountPicker('go-tree',goChoices,selectGoTerm,'go-tree-status');
  for(const id of ['network-hops','network-limit','edge-threshold'])$(id).addEventListener(id==='edge-threshold'?'input':'change',updateNetworks);
  $('association-hops').addEventListener('change',()=>{$('network-hops').value=$('association-hops').value;updateNetworks();});
  $('prediction-threshold').addEventListener('input',updatePredictions);$('prediction-top-n').addEventListener('change',updatePredictions);
  for(const id of ['go-parent-hops','go-child-hops'])$(id).addEventListener('change',updateGo);
  for(const [button,id] of [['network-fit','protein-network'],['association-fit','association-network'],['go-fit','go-tree-view']])$(button).onclick=()=>{const cy=cyViews.get(id);cy.resize();cy.fit(undefined,25);};
  $('go-readable').onclick=()=>{const cy=cyViews.get('go-tree-view');cy.zoom(1);cy.center(cy.getElementById(goRoot));};
  const checkpoint=D.provenance.find(item=>item.path.endsWith('/best_model.pt'));
  $('predicted-provenance').textContent=`${checkpoint?.path||'Current best checkpoint'} · SHA256 ${source.predictions.provenance.checkpoint.sha256.slice(0,12)}… · max true-path propagation (is_a / part_of) · full processed annotation context · 1,000 analyzed terms.`;
  selectProtein(source.ids[0]);selectGoTerm(goRoot);
  window.aid2goExplorers={selectProtein,selectGoTerm,proteinNeighborhood,goNeighborhood,predictedRows,
    get root(){return source.ids[root];},get network(){return networkView;},charts:cyViews,
    score:(protein,term)=>scores[proteinIndex.get(protein)*D.go.length+analyzedIndex.get(term)]};
  document.body.dataset.galleryReady='true';document.body.dataset.explorersReady='true';
  const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){const cy=cyViews.get(entry.target.id);if(cy){cy.resize();}}},{threshold:.2});
  for(const id of ['protein-network','association-network','go-tree-view'])observer.observe($(id));
})();
