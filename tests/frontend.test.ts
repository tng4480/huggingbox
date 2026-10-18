import {test} from 'node:test';
import assert from 'node:assert/strict';
import {useAppStore} from '../src/stores/appStore';
import {useExecution} from '../src/hooks/useExecution';
import {getCompatibility} from '../src/utils/ramEstimation';

const g=globalThis as any;
g.events=new Map();g.approve=true;
let calls: {cmd:string;args:any}[]=[];
let custom:(cmd:string,args:any)=>any=()=>undefined;
let fingerprint='v1';
g.bridge=async(cmd:string,args:any)=>{
 calls.push({cmd,args});const value=custom(cmd,args);if(value!==undefined)return value;
 if(cmd==='environment_fingerprint')return fingerprint;
 if(cmd==='list_model_environments'||cmd==='check_packages'||cmd==='list_downloaded_models')return [];
 if(cmd==='is_model_downloaded')return true;
 if(cmd==='probe_model_dependencies')return {missingPackages:[],requiredPackages:[]};
 return null;
};
const api=useExecution();
const options=(name:string)=>({modelId:`test/${name}`,storagePath:'/models',runtimeSourceCode:'# RUNTIME: transformers_llm\nprint(1)',scriptRelativePath:'huggingbox_main.py',userInput:'hello'});
const done=async(code=0)=>{await new Promise(setImmediate);const id=useAppStore.getState().activeJobId;g.events.get('execution-done')({payload:{job_id:id,exit_code:code}})};
function reset(){calls=[];custom=()=>undefined;fingerprint='v1';g.approve=true;useAppStore.setState({activeJobId:null,executionState:'idle',executionError:null});}

test('failed first setup remains repairable on retry',async()=>{
 reset();let failures=1;
 custom=(cmd,args)=>{if(cmd==='check_packages'&&args.packages.includes('huggingface_hub'))return ['huggingface_hub'];if(cmd==='install_packages'&&failures--)throw Error('temporary network failure');};
 await api.runCode(options('retry'));assert.equal(useAppStore.getState().executionState,'error');
 calls=[];await api.runCode(options('retry'));assert(calls.some(c=>c.cmd==='install_packages'));assert(calls.some(c=>c.cmd==='run_python_code'));await done();
});
test('stop during preparation prevents launch and waits for settlement',async()=>{
 reset();let release:any,entered:any;const ready=new Promise(r=>entered=r);let once=true;
 custom=cmd=>{if(cmd==='check_packages'&&once){once=false;entered();return new Promise(r=>release=r)}};
 const run=api.runCode(options('cancel'));await ready;const stop=api.cancelExecution();
 assert.equal(useAppStore.getState().executionState,'stopping');release([]);await Promise.all([run,stop]);
 assert(!calls.some(c=>c.cmd==='run_python_code'));assert.equal(useAppStore.getState().executionState,'cancelled');
});
test('repeated runs reuse verified setup; changed environments trigger repair',async()=>{
 reset();await api.runCode(options('warm'));await done();calls=[];
 await api.runCode(options('warm'));assert(!calls.some(c=>c.cmd==='probe_model_dependencies'||c.cmd==='install_packages'));await done();
 fingerprint='v2';calls=[];await api.runCode(options('warm'));assert(calls.some(c=>c.cmd==='probe_model_dependencies'));await done();
});
test('navigation preserves active job and drafts, and stale events are ignored',async()=>{
 reset();const store=useAppStore.getState();store.navigateToModel('test/A');store.setGeneratedCode('print("A")');store.setWorkspaceDraft('test/A',{inputValue:'keep me',seed:'123'});
 await api.runCode(options('A'));const id=useAppStore.getState().activeJobId;store.navigateToModel('test/B');
 assert.equal(useAppStore.getState().activeJobId,id);assert.equal(useAppStore.getState().activeExecutionModelId,'test/A');
 g.events.get('execution-done')({payload:{job_id:'old',exit_code:0}});assert.equal(useAppStore.getState().activeJobId,id);
 store.navigateToModel('test/A');assert.equal(useAppStore.getState().generatedCode,'print("A")');assert.equal(useAppStore.getState().workspaceDrafts['test/A'].inputValue,'keep me');await done();
});
test('late model metadata cannot overwrite the selected model',()=>{
 reset();const store=useAppStore.getState();store.navigateToModel('test/B');const b={id:'test/B',modelId:'test/B'} as any;store.setModelDetail(b);store.setModelDetail({id:'test/A',modelId:'test/A'} as any);assert.equal(useAppStore.getState().modelDetail?.id,'test/B');
});
test('empty scripts cannot launch or trigger preparation',async()=>{reset();await api.runCode({...options('empty'),runtimeSourceCode:''});assert.equal(calls.length,0);assert.match(useAppStore.getState().executionError!,/empty/)});
test('history records the run input, is bounded, and survives view changes',()=>{
 reset();for(let i=0;i<25;i++){useAppStore.setState({activeJobId:`history-${i}`,activeExecutionModelId:'test/A',activeRunInput:`input ${i}`,executionOutput:'result'});useAppStore.getState().finishRun('completed');}
 useAppStore.getState().navigateToModel('test/B');assert.equal(useAppStore.getState().runHistory.length,20);assert.equal(useAppStore.getState().runHistory[0].input,'input 24');
});
test('unknown diffusion weights never receive a positive RAM-fit estimate',()=>{
 const model={id:'test/image',modelId:'test/image',pipeline_tag:'text-to-image',siblings:[],tags:[],downloads:0,author:'test',lastModified:''};assert.equal(getCompatibility(model,16*1024**3),'unknown');
});


test('CLI generation and repair route without API keys and redact stored secrets', async()=>{
 const {generateCodeWithAssistant,fixCodeWithAssistant}=await import('../src/services/hfAutoRunner');
 for(const provider of ['codex-cli','claude-code'] as const){
  reset();const settings={...useAppStore.getState().settings,codeGenerationProvider:provider,hfToken:'secret-hf',claudeApiKey:'',codexCliPath:'/bin/codex',claudeCodePath:'/bin/claude'};
  custom=(cmd)=>cmd==='generate_code_with_cli'?{text:JSON.stringify({code:'print(1)',analysis:'ok',dependencies:[]})}:undefined;
  const model={id:'test/model',modelId:'test/model',readme:'Model documentation secret-hf',tags:[],siblings:[]} as any;
  const system={totalRam:1,availableRam:1,gpuName:null,os:'test'} as any;
  await generateCodeWithAssistant(model,settings,system);
  await fixCodeWithAssistant(model,settings,system,{currentCode:'print(1)',stderrOutput:'secret-hf',installedModules:''});
  assert.equal(calls.length,2);for(const call of calls){assert.equal(call.cmd,'generate_code_with_cli');assert.equal(call.args.provider,provider);assert(!call.args.prompt.includes('secret-hf'));assert(!('apiKey' in call.args));}
 }
});
