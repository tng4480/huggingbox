import {build} from 'esbuild';
import {spawnSync} from 'node:child_process';
import {mkdirSync} from 'node:fs';
mkdirSync('work/tests',{recursive:true});
await build({entryPoints:['tests/frontend.test.ts'],outfile:'work/tests/frontend.test.mjs',bundle:true,platform:'node',format:'esm',banner:{js:`globalThis.localStorage={data:new Map(),getItem(k){return this.data.get(k)??null},setItem(k,v){this.data.set(k,v)},removeItem(k){this.data.delete(k)}};globalThis.window=globalThis;`},plugins:[{name:'native-boundary',setup(b){
 for(const [pattern,path] of [[/^react$/,'react'],[/^@tauri-apps\/api\/core$/,'core'],[/^@tauri-apps\/api\/event$/,'event'],[/^@tauri-apps\/plugin-dialog$/,'dialog']]) b.onResolve({filter:pattern},()=>({path,namespace:'mock'}));
 b.onLoad({filter:/.*/,namespace:'mock'},a=>({loader:'js',contents:{
 react:'export const useCallback=f=>f;export const useEffect=f=>{f()};export const useRef=x=>({current:x});export const useSyncExternalStore=()=>{};export const useDebugValue=()=>{};export default {useCallback,useEffect,useRef,useSyncExternalStore,useDebugValue};',
 core:'export const invoke=(...args)=>globalThis.bridge(...args);',event:'export const listen=async(n,fn)=>{globalThis.events.set(n,fn);return ()=>{}};',dialog:'export const confirm=async()=>globalThis.approve;export const message=async()=>{};'
 }[a.path]}));
}}]});
const result=spawnSync(process.execPath,['--test','work/tests/frontend.test.mjs'],{stdio:'inherit'});process.exit(result.status??1);
