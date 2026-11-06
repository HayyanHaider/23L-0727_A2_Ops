// Execute a Postman collection with Newman; redact the key before writing evidence.
const fs = require('node:fs');
const newman = require('newman');
const key = process.env.INFERENCE_API_KEY;
const base = process.env.API_URL;
if (!key || !base || !base.startsWith('https://')) throw new Error('Set HTTPS API_URL and INFERENCE_API_KEY');
const collection = JSON.parse(fs.readFileSync('postman/VisionGuard.postman_collection.json','utf8'));
newman.run({collection,envVar:[{key:'baseUrl',value:base},{key:'apiKey',value:key}],reporters:['cli']},(err,summary)=>{
  fs.mkdirSync('outputs',{recursive:true});
  const result={project:'23L-0727_A2_Ops',timestamp:new Date().toISOString(),baseUrl:base,
    stats:summary?.run?.stats,failures:(summary?.run?.failures||[]).map(x=>({source:x.source?.name,message:x.error?.message})),
    requests:(summary?.run?.executions||[]).map(x=>({name:x.item.name,status:x.response?.code,
      response:x.response?.stream?.toString(),assertions:x.assertions}))};
  fs.writeFileSync('outputs/postman-results.json',JSON.stringify(result,null,2).split(key).join('[REDACTED]'));
  if(err || result.failures.length)process.exitCode=1;
});
