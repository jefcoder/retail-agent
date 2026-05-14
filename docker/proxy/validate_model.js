// Inference proxy: validates outgoing requests against the per-provider
// allowlist before forwarding to either Chutes or OpenRouter, dispatched
// by the bearer token's prefix.
//
// Allowlists are the union of model IDs on each side of model_pairs.json
// (loaded once per worker at module init). No external Backend fetch.
//
// Provider dispatch:
//   - Bearer token starts with "sk-or-" → OpenRouter (allowlist enforced)
//   - Any other token shape (e.g. cak_*) → Chutes (allowlist enforced)
//
// Cross-provider model-name rewriting: agents can use any model identifier
// listed in model_pairs.json regardless of which provider funds the run. If
// the request `model` matches the inactive provider's side of a pair we
// rewrite the body's `model` field to the active provider's side before
// allowlist validation, so the same agent code works on either provider.
// Models with no pair entry pass through unchanged and hit the allowlist
// check as before.

import fs from "fs";

function detectProvider(r) {
  var auth = r.headersIn["Authorization"] || "";
  if (auth.indexOf("Bearer sk-or-") === 0) {
    return "openrouter";
  }
  return "chutes";
}

function pushUnique(arr, name) {
  if (name && arr.indexOf(name) === -1) {
    arr.push(name);
  }
}

// Lookup tables for both directions, populated once at module init. Each
// nginx worker loads the file independently; reload (`nginx -s reload`)
// picks up edits.
var MODEL_PAIRS_PATH = "/etc/nginx/model_pairs.json";
var _pairsByChutes = {};
var _pairsByOpenrouter = {};
var _allowlistChutes = [];
var _allowlistOpenrouter = [];
try {
  var _pairsDoc = JSON.parse(fs.readFileSync(MODEL_PAIRS_PATH, "utf8"));
  for (var i = 0; i < _pairsDoc.pairs.length; i++) {
    var p = _pairsDoc.pairs[i];
    _pairsByChutes[p.chutes] = p.openrouter;
    _pairsByOpenrouter[p.openrouter] = p.chutes;
    pushUnique(_allowlistChutes, p.chutes);
    pushUnique(_allowlistOpenrouter, p.openrouter);
  }
} catch (e) {
  // Don't crash the worker — surface the error so it's noticeable.
  ngx.log(ngx.ERR, "model_pairs load failed: " + e.message);
}

// Returns the request model rewritten for `activeProvider`, or null if no
// rewrite is needed (already on the active side, or unknown — let allowlist
// validation handle it).
function rewriteModelFor(activeProvider, requested) {
  if (activeProvider === "chutes") {
    if (_pairsByChutes[requested] !== undefined) return null;
    if (_pairsByOpenrouter[requested] !== undefined) return _pairsByOpenrouter[requested];
  } else if (activeProvider === "openrouter") {
    if (_pairsByOpenrouter[requested] !== undefined) return null;
    if (_pairsByChutes[requested] !== undefined) return _pairsByChutes[requested];
  }
  return null;
}

function allowlistFor(provider) {
  return provider === "openrouter" ? _allowlistOpenrouter : _allowlistChutes;
}

function validate(r) {
  var provider = detectProvider(r);
  var upstreamLocation = provider === "openrouter" ? "/_openrouter_proxy/" : "/_chutes_proxy/";

  if (r.method !== "POST") {
    var passUri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
    r.subrequest(passUri, { method: r.method, args: r.variables.args || "" }, function (reply) {
      for (var h in reply.headersOut) {
        r.headersOut[h] = reply.headersOut[h];
      }
      r.return(reply.status, reply.responseText);
    });
    return;
  }

  var body = r.requestText;

  if (!body) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Missing or unreadable request body" }));
    return;
  }

  var parsed;
  try {
    parsed = JSON.parse(body);
  } catch (e) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Invalid JSON in request body" }));
    return;
  }

  if (!parsed.model) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Missing 'model' field in request body" }));
    return;
  }

  if (parsed.stream === true) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(400, JSON.stringify({ error: "Streaming is not supported through the proxy" }));
    return;
  }

  var rewritten = rewriteModelFor(provider, parsed.model);
  var forwardBody = body;
  if (rewritten !== null) {
    parsed.model = rewritten;
    forwardBody = JSON.stringify(parsed);
  }

  var allowed = allowlistFor(provider);
  if (!allowed || allowed.length === 0) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(503, JSON.stringify({ error: "Inference allowlist unavailable" }));
    return;
  }

  if (allowed.indexOf(parsed.model) === -1) {
    r.error("Model not allowed for " + provider + ": " + parsed.model);
    r.headersOut["Content-Type"] = "application/json";
    r.return(
      403,
      JSON.stringify({
        error: "Model '" + parsed.model + "' is not allowed for provider " + provider,
        allowed_models: allowed,
      })
    );
    return;
  }

  var uri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
  r.subrequest(
    uri,
    { method: "POST", body: forwardBody, args: r.variables.args || "" },
    function (reply) {
      for (var h in reply.headersOut) {
        r.headersOut[h] = reply.headersOut[h];
      }
      r.return(reply.status, reply.responseText);
    }
  );
}

export default { validate };
