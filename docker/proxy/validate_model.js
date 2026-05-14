// Inference proxy: validates /inference/* POST bodies against OpenRouter
// allowlist in model_pairs.json, then forwards to OpenRouter only.
//
// Requires Authorization: Bearer sk-or-... (OpenRouter API key shape).
// No external Backend fetch.

import fs from "fs";

function pushUnique(arr, name) {
  if (name && arr.indexOf(name) === -1) {
    arr.push(name);
  }
}

var MODEL_PAIRS_PATH = "/etc/nginx/model_pairs.json";
var _allowlist = [];
try {
  var doc = JSON.parse(fs.readFileSync(MODEL_PAIRS_PATH, "utf8"));
  var raw = doc.allowed_models || [];
  for (var i = 0; i < raw.length; i++) {
    pushUnique(_allowlist, raw[i]);
  }
} catch (e) {
  ngx.log(ngx.ERR, "model_pairs load failed: " + e.message);
}

function isOpenRouterAuth(r) {
  var auth = r.headersIn["Authorization"] || "";
  return auth.indexOf("Bearer sk-or-") === 0;
}

function validate(r) {
  if (!isOpenRouterAuth(r)) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(
      403,
      JSON.stringify({
        error:
          "OpenRouter-only proxy: use Authorization Bearer sk-or-... (OpenRouter API key).",
      })
    );
    return;
  }

  var upstreamLocation = "/_openrouter_proxy/";

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

  var allowed = _allowlist;
  if (!allowed || allowed.length === 0) {
    r.headersOut["Content-Type"] = "application/json";
    r.return(503, JSON.stringify({ error: "Inference allowlist unavailable" }));
    return;
  }

  if (allowed.indexOf(parsed.model) === -1) {
    r.error("Model not allowed for openrouter: " + parsed.model);
    r.headersOut["Content-Type"] = "application/json";
    r.return(
      403,
      JSON.stringify({
        error: "Model '" + parsed.model + "' is not allowed",
        allowed_models: allowed,
      })
    );
    return;
  }

  var uri = upstreamLocation + r.uri.replace(/^\/inference\//, "");
  r.subrequest(
    uri,
    { method: "POST", body: body, args: r.variables.args || "" },
    function (reply) {
      for (var h in reply.headersOut) {
        r.headersOut[h] = reply.headersOut[h];
      }
      r.return(reply.status, reply.responseText);
    }
  );
}

export default { validate };
