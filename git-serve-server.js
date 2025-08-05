// @ts-check

const https = require('https');
const http = require('http');
const fs = require('fs');

const express = require("express");

const gitstatic = (() => {
  const child = require("child_process"),
    mime = require("mime"),
    path = require("path");

  const shaRe = /^[0-9a-f]{40}$/,
    emailRe = /^<.*@.*>$/;

  function readBlob(repository, revision, file, callback) {
    const git = child.spawn("git", ["cat-file", "blob", revision + ":" + file], { cwd: repository }),
      data = [];
    let exit;

    git.stdout.on("data", function(chunk) {
      data.push(chunk);
    });

    git.on("exit", function(code) {
      exit = code;
    });

    git.on("close", function() {
      if (exit > 0) return callback(error(exit));
      callback(null, Buffer.concat(data));
    });

    git.stdin.end();
  }

  function getBranches(repository, callback) {
    child.exec("git branch -l", { cwd: repository }, function(error, stdout) {
      if (error) return callback(error);
      callback(null, stdout.split(/\n/).slice(0, -1).map(function(s) { return s.slice(2); }));
    });
  }

  function getSha(repository, revision, callback) {
    child.exec("git rev-parse '" + revision.replace(/'/g, "'\\''") + "'", {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      callback(null, stdout.trim());
    });
  }

  function getBranchCommits(repository, callback) {
    child.exec("git for-each-ref refs/heads/ --sort=-authordate --format='%(objectname)\t%(refname:short)\t%(authordate:iso8601)\t%(authoremail)'", {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      callback(null, stdout.split("\n").map(function(line) {
        var fields = line.split("\t"),
          sha = fields[0],
          ref = fields[1],
          date = new Date(fields[2]),
          author = fields[3];
        if (!shaRe.test(sha) || !date || !emailRe.test(author)) return;
        return {
          sha: sha,
          ref: ref,
          date: date,
          author: author.substring(1, author.length - 1)
        };
      }).filter(function(commit) {
        return commit;
      }));
    });
  }

  function getCommit(repository, revision, callback) {
    if (arguments.length < 3) callback = revision, revision = null;
    child.exec(shaRe.test(revision)
        ? "git log -1 --date=iso " + revision + " --format='%H\n%ad'"
        : "git for-each-ref --count 1 --sort=-authordate 'refs/heads/" + (revision ? revision.replace(/'/g, "'\\''") : "") + "' --format='%(objectname)\n%(authordate:iso8601)'", {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      var lines = stdout.split("\n"),
        sha = lines[0],
        date = new Date(lines[1]);
      if (!shaRe.test(sha) || !date) return void callback(new Error("unable to get commit"));
      callback(null, {
        sha: sha,
        date: date
      });
    });
  }

  function getRelatedCommits(repository, branch, sha, callback) {
    if (!shaRe.test(sha)) return callback(new Error("invalid SHA: " + sha));
    child.exec("git log --format='%H' '" + branch.replace(/'/g, "'\\''") + "' | grep -C1 " + sha, {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      var shas = stdout.split(/\n/),
        i = shas.indexOf(sha);

      callback(null, {
        previous: shas[i + 1],
        next: shas[i - 1]
      });
    });
  }

  function listCommits(repository, sha1, sha2, callback) {
    if (!shaRe.test(sha1)) return callback(new Error("invalid SHA: " + sha1));
    if (!shaRe.test(sha2)) return callback(new Error("invalid SHA: " + sha2));
    child.exec("git log --format='%H\t%ad' " + sha1 + ".." + sha2, {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      callback(null, stdout.split(/\n/).slice(0, -1).map(function(commit) {
        var fields = commit.split(/\t/);
        return {
          sha: fields[0],
          date: new Date(fields[1])
        };
      }));
    });
  }

  function listAllCommits(repository, callback) {
    child.exec("git log --branches --format='%H\t%ad\t%an\t%s'", {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      callback(null, stdout.split("\n").slice(0, -1).map(function(commit) {
        var fields = commit.split("\t");
        return {
          sha: fields[0],
          date: new Date(fields[1]),
          author: fields[2],
          subject: fields[3]
        };
      }));
    });
  }

  function listTree(repository, revision, callback) {
    child.exec("git ls-tree -r " + revision, {cwd: repository}, function(error, stdout) {
      if (error) return callback(error);
      callback(null, stdout.split(/\n/).slice(0, -1).map(function(commit) {
        var fields = commit.split(/\t/);
        return {
          sha: fields[0].split(/\s/)[2],
          name: fields[1]
        };
      }));
    });
  }

  function route() {
    var repository = defaultRepository,
      revision = defaultRevision,
      file = defaultFile,
      type = defaultType;

    function route(request, response) {
      var repository_,
        revision_,
        file_;

      // @ts-ignore
      if ((repository_ = repository(request.url)) == null
          || (revision_ = revision(request.url)) == null
          || (file_ = file(request.url)) == null) return serveNotFound();

      readBlob(repository_, revision_, file_, function(error, data) {
        if (error) return error.code === 128 ? serveNotFound() : serveError(error);
        response.writeHead(200, {
          "Content-Type": type(file_),
          "Cache-Control": "public, max-age=300"
        });
        response.end(data);
      });

      function serveError(error) {
        response.writeHead(500, {"Content-Type": "text/plain"});
        response.end(error + "");
      }

      function serveNotFound() {
        response.writeHead(404, {"Content-Type": "text/plain"});
        response.end("File not found.");
      }
    }

    route.repository = function(_) {
      if (!arguments.length) return repository;
      repository = functor(_);
      return route;
    };

    route.sha =
    route.revision = function(_) {
      if (!arguments.length) return revision;
      revision = functor(_);
      return route;
    };

    route.file = function(_) {
      if (!arguments.length) return file;
      file = functor(_);
      return route;
    };

    route.type = function(_) {
      if (!arguments.length) return type;
      type = functor(_);
      return route;
    };

    return route;
  }

  function functor(_) {
    return typeof _ === "function" ? _ : function() { return _; };
  }

  function defaultRepository() {
    return path.join(__dirname, "repository");
  }

  function defaultRevision(url) {
    return decodeURIComponent(url.substring(1, url.indexOf("/", 1)));
  }

  function defaultFile(url) {
    url = url.substring(url.indexOf("/", 1) + 1);
    const pathIdx = url.indexOf('?');
    if (pathIdx !== -1) {
      url = url.slice(0, pathIdx);
    }

    return decodeURIComponent(url);
  }

  function defaultType(file) {
    var type = mime.getType(file) || "text/plain";
    return text(type) ? type + "; charset=utf-8" : type;
  }

  function text(type) {
    return /^(text\/)|(application\/(javascript|json)|image\/svg$)/.test(type);
  }

  function error(code) {
    var e = new Error;
    // @ts-ignore
    e.code = code;
    return e;
  }

  return {readBlob, getBranches, getSha, getBranchCommits, getCommit, getRelatedCommits, listCommits, listAllCommits, listTree, route};
})();

const repository = '.git';

const app = express();
app.get(/^\/.+/, gitstatic.route().repository(repository));
app.get(/\//, (req, res) => {
  gitstatic.listAllCommits(repository, (err, commits) => {
    console.log(err, commits);

    res.send(
      commits.map((commit) => {
        return `<a href="/${commit.sha}/public/index.html" target="_blank"><span style="font-family: monospace;">${commit.sha.slice(0, 7)} - ${commit.date.toISOString()}</span></a> - <a href="https://github.com/morethanwords/tweb/commit/${commit.sha}" target="_blank">${commit.subject}</a><br>`;
      }).join('')
    );
  });
});

const { networkInterfaces } = require('os');
const nets = networkInterfaces();
const results = {};

for(const name of Object.keys(nets)) {
  for(const net of nets[name]) {
    // Skip over non-IPv4 and internal (i.e. 127.0.0.1) addresses
    if(net.family === 'IPv4' && !net.internal) {
      if(!results[name]) {
        results[name] = [];
      }
      results[name].push(net.address);
    }
  }
}

const useHttp = false;
const transport = useHttp ? http : https;
let options = {};
if(!useHttp) {
  options.key = fs.readFileSync(__dirname + '/certs/server-key.pem');
  options.cert = fs.readFileSync(__dirname + '/certs/server-cert.pem');
}

console.log(results);

const port = 3000;
const protocol = useHttp ? 'http' : 'https';
console.log('Listening port:', port);
function createServer(host) {
  const server = transport.createServer(options, app);
  server.listen(port, host, () => {
    console.log('Host:', `${protocol}://${host || 'localhost'}:${port}/`);
  });

  server.on('error', (e) => {
    // @ts-ignore
    if(e.code === 'EADDRINUSE') {
      console.log('Address in use:', host);
      server.close();
    }
  });
}

for(const name in results) {
  const ips = results[name];
  for(const ip of ips) {
    createServer(ip);
  }
}

createServer();
