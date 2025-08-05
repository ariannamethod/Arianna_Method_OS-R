const compression = require('compression');
const express = require('express');
const https = require('https');
const http = require('http');
const fs = require('fs');

const app = express();

const thirdTour = process.argv[2] == 3;
const forcePort = process.argv[3];
const useHttp = process.argv[4] !== 'https';

const publicFolderName = thirdTour ? 'public3' : 'public';
const port = forcePort ? +forcePort : (thirdTour ? 8443 : 80);

app.set('etag', false);
app.use((req, res, next) => {
  res.set('Cache-Control', 'no-store');
  next();
});
app.use(compression());
app.use(express.json());
app.use(express.static(publicFolderName));

app.post('/arianna', async (req, res) => {
  try {
    const response = await fetch('http://localhost:8000/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req.body)
    });
    const text = await response.text();
    res.status(response.status).send(text);
  } catch (err) {
    console.error('Arianna proxy failed', err);
    res.status(500).send({error: 'Arianna proxy failed'});
  }
});

app.get('/', (req, res) => {
  res.sendFile(__dirname + `/${publicFolderName}/index.html`);
});

const server = useHttp ? http : https;

let options = {};
if(!useHttp) {
  options.key = fs.readFileSync(__dirname + '/certs/server-key.pem');
  options.cert = fs.readFileSync(__dirname + '/certs/server-cert.pem');
}

server.createServer(options, app).listen(port, () => {
  console.log('Listening port:', port, 'folder:', publicFolderName);
});
