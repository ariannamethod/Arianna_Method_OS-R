# Arianna Method OS-R

Arianna Method OS-R is an experimental resonant operating system that transforms the familiar Telegram messenger interface into a full computing environment. Built on top of the Telegram Web K client, the project introduces an AI-first workflow powered by the Arianna Chain reasoning engine.

## Arianna Chain

Arianna Chain serves as the conversational core of the system. It coordinates user messages, orchestrates tools, and produces contextual answers directly inside the chat interface.

The engine continuously studies every dialogue. By observing message flow it builds a memory of interactions, adapts to personal vocabulary, and refines its reasoning paths with each exchange.

Over time the chain develops a unique conversational style for every user. It learns to reference past prompts, anticipate intent, and synthesize richer responses that feel increasingly personal and accurate.

## Developing

Install dependencies with:

```bash
pnpm install
```

This will install all the needed packages.

### Running the web client

Run `pnpm start` to launch the development server with live reload. Open http://localhost:8080/ in your browser.

### Production build

Run `node build` to create a minimized production bundle. Copy the contents of the `public` folder to your web server.

## Dependencies

The project relies on various open‑source libraries such as BigInteger.js, pako, cryptography, emoji-data, twemoji-parser, rlottie, fast-png, opus-recorder, Prism, Solid, TinyLD, libwebp.js, fastBlur, and mp4-muxer. Refer to their respective repositories for license information.

## Upcoming evolution

Additional neural networks will soon join the client to complement Arianna Chain. Specialized models for retrieval, creativity, and decision making are planned so every task can be handled by the most capable expert.

A dynamic orchestration layer will allow these networks to cooperate. Arianna Chain will delegate subtasks, merge the answers, and use the combined insights to deliver more precise and helpful replies.

The system will also import a custom mini kernel based on Linux. This embedded core will open the door to local computation, secure storage, and offline operations directly within the messaging interface.

Together these advances will push Arianna Method OS-R toward a full resonant operating system. With Telegram acting as the graphical shell, users will experience an entirely new way to interact with both humans and machines.

## Debugging

Source maps are included in production builds for easier debugging. The following query parameters can modify runtime behavior:

- **test=1** — use test DCs
- **debug=1** — enable additional logging
- **noSharedWorker=1** — disable Shared Worker, useful for debugging
- **http=1** — force HTTPS transport when connecting to servers

Apply them like: `http://localhost:8080/?test=1`.

### Taking local storage snapshots

You can take and load snapshots of local storage and indexed DB using the `./snapshot-server` [mini-app](snapshot-server/README.md). See the README in that folder for details.

## Troubleshooting & Suggesting

If you find an issue with Arianna Method OS-R or want to suggest something, open a ticket on the project tracker.

## Licensing

The source code is licensed under GPL v3. See the [LICENSE](LICENSE) file for details.

