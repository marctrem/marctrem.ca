<script lang="ts">
	import { onMount } from 'svelte';
	import { loadModel, evaluate, formToFeatures } from '$lib/titanic/fly';
	import type { Prediction } from '$lib/titanic/fly';
	import { createBrain } from '$lib/titanic/brain';
	import type { BrainView } from '$lib/titanic/brain';

	const BASE = '/titanic-fly-classifier';

	type XYZ = [number, number, number];
	type CoordLayer = { size: number; xyz: Array<XYZ | null> };
	type Coords = {
		layers: CoordLayer[];
		sensors: { size: number; xyz: Array<XYZ | null> };
		readout_idx: number[];
		readout_layer: number;
	};

	let pclass = $state('3');
	let sex = $state('male');
	let title = $state('Mr');
	let embarked = $state('S');
	let age = $state('28');
	let fare = $state('15');
	let sibsp = $state('0');
	let parch = $state('0');
	let deck = $state('');

	let ready = $state(false);
	let prob = $state<number | null>(null);
	let inferMs = $state<number | null>(null);
	let renderMs = $state<number | null>(null);
	let avgMs = $state<number | null>(null);

	let cardEl = $state<HTMLCanvasElement>();
	let retinaEl = $state<HTMLCanvasElement>();
	let brainEl = $state<HTMLCanvasElement>();
	let brain: BrainView | null = null;
	let coords: Coords | null = null;
	const samples: number[] = [];

	function colorSigned(v: number): [number, number, number] {
		const x = Math.max(-1, Math.min(1, v));
		if (x >= 0) {
			const t = x;
			return [0.35 + 0.65 * t, 0.35 + 0.35 * t, 0.35 - 0.28 * t];
		}
		const t = -x;
		return [0.35 - 0.22 * t, 0.35 + 0.2 * t, 0.35 + 0.65 * t];
	}

	function drawGray(canvas: HTMLCanvasElement, arr: Float32Array, w: number, h: number) {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const im = ctx.createImageData(w, h);
		for (let i = 0; i < w * h; i++) {
			const v = Math.max(0, Math.min(255, Math.round(arr[i] * 255)));
			im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v;
			im.data[i * 4 + 3] = 255;
		}
		ctx.putImageData(im, 0, 0);
	}

	function drawSensors(canvas: HTMLCanvasElement, sensors: Float32Array) {
		const GW = 8;
		const GH = 38;
		const W = 192;
		const H = 64;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const im = ctx.createImageData(W, H);
		for (let py = 0; py < H; py++)
			for (let px = 0; px < W; px++) {
				const cy = Math.floor((py / H) * GH);
				const cx = Math.floor((px / W) * GW);
				const v = Math.max(0, Math.min(255, Math.round(sensors[cx * GH + cy] * 255)));
				const o = (py * W + px) * 4;
				im.data[o] = im.data[o + 1] = im.data[o + 2] = v;
				im.data[o + 3] = 255;
			}
		ctx.putImageData(im, 0, 0);
	}

	function renderBrain(res: Prediction) {
		if (!brain || !coords) return;
		const pts: number[] = [];
		const push = (xyz: XYZ | null, col: [number, number, number], size: number) => {
			if (!xyz) return;
			pts.push(xyz[0], xyz[1], xyz[2], col[0], col[1], col[2], size);
		};
		for (let i = 0; i < coords.sensors.xyz.length; i++)
			push(coords.sensors.xyz[i], colorSigned(res.sensors[i]), 3.5);
		const ri = new Set(coords.readout_idx);
		for (let k = 0; k < coords.layers.length; k++) {
			const L = coords.layers[k];
			const acts = res.layers[k] ?? new Float32Array(L.xyz.length);
			for (let i = 0; i < L.xyz.length; i++) {
				const readout = k === coords.readout_layer && ri.has(i);
				push(L.xyz[i], colorSigned(acts[i] ?? 0), readout ? 7 : 3.5);
			}
		}
		brain.render(new Float32Array(pts));
	}

	function run() {
		if (!ready || !cardEl || !retinaEl) return;
		const t0 = performance.now();
		const res = evaluate(
			formToFeatures({ pclass, sex, title, embarked, age, fare, sibsp, parch, deck })
		);
		const t1 = performance.now();
		drawGray(cardEl, res.img, 192, 64);
		drawSensors(retinaEl, res.sensors);
		renderBrain(res);
		const t2 = performance.now();
		prob = res.p;
		inferMs = t1 - t0;
		renderMs = t2 - t1;
		samples.push(inferMs);
		if (samples.length > 24) samples.shift();
		avgMs = samples.reduce((a, b) => a + b, 0) / samples.length;
	}

	onMount(async () => {
		await loadModel(BASE);
		coords = (await (await fetch(`${BASE}/brain_coords.json`)).json()) as Coords;
		if (brainEl) {
			brain = createBrain(brainEl);
			brain?.init();
		}
		ready = true;
		run();
	});
</script>

<svelte:head>
	<title>Titanic fly classifier</title>
	<meta
		name="description"
		content="A fruit-fly connectome network that predicts Titanic survival: a passenger card is tiled onto the fly's optic neurons and processed through frozen optic-lobe wiring."
	/>
</svelte:head>

<main>
	<h1>Titanic fly classifier</h1>
	<p class="tagline">
		A passenger is rendered as a 192&times;64 card, tiled retinotopically onto the fly's 302 optic
		sensory neurons, and processed through the frozen
		<code>optic lobe &rarr; central &rarr; descending</code> wiring. The motor readout raises its
		arm for <em>survive</em>, lowers it for <em>not</em>. Change any field &mdash; it evaluates
		live.
	</p>

	<div class="wrap">
		<form onsubmit={(e) => e.preventDefault()}>
			<div>
				<label for="pclass">Passenger class</label>
				<select id="pclass" bind:value={pclass} onchange={run}>
					<option value="3">3rd</option>
					<option value="2">2nd</option>
					<option value="1">1st</option>
				</select>
			</div>
			<div>
				<label for="sex">Sex</label>
				<select id="sex" bind:value={sex} onchange={run}>
					<option value="male">male</option>
					<option value="female">female</option>
				</select>
			</div>
			<div>
				<label for="title">Title</label>
				<select id="title" bind:value={title} onchange={run}>
					<option value="Mr">Mr</option>
					<option value="Mrs">Mrs</option>
					<option value="Miss">Miss</option>
					<option value="Master">Master</option>
					<option value="Dr">Dr</option>
					<option value="Rev">Rev</option>
					<option value="Rare">Rare</option>
				</select>
			</div>
			<div>
				<label for="embarked">Embarked</label>
				<select id="embarked" bind:value={embarked} onchange={run}>
					<option value="S">Southampton</option>
					<option value="C">Cherbourg</option>
					<option value="Q">Queenstown</option>
				</select>
			</div>
			<div>
				<label for="age">Age</label>
				<input id="age" type="number" min="0" max="80" bind:value={age} oninput={run} />
			</div>
			<div>
				<label for="fare">Fare</label>
				<input id="fare" type="number" min="0" step="0.1" bind:value={fare} oninput={run} />
			</div>
			<div>
				<label for="sibsp">Siblings / spouses</label>
				<input id="sibsp" type="number" min="0" max="8" bind:value={sibsp} oninput={run} />
			</div>
			<div>
				<label for="parch">Parents / children</label>
				<input id="parch" type="number" min="0" max="8" bind:value={parch} oninput={run} />
			</div>
			<div>
				<label for="deck">Cabin deck</label>
				<select id="deck" bind:value={deck} onchange={run}>
					<option value="">unknown</option>
					<option value="A">A</option>
					<option value="B">B</option>
					<option value="C">C</option>
					<option value="D">D</option>
					<option value="E">E</option>
					<option value="F">F</option>
					<option value="G">G</option>
				</select>
			</div>
		</form>

		<div class="view">
			<div class="row">
				<div>
					<canvas bind:this={cardEl} width="192" height="64"></canvas>
					<div class="cap">Passenger card (input to the eye)</div>
				</div>
				<div>
					<canvas bind:this={retinaEl} width="192" height="64"></canvas>
					<div class="cap">Optic-sensor activations (retinotopic tiling)</div>
				</div>
			</div>

			<canvas bind:this={brainEl} width="620" height="330" class="brain"></canvas>
			<div class="cap">
				Brain-shaped activation map &mdash; each neuron of the model's sub-circuit drawn at its real
				FAFB position, coloured by activation. <span class="sw blue"></span> inhibited,
				<span class="sw orange"></span> excited. Bright ring = descending/motor readout.
				<strong>Drag to rotate, scroll to zoom.</strong>
			</div>

			{#if prob !== null}
				<div class="result" class:up={prob >= 0.5} class:down={prob < 0.5}>
					<div class="arm">{prob >= 0.5 ? '\u2191' : '\u2193'}</div>
					<div>
						<strong>{prob >= 0.5 ? 'Survives' : 'Does not survive'}</strong>
						<div class="prob">
							motor readout probability of survival: {(prob * 100).toFixed(1)}%
						</div>
					</div>
				</div>
			{:else}
				<div class="result">Loading model&hellip;</div>
			{/if}

			<div class="lat">
				inference {inferMs === null ? '\u2014' : inferMs.toFixed(1)} ms (render {(
					renderMs ?? 0
				).toFixed(1)} ms) &middot; rolling avg
				{avgMs === null ? '\u2014' : avgMs.toFixed(1)} ms &middot; forward: raster &rarr; 302 sensors
				&rarr; 768 &rarr; 768 &rarr; 93 &rarr; 127 &rarr; motor arm
			</div>
		</div>
	</div>
</main>

<style>
	:global(body) {
		max-width: 1100px;
		margin: 0 auto;
		padding: 28px;
	}

	h1 {
		font-size: 22px;
		margin: 0 0 4px;
	}

	.tagline {
		color: var(--muted);
		margin: 0 0 22px;
	}

	code {
		font-size: 0.92em;
	}

	.wrap {
		display: grid;
		grid-template-columns: 380px 1fr;
		gap: 28px;
		align-items: start;
	}

	form {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px 14px;
	}

	label {
		display: block;
		font-size: 12px;
		color: var(--muted);
		margin-bottom: 3px;
	}

	select,
	input {
		width: 100%;
		padding: 6px 8px;
		border: 1px solid color-mix(in srgb, var(--fg) 25%, transparent);
		border-radius: 6px;
		background: color-mix(in srgb, var(--bg) 92%, var(--fg));
		color: var(--fg);
		font: inherit;
		font-size: 14px;
	}

	.row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 18px;
	}

	canvas {
		display: block;
		image-rendering: pixelated;
		border: 1px solid color-mix(in srgb, var(--fg) 18%, transparent);
		border-radius: 6px;
		background: var(--bg);
		width: 100%;
		height: auto;
	}

	.brain {
		margin-top: 18px;
		cursor: grab;
	}

	.cap {
		font-size: 12px;
		color: var(--muted);
		margin: 6px 0 0;
	}

	.sw {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 2px;
		margin-right: 2px;
	}
	.sw.blue {
		background: #3aa0ff;
	}
	.sw.orange {
		background: #f0a63c;
	}

	.result {
		margin-top: 18px;
		padding: 12px 14px;
		border-radius: 8px;
		display: flex;
		gap: 12px;
		align-items: center;
		background: color-mix(in srgb, var(--bg) 80%, var(--fg));
		border: 1px solid color-mix(in srgb, var(--fg) 15%, transparent);
	}
	.result.up {
		background: color-mix(in srgb, #e9f6ec 80%, var(--bg));
		border-color: #b7e0c0;
	}
	.result.down {
		background: color-mix(in srgb, #f7ecec 80%, var(--bg));
		border-color: #e6b8b8;
	}
	.arm {
		font-size: 26px;
	}
	.prob {
		font-size: 13px;
		color: var(--muted);
	}

	.lat {
		margin-top: 10px;
		font-size: 12px;
		color: var(--muted);
		font-variant-numeric: tabular-nums;
	}

	@media (max-width: 860px) {
		.wrap {
			grid-template-columns: 1fr;
		}
	}
</style>
